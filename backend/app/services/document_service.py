"""Document management service for spaces, folders, tags, and document organization.

Provides:
- Space CRUD with name uniqueness validation (1-50 chars)
- Folder CRUD with nesting limit (max 10 levels) and sibling uniqueness
- Folder tree query
- Tag management (1-20 tags per document, 1-30 chars each)
- Document filtering (by space/folder/tag, paginated, 20 per page)
- Document move (update space_id and/or folder_id)
- Cascade delete (space/folder deletion cascades to sub-folders and documents)
- Cascade index cleanup (Qdrant points + OpenSearch docs) when a document or its
  parent space is deleted, via :class:`IndexingService.delete_document_chunks`.
"""

import logging
import uuid

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    ConflictException,
    NotFoundException,
    ValidationException,
)
from app.models.document import Document
from app.models.document_tag import DocumentTag
from app.models.folder import Folder
from app.models.space import Space


logger = logging.getLogger(__name__)


class DocumentService:
    """Service for managing spaces, folders, tags, and document organization."""

    def __init__(self, db: AsyncSession):
        self.db = db

    # ─── Space CRUD ────────────────────────────────────────────────────

    async def create_space(
        self, name: str, description: str | None, created_by: uuid.UUID
    ) -> Space:
        """Create a new space with unique name validation.

        除业务层显式查重外，捕获数据库层 ``IntegrityError`` 以应对并发双写场景：
        即两个请求同时通过业务层去重检查，但只有一个能写入成功，另一个会触发
        SQL 唯一约束异常，需要被转换为业务 ``ConflictException``。

        创建后自动给 ``created_by`` 写入 ``write`` 级别的 Permission,
        让创建者立刻能在搜索 / RAG / 文档列表中看到自己创建的空间。
        """
        from app.models.permission import AccessLevel, Permission, ResourceType

        self._validate_space_name(name)
        await self._check_space_name_unique(name)

        space = Space(name=name, description=description, created_by=created_by)
        self.db.add(space)
        try:
            await self.db.flush()
        except IntegrityError as exc:
            # 并发场景：业务层去重已通过但底层唯一约束失败，回滚后抛出冲突
            await self.db.rollback()
            raise ConflictException(f"空间名称 '{name}' 已存在") from exc
        await self.db.refresh(space)

        # 给创建者播种 write 权限,使其能立刻访问
        owner_permission = Permission(
            resource_id=space.id,
            resource_type=ResourceType.space,
            user_id=created_by,
            access_level=AccessLevel.write,
        )
        self.db.add(owner_permission)
        await self.db.flush()

        return space

    async def list_spaces(self) -> list[Space]:
        """List all spaces."""
        stmt = select(Space).order_by(Space.created_at.desc())
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_space(self, space_id: uuid.UUID) -> Space:
        """Get a space by ID."""
        stmt = select(Space).where(Space.id == space_id)
        result = await self.db.execute(stmt)
        space = result.scalar_one_or_none()
        if not space:
            raise NotFoundException("Space", str(space_id))
        return space

    async def update_space(
        self, space_id: uuid.UUID, name: str | None = None, description: str | None = None
    ) -> Space:
        """Update a space's name and/or description."""
        space = await self.get_space(space_id)

        if name is not None:
            self._validate_space_name(name)
            if name != space.name:
                await self._check_space_name_unique(name)
            space.name = name

        if description is not None:
            space.description = description

        await self.db.flush()
        await self.db.refresh(space)
        return space

    async def delete_space(self, space_id: uuid.UUID) -> None:
        """Delete a space and cascade delete all folders and documents.

        Before the SQL-level cascade fires (``ON DELETE CASCADE`` on
        ``documents.space_id``), we collect every ``document_id`` in the
        space and ask :class:`IndexingService` to drop the matching
        chunks from Qdrant and OpenSearch. The cleanup runs **before**
        the row delete so that:

        - if Qdrant fails, ``IndexingError`` propagates and the
          transaction never commits — PostgreSQL stays consistent with
          the still-populated search backends and the operation can be
          safely retried;
        - if OpenSearch fails *after* Qdrant succeeded,
          ``delete_document_chunks`` returns ``opensearch_error`` instead
          of raising (partial-cleanup contract); we log it and proceed
          with the SQL delete because Qdrant is already clean and the
          surviving OpenSearch docs are filtered by ABAC and reaped
          out-of-band.
        """
        space = await self.get_space(space_id)

        # Collect document IDs *before* deleting the space; once the SQL
        # cascade fires we lose the ability to enumerate them.
        stmt = select(Document.id).where(Document.space_id == space_id)
        result = await self.db.execute(stmt)
        document_ids = [str(row[0]) for row in result.all()]

        # Cleanup search indices first — failure here aborts the cascade.
        if document_ids:
            self._cleanup_document_indices(document_ids)

        await self.db.delete(space)
        await self.db.flush()

    # ─── Folder CRUD ───────────────────────────────────────────────────

    async def create_folder(
        self, space_id: uuid.UUID, name: str, parent_id: uuid.UUID | None = None
    ) -> Folder:
        """Create a folder within a space, respecting nesting limits."""
        self._validate_folder_name(name)

        # Verify space exists
        await self.get_space(space_id)

        # Calculate depth
        depth = 1
        if parent_id:
            parent = await self._get_folder(parent_id)
            if parent.space_id != space_id:
                raise ValidationException("父目录不属于该空间")
            depth = parent.depth + 1

        # Check nesting limit
        if depth > 10:
            raise ValidationException("目录嵌套层级不能超过 10 级")

        # Check sibling uniqueness
        await self._check_folder_name_unique(space_id, parent_id, name)

        folder = Folder(space_id=space_id, parent_id=parent_id, name=name, depth=depth)
        self.db.add(folder)
        try:
            await self.db.flush()
        except IntegrityError as exc:
            # 并发双写：业务层去重已通过但 (space_id, parent_id, name) 唯一约束触发
            await self.db.rollback()
            raise ConflictException(f"同级目录下已存在名为 '{name}' 的目录") from exc
        await self.db.refresh(folder)
        return folder

    async def list_folders(
        self, space_id: uuid.UUID, parent_id: uuid.UUID | None = None
    ) -> list[Folder]:
        """List folders in a space, optionally filtered by parent."""
        await self.get_space(space_id)

        stmt = select(Folder).where(
            Folder.space_id == space_id,
            Folder.parent_id == parent_id,
        ).order_by(Folder.name)
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def delete_folder(self, folder_id: uuid.UUID) -> None:
        """Delete a folder and cascade delete sub-folders and documents.

        ``documents.folder_id`` is declared ``ON DELETE SET NULL``, so
        deleting a folder does **not** delete the documents that lived
        inside it — they simply detach to the space root. There is
        therefore no Qdrant/OpenSearch cleanup to perform here. Only
        ``delete_space`` and ``delete_document`` trigger index cleanup.
        """
        folder = await self._get_folder(folder_id)
        await self.db.delete(folder)
        await self.db.flush()

    async def delete_document(self, document_id: uuid.UUID) -> None:
        """Delete a single document and cascade-clean its search indices.

        Order of operations mirrors :meth:`delete_space`: index cleanup
        runs **before** the SQL delete so a Qdrant failure aborts the
        entire operation and keeps PostgreSQL in sync with the search
        backends. An OpenSearch-only failure (``opensearch_error`` set
        on the cleanup result) is logged but does not block the SQL
        delete — the partial cleanup is documented on
        :meth:`IndexingService.delete_document_chunks`.
        """
        document = await self._get_document(document_id)

        self._cleanup_document_indices([str(document.id)])

        await self.db.delete(document)
        await self.db.flush()

    def _cleanup_document_indices(self, document_ids: list[str]) -> None:
        """Drop Qdrant points and OpenSearch docs for ``document_ids``.

        Iterates one document at a time because both backends are
        keyed by ``document_id`` (Qdrant via payload filter, OpenSearch
        via ``term`` query) and the operation count is bounded by the
        space size — typically tens to hundreds — not worth the
        complexity of a multi-id batch API. Each call is independent
        so a failure on document N does not prevent already-cleaned
        documents 1..N-1 from staying clean.
        """
        # Imported lazily so unit tests for non-delete paths don't have
        # to provide Qdrant/OpenSearch test doubles, and so the import
        # cycle stays one-way (DocumentService → IndexingService).
        from app.services.indexing_service import IndexingService

        service = IndexingService()
        for doc_id in document_ids:
            res = service.delete_document_chunks(doc_id)
            # Partial-cleanup signal from delete_document_chunks: Qdrant
            # is clean but OpenSearch failed. Log and continue — the
            # PostgreSQL row is about to be removed and ABAC will hide
            # the residual OpenSearch docs from search results.
            if res.get("opensearch_error"):
                logger.warning(
                    "Partial index cleanup for document %s: %s",
                    doc_id,
                    res["opensearch_error"],
                )

    # ─── Folder Tree ───────────────────────────────────────────────────

    async def get_folder_tree(self, space_id: uuid.UUID) -> list[dict]:
        """Get the full folder tree for a space."""
        await self.get_space(space_id)

        stmt = select(Folder).where(Folder.space_id == space_id).order_by(Folder.name)
        result = await self.db.execute(stmt)
        folders = list(result.scalars().all())

        return self._build_tree(folders)

    # ─── Tag Management ────────────────────────────────────────────────

    async def add_tag(self, document_id: uuid.UUID, tag_name: str) -> DocumentTag:
        """Add a tag to a document."""
        self._validate_tag_name(tag_name)

        # Verify document exists
        document = await self._get_document(document_id)

        # Check tag count limit
        tag_count = await self._get_document_tag_count(document_id)
        if tag_count >= 20:
            raise ValidationException("每个文档最多添加 20 个标签")

        # Check if tag already exists
        existing = await self._get_tag(document_id, tag_name)
        if existing:
            raise ConflictException(f"标签 '{tag_name}' 已存在")

        tag = DocumentTag(document_id=document_id, tag_name=tag_name)
        self.db.add(tag)
        try:
            await self.db.flush()
        except IntegrityError as exc:
            # 并发双写：业务层去重通过但 (document_id, tag_name) 唯一约束触发
            await self.db.rollback()
            raise ConflictException(f"标签 '{tag_name}' 已存在") from exc
        await self.db.refresh(tag)
        return tag

    async def remove_tag(self, document_id: uuid.UUID, tag_name: str) -> None:
        """Remove a tag from a document."""
        tag = await self._get_tag(document_id, tag_name)
        if not tag:
            raise NotFoundException("Tag", tag_name)
        await self.db.delete(tag)
        await self.db.flush()

    async def list_tags(self) -> list[str]:
        """List all unique tag names in the system."""
        stmt = select(DocumentTag.tag_name).distinct().order_by(DocumentTag.tag_name)
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    # ─── Document Filtering ────────────────────────────────────────────

    async def list_documents(
        self,
        space_id: uuid.UUID | None = None,
        folder_id: uuid.UUID | None = None,
        tag: str | None = None,
        page: int = 1,
        page_size: int = 20,
        allowed_space_ids: list[uuid.UUID] | None = None,
    ) -> dict:
        """List documents with filtering and pagination.

        当 ``allowed_space_ids`` 不为 ``None`` 时, 强制按集合过滤
        (用于非 admin 的权限隔离); 为 ``None`` 表示不限制 (admin)。
        """
        stmt = select(Document)
        count_stmt = select(func.count(Document.id))

        # Apply filters
        if space_id:
            stmt = stmt.where(Document.space_id == space_id)
            count_stmt = count_stmt.where(Document.space_id == space_id)
        if allowed_space_ids is not None:
            stmt = stmt.where(Document.space_id.in_(allowed_space_ids))
            count_stmt = count_stmt.where(Document.space_id.in_(allowed_space_ids))
        if folder_id:
            stmt = stmt.where(Document.folder_id == folder_id)
            count_stmt = count_stmt.where(Document.folder_id == folder_id)
        if tag:
            stmt = stmt.join(DocumentTag).where(DocumentTag.tag_name == tag)
            count_stmt = count_stmt.join(DocumentTag).where(DocumentTag.tag_name == tag)

        # Get total count
        total_result = await self.db.execute(count_stmt)
        total = total_result.scalar() or 0

        # Apply pagination
        offset = (page - 1) * page_size
        stmt = stmt.order_by(Document.created_at.desc()).offset(offset).limit(page_size)

        result = await self.db.execute(stmt)
        documents = list(result.scalars().all())

        return {
            "items": documents,
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": (total + page_size - 1) // page_size if total > 0 else 0,
        }

    # ─── Document Move ─────────────────────────────────────────────────

    async def move_document(
        self,
        document_id: uuid.UUID,
        target_space_id: uuid.UUID | None = None,
        target_folder_id: uuid.UUID | None = None,
    ) -> Document:
        """Move a document to a different space and/or folder."""
        document = await self._get_document(document_id)

        if target_space_id:
            # Verify target space exists
            await self.get_space(target_space_id)
            document.space_id = target_space_id

        if target_folder_id:
            # Verify target folder exists and belongs to the correct space
            folder = await self._get_folder(target_folder_id)
            effective_space_id = target_space_id or document.space_id
            if folder.space_id != effective_space_id:
                raise ValidationException("目标目录不属于目标空间")
            document.folder_id = target_folder_id
        elif target_space_id:
            # Moving to a new space without specifying folder clears folder_id
            document.folder_id = None

        await self.db.flush()
        await self.db.refresh(document)
        return document

    # ─── Private Helpers ───────────────────────────────────────────────

    def _validate_space_name(self, name: str) -> None:
        """Validate space name: 1-50 characters."""
        if not name or len(name.strip()) == 0:
            raise ValidationException("空间名称不能为空")
        if len(name) > 50:
            raise ValidationException("空间名称长度不能超过 50 个字符")

    def _validate_folder_name(self, name: str) -> None:
        """Validate folder name: 1-50 characters."""
        if not name or len(name.strip()) == 0:
            raise ValidationException("目录名称不能为空")
        if len(name) > 50:
            raise ValidationException("目录名称长度不能超过 50 个字符")

    def _validate_tag_name(self, tag_name: str) -> None:
        """Validate tag name: 1-30 characters."""
        if not tag_name or len(tag_name.strip()) == 0:
            raise ValidationException("标签名称不能为空")
        if len(tag_name) > 30:
            raise ValidationException("标签名称长度不能超过 30 个字符")

    async def _check_space_name_unique(self, name: str) -> None:
        """Check that space name is unique in the system."""
        stmt = select(Space).where(Space.name == name)
        result = await self.db.execute(stmt)
        if result.scalar_one_or_none():
            raise ConflictException(f"空间名称 '{name}' 已存在")

    async def _check_folder_name_unique(
        self, space_id: uuid.UUID, parent_id: uuid.UUID | None, name: str
    ) -> None:
        """Check that folder name is unique within the same parent."""
        stmt = select(Folder).where(
            Folder.space_id == space_id,
            Folder.parent_id == parent_id,
            Folder.name == name,
        )
        result = await self.db.execute(stmt)
        if result.scalar_one_or_none():
            raise ConflictException(f"同级目录下已存在名为 '{name}' 的目录")

    async def _get_folder(self, folder_id: uuid.UUID) -> Folder:
        """Get a folder by ID or raise NotFoundException."""
        stmt = select(Folder).where(Folder.id == folder_id)
        result = await self.db.execute(stmt)
        folder = result.scalar_one_or_none()
        if not folder:
            raise NotFoundException("Folder", str(folder_id))
        return folder

    async def _get_document(self, document_id: uuid.UUID) -> Document:
        """Get a document by ID or raise NotFoundException."""
        stmt = select(Document).where(Document.id == document_id)
        result = await self.db.execute(stmt)
        document = result.scalar_one_or_none()
        if not document:
            raise NotFoundException("Document", str(document_id))
        return document

    async def _get_document_tag_count(self, document_id: uuid.UUID) -> int:
        """Get the number of tags on a document."""
        stmt = select(func.count(DocumentTag.id)).where(
            DocumentTag.document_id == document_id
        )
        result = await self.db.execute(stmt)
        return result.scalar() or 0

    async def _get_tag(
        self, document_id: uuid.UUID, tag_name: str
    ) -> DocumentTag | None:
        """Get a specific tag on a document."""
        stmt = select(DocumentTag).where(
            DocumentTag.document_id == document_id,
            DocumentTag.tag_name == tag_name,
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    def _build_tree(self, folders: list[Folder]) -> list[dict]:
        """Build a tree structure from a flat list of folders."""
        folder_map: dict[uuid.UUID, dict] = {}
        roots: list[dict] = []

        # Create nodes
        for folder in folders:
            folder_map[folder.id] = {
                "id": str(folder.id),
                "name": folder.name,
                "depth": folder.depth,
                "parent_id": str(folder.parent_id) if folder.parent_id else None,
                "children": [],
            }

        # Build tree
        for folder in folders:
            node = folder_map[folder.id]
            if folder.parent_id and folder.parent_id in folder_map:
                folder_map[folder.parent_id]["children"].append(node)
            else:
                roots.append(node)

        return roots
