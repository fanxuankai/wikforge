"""Document management API routes for spaces, folders, tags, and documents.

Endpoints:
- POST/GET/PUT/DELETE /api/spaces - Space CRUD
- POST/GET /api/spaces/{id}/folders - Folder CRUD within a space
- GET /api/spaces/{id}/tree - Folder tree
- DELETE /api/folders/{id} - Delete folder
- POST/DELETE /api/documents/{id}/tags - Tag management
- GET /api/tags - List all tags
- GET /api/documents - Document listing with filters
- PATCH /api/documents/{id}/move - Move document
- POST /api/documents/upload - Upload files
- POST /api/documents/import-url - Import from URL
- GET /api/documents/{id}/progress - Get processing progress
- POST /api/documents/{id}/retry - Retry failed document
"""

import uuid
from typing import List

from fastapi import APIRouter, Depends, File, Query, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user, is_admin_user
from app.core.database import get_db
from app.models.permission import AccessLevel, Permission, ResourceType
from app.models.user import User
from app.services.document_service import DocumentService
from app.services.upload_service import UploadService


async def _get_user_allowed_space_ids(
    user: User, db: AsyncSession
) -> set[uuid.UUID]:
    """返回用户具有 read/write 权限的空间 ID 集合。

    Admin (邮箱与 INITIAL_ADMIN_EMAIL 匹配) 返回 None 表示「全部空间」,
    调用方按 None 短路, 不做过滤。
    """
    if is_admin_user(user):
        return None  # type: ignore[return-value]
    stmt = select(Permission.resource_id).where(
        Permission.user_id == user.id,
        Permission.resource_type == ResourceType.space,
        Permission.access_level.in_([AccessLevel.read, AccessLevel.write]),
    )
    result = await db.execute(stmt)
    return {sid for sid in result.scalars().all()}

router = APIRouter(tags=["documents"])


# ─── Request/Response Schemas ──────────────────────────────────────────


class CreateSpaceRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    description: str | None = Field(None, max_length=200)


class UpdateSpaceRequest(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=50)
    description: str | None = Field(None, max_length=200)


class SpaceResponse(BaseModel):
    id: str
    name: str
    description: str | None
    created_by: str
    created_at: str
    updated_at: str

    model_config = {"from_attributes": True}


class CreateFolderRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    parent_id: str | None = None


class FolderResponse(BaseModel):
    id: str
    space_id: str
    parent_id: str | None
    name: str
    depth: int
    created_at: str

    model_config = {"from_attributes": True}


class FolderTreeNode(BaseModel):
    id: str
    name: str
    depth: int
    parent_id: str | None
    children: list["FolderTreeNode"] = []


class AddTagRequest(BaseModel):
    tag_name: str = Field(..., min_length=1, max_length=30)


class TagResponse(BaseModel):
    id: str
    document_id: str
    tag_name: str

    model_config = {"from_attributes": True}


class MoveDocumentRequest(BaseModel):
    target_space_id: str | None = None
    target_folder_id: str | None = None


class DocumentResponse(BaseModel):
    id: str
    space_id: str
    folder_id: str | None
    title: str
    file_type: str
    file_size: int
    status: str
    created_at: str
    updated_at: str

    model_config = {"from_attributes": True}


class PaginatedDocumentsResponse(BaseModel):
    items: list[DocumentResponse]
    total: int
    page: int
    page_size: int
    pages: int


class ImportUrlRequest(BaseModel):
    url: str = Field(..., min_length=1)
    space_id: str
    folder_id: str | None = None


class UploadDocumentResponse(BaseModel):
    id: str
    title: str
    file_type: str
    file_size: int
    storage_path: str
    status: str
    created_at: str

    model_config = {"from_attributes": True}


class DocumentProgressResponse(BaseModel):
    document_id: str
    stage: str
    progress: int
    updated_at: str | None


# ─── Dependencies ──────────────────────────────────────────────────────


async def get_document_service(
    db: AsyncSession = Depends(get_db),
) -> DocumentService:
    """Dependency to get DocumentService instance."""
    return DocumentService(db=db)


async def get_upload_service(
    db: AsyncSession = Depends(get_db),
) -> UploadService:
    """Dependency to get UploadService instance."""
    return UploadService(db=db)


# ─── Space Endpoints ───────────────────────────────────────────────────


@router.post("/api/spaces", response_model=SpaceResponse, status_code=201)
async def create_space(
    body: CreateSpaceRequest,
    current_user: User = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
):
    """Create a new space."""
    space = await service.create_space(
        name=body.name,
        description=body.description,
        created_by=current_user.id,
    )
    return _space_to_response(space)


@router.get("/api/spaces", response_model=list[SpaceResponse])
async def list_spaces(
    current_user: User = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
    db: AsyncSession = Depends(get_db),
):
    """List spaces the current user can access (admin: all)."""
    allowed = await _get_user_allowed_space_ids(current_user, db)
    spaces = await service.list_spaces()
    if allowed is not None:
        spaces = [s for s in spaces if s.id in allowed]
    return [_space_to_response(s) for s in spaces]


@router.put("/api/spaces/{space_id}", response_model=SpaceResponse)
async def update_space(
    space_id: uuid.UUID,
    body: UpdateSpaceRequest,
    current_user: User = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
):
    """Update a space."""
    space = await service.update_space(
        space_id=space_id,
        name=body.name,
        description=body.description,
    )
    return _space_to_response(space)


@router.delete("/api/spaces/{space_id}", status_code=204)
async def delete_space(
    space_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
):
    """Delete a space and all its contents (cascade)."""
    await service.delete_space(space_id)


# ─── Folder Endpoints ──────────────────────────────────────────────────


@router.post(
    "/api/spaces/{space_id}/folders", response_model=FolderResponse, status_code=201
)
async def create_folder(
    space_id: uuid.UUID,
    body: CreateFolderRequest,
    current_user: User = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
):
    """Create a folder within a space."""
    parent_id = uuid.UUID(body.parent_id) if body.parent_id else None
    folder = await service.create_folder(
        space_id=space_id,
        name=body.name,
        parent_id=parent_id,
    )
    return _folder_to_response(folder)


@router.get("/api/spaces/{space_id}/folders", response_model=list[FolderResponse])
async def list_folders(
    space_id: uuid.UUID,
    parent_id: uuid.UUID | None = Query(None),
    current_user: User = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
    db: AsyncSession = Depends(get_db),
):
    """List folders in a space, optionally filtered by parent.

    会校验用户对 space 是否有权限, 否则返回 403。
    """
    allowed = await _get_user_allowed_space_ids(current_user, db)
    if allowed is not None and space_id not in allowed:
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail="无权访问该空间")
    folders = await service.list_folders(space_id=space_id, parent_id=parent_id)
    return [_folder_to_response(f) for f in folders]


@router.get("/api/spaces/{space_id}/tree", response_model=list[FolderTreeNode])
async def get_folder_tree(
    space_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
):
    """Get the full folder tree for a space."""
    tree = await service.get_folder_tree(space_id)
    return tree


@router.delete("/api/folders/{folder_id}", status_code=204)
async def delete_folder(
    folder_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
):
    """Delete a folder and all its contents (cascade)."""
    await service.delete_folder(folder_id)


# ─── Tag Endpoints ─────────────────────────────────────────────────────


@router.post("/api/documents/{document_id}/tags", response_model=TagResponse, status_code=201)
async def add_tag(
    document_id: uuid.UUID,
    body: AddTagRequest,
    current_user: User = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
):
    """Add a tag to a document."""
    tag = await service.add_tag(document_id=document_id, tag_name=body.tag_name)
    return _tag_to_response(tag)


@router.delete("/api/documents/{document_id}/tags/{tag_name}", status_code=204)
async def remove_tag(
    document_id: uuid.UUID,
    tag_name: str,
    current_user: User = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
):
    """Remove a tag from a document."""
    await service.remove_tag(document_id=document_id, tag_name=tag_name)


@router.get("/api/tags", response_model=list[str])
async def list_tags(
    current_user: User = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
):
    """List all unique tags in the system."""
    return await service.list_tags()


# ─── Document Endpoints ────────────────────────────────────────────────


@router.get("/api/documents", response_model=PaginatedDocumentsResponse)
async def list_documents(
    space_id: uuid.UUID | None = Query(None),
    folder_id: uuid.UUID | None = Query(None),
    tag: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
    db: AsyncSession = Depends(get_db),
):
    """List documents with optional filtering and pagination.

    会强制按用户可访问空间过滤; admin 看全部。
    """
    allowed = await _get_user_allowed_space_ids(current_user, db)
    if allowed is not None:
        # 如果用户指定了 space_id 但没权限,直接返回空
        if space_id is not None and space_id not in allowed:
            return PaginatedDocumentsResponse(
                items=[], total=0, page=page, page_size=page_size, pages=0
            )
        # 没权限访问任何空间 -> 空列表
        if not allowed:
            return PaginatedDocumentsResponse(
                items=[], total=0, page=page, page_size=page_size, pages=0
            )

    result = await service.list_documents(
        space_id=space_id,
        folder_id=folder_id,
        tag=tag,
        page=page,
        page_size=page_size,
        allowed_space_ids=list(allowed) if allowed is not None else None,
    )
    return PaginatedDocumentsResponse(
        items=[_document_to_response(d) for d in result["items"]],
        total=result["total"],
        page=result["page"],
        page_size=result["page_size"],
        pages=result["pages"],
    )


@router.patch("/api/documents/{document_id}/move", response_model=DocumentResponse)
async def move_document(
    document_id: uuid.UUID,
    body: MoveDocumentRequest,
    current_user: User = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
):
    """Move a document to a different space and/or folder."""
    target_space_id = uuid.UUID(body.target_space_id) if body.target_space_id else None
    target_folder_id = uuid.UUID(body.target_folder_id) if body.target_folder_id else None

    document = await service.move_document(
        document_id=document_id,
        target_space_id=target_space_id,
        target_folder_id=target_folder_id,
    )
    return _document_to_response(document)


# ─── Upload Endpoints ──────────────────────────────────────────────────


@router.post("/api/documents/upload", response_model=list[UploadDocumentResponse], status_code=201)
async def upload_documents(
    files: List[UploadFile] = File(...),
    space_id: str = Query(...),
    folder_id: str | None = Query(None),
    current_user: User = Depends(get_current_user),
    service: UploadService = Depends(get_upload_service),
):
    """Upload multiple documents (max 50 files, max 100MB each).

    Supported formats: PDF, DOCX, PPTX, TXT, MD, HTML.
    """
    space_uuid = uuid.UUID(space_id)
    folder_uuid = uuid.UUID(folder_id) if folder_id else None

    documents = await service.upload_files(
        files=files,
        space_id=space_uuid,
        folder_id=folder_uuid,
        uploaded_by=current_user.id,
    )
    return [_upload_doc_to_response(d) for d in documents]


@router.post("/api/documents/import-url", response_model=UploadDocumentResponse, status_code=201)
async def import_url(
    body: ImportUrlRequest,
    current_user: User = Depends(get_current_user),
    service: UploadService = Depends(get_upload_service),
):
    """Import a document from a URL (30 second timeout, trafilatura extraction)."""
    space_uuid = uuid.UUID(body.space_id)
    folder_uuid = uuid.UUID(body.folder_id) if body.folder_id else None

    document = await service.import_url(
        url=body.url,
        space_id=space_uuid,
        folder_id=folder_uuid,
        uploaded_by=current_user.id,
    )
    return _upload_doc_to_response(document)


@router.get("/api/documents/{document_id}/progress", response_model=DocumentProgressResponse)
async def get_document_progress(
    document_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: UploadService = Depends(get_upload_service),
):
    """Get document processing progress (cached in Redis, updated every 5 seconds)."""
    progress = await service.get_document_progress(document_id)
    return DocumentProgressResponse(**progress)


@router.post("/api/documents/{document_id}/retry", response_model=UploadDocumentResponse)
async def retry_document(
    document_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: UploadService = Depends(get_upload_service),
):
    """Retry processing a failed document (resets status, re-enqueues).

    Max 3 retries allowed. After 3 failures, document is marked as permanent failure.
    """
    document = await service.retry_document(document_id)
    return _upload_doc_to_response(document)


# ─── Response Helpers ──────────────────────────────────────────────────


def _space_to_response(space) -> SpaceResponse:
    return SpaceResponse(
        id=str(space.id),
        name=space.name,
        description=space.description,
        created_by=str(space.created_by),
        created_at=space.created_at.isoformat(),
        updated_at=space.updated_at.isoformat(),
    )


def _folder_to_response(folder) -> FolderResponse:
    return FolderResponse(
        id=str(folder.id),
        space_id=str(folder.space_id),
        parent_id=str(folder.parent_id) if folder.parent_id else None,
        name=folder.name,
        depth=folder.depth,
        created_at=folder.created_at.isoformat(),
    )


def _tag_to_response(tag) -> TagResponse:
    return TagResponse(
        id=str(tag.id),
        document_id=str(tag.document_id),
        tag_name=tag.tag_name,
    )


def _document_to_response(document) -> DocumentResponse:
    return DocumentResponse(
        id=str(document.id),
        space_id=str(document.space_id),
        folder_id=str(document.folder_id) if document.folder_id else None,
        title=document.title,
        file_type=document.file_type,
        file_size=document.file_size,
        status=document.status.value if hasattr(document.status, "value") else document.status,
        created_at=document.created_at.isoformat(),
        updated_at=document.updated_at.isoformat(),
    )


def _upload_doc_to_response(document) -> UploadDocumentResponse:
    return UploadDocumentResponse(
        id=str(document.id),
        title=document.title,
        file_type=document.file_type,
        file_size=document.file_size,
        storage_path=document.storage_path,
        status=document.status.value if hasattr(document.status, "value") else document.status,
        created_at=document.created_at.isoformat(),
    )


# ─── Document Delete Endpoints ─────────────────────────────────────────


class BatchDeleteRequest(BaseModel):
    """批量删除请求体: 可按 ids 删除, 或按 status 过滤删除 (推荐 'failed' 清理)。"""

    ids: list[str] | None = None
    status: str | None = None  # 仅接受单个 status 字符串, 如 "failed"


class BatchDeleteResponse(BaseModel):
    deleted: int
    ids: list[str]


@router.delete("/api/documents/{document_id}", status_code=204)
async def delete_document(
    document_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
    db: AsyncSession = Depends(get_db),
):
    """单个文档删除 (会清理 MinIO / Qdrant / OpenSearch 中的关联数据)。"""
    # 权限校验: 用户必须对这个文档所在的 space 有访问权
    allowed = await _get_user_allowed_space_ids(current_user, db)
    if allowed is not None:
        # 读 document 的 space_id 校验权限
        from fastapi import HTTPException
        from app.models.document import Document
        doc_row = (
            await db.execute(select(Document.space_id).where(Document.id == document_id))
        ).scalar_one_or_none()
        if doc_row is None:
            raise HTTPException(status_code=404, detail="文档不存在")
        if doc_row not in allowed:
            raise HTTPException(status_code=403, detail="无权访问该文档")

    await service.delete_document(document_id=document_id)
    return None


@router.post("/api/documents/batch-delete", response_model=BatchDeleteResponse)
async def batch_delete_documents(
    body: BatchDeleteRequest,
    current_user: User = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
    db: AsyncSession = Depends(get_db),
) -> BatchDeleteResponse:
    """批量删除文档 (常用于一键清理 status='failed' 的僵尸文档)。

    传 ``ids`` (优先): 按 UUID 列表精确删除。
    传 ``status``: 删除该状态下当前用户能访问的所有文档 (admin 可访问全部)。
    两者必须二选一传。
    """
    from fastapi import HTTPException
    from app.models.document import Document

    if not body.ids and not body.status:
        raise HTTPException(status_code=400, detail="ids 或 status 必须至少传一个")
    if body.ids and body.status:
        raise HTTPException(status_code=400, detail="ids 与 status 不能同时使用")

    allowed = await _get_user_allowed_space_ids(current_user, db)

    # 根据条件解析出待删除 id 列表
    target_ids: list[uuid.UUID] = []
    if body.ids:
        for s in body.ids:
            try:
                target_ids.append(uuid.UUID(s))
            except ValueError:
                raise HTTPException(status_code=400, detail=f"非法 UUID: {s}")
        # 校验权限: 必须全部都在 allowed 范围内
        if allowed is not None:
            stmt = select(Document.id, Document.space_id).where(Document.id.in_(target_ids))
            rows = (await db.execute(stmt)).all()
            for row_id, row_space in rows:
                if row_space not in allowed:
                    raise HTTPException(
                        status_code=403, detail=f"无权删除文档 {row_id}"
                    )
    else:
        # by status
        stmt = select(Document.id).where(Document.status == body.status)
        if allowed is not None:
            stmt = stmt.where(Document.space_id.in_(allowed))
        target_ids = list((await db.execute(stmt)).scalars().all())

    deleted_ids: list[str] = []
    for doc_id in target_ids:
        try:
            await service.delete_document(document_id=doc_id)
            deleted_ids.append(str(doc_id))
        except Exception:  # noqa: BLE001 — 某一条失败不阻塞其它
            continue

    return BatchDeleteResponse(deleted=len(deleted_ids), ids=deleted_ids)


# ─── Document Download (presigned URL) ─────────────────────────────────


class DocumentDownloadResponse(BaseModel):
    """文档下载响应: 返回 presigned URL 让前端直接从 MinIO 拉文件,
    避免后端做反向代理 / 暴露内部 storage_path。
    """

    url: str
    expires_in: int


@router.get(
    "/api/documents/{document_id}/download-url",
    response_model=DocumentDownloadResponse,
)
async def get_document_download_url(
    document_id: uuid.UUID,
    expires_in: int = Query(600, ge=60, le=86400),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DocumentDownloadResponse:
    """生成 MinIO 预签名 GET URL, 默认 10 分钟 TTL。

    权限校验: 用户必须对文档所在 space 有访问权 (admin 全通)。
    返回的 URL 直接指向 MinIO, 前端可直接 <a href> 或 <iframe>。
    """
    from fastapi import HTTPException
    from app.core.minio import generate_presigned_get_url
    from app.models.document import Document

    doc = (
        await db.execute(
            select(Document.storage_path, Document.space_id, Document.title).where(
                Document.id == document_id
            )
        )
    ).one_or_none()
    if doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")

    storage_path, space_id, _ = doc

    allowed = await _get_user_allowed_space_ids(current_user, db)
    if allowed is not None and space_id not in allowed:
        raise HTTPException(status_code=403, detail="无权访问该文档")

    if not storage_path:
        raise HTTPException(status_code=404, detail="文档存储路径丢失")

    url = generate_presigned_get_url(storage_path, expires_in=expires_in)
    if not url:
        raise HTTPException(status_code=500, detail="生成下载链接失败")

    return DocumentDownloadResponse(url=url, expires_in=expires_in)
