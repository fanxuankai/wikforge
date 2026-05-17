"""Unit tests for document management service (spaces, folders, tags, documents).

Tests cover:
- Space CRUD with name uniqueness validation (1-50 chars)
- Folder CRUD with nesting limit (max 10 levels) and sibling uniqueness
- Folder tree query
- Tag management (1-20 tags per document, 1-30 chars each)
- Document filtering (by space/folder/tag, paginated)
- Document move
- Cascade delete logic
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.exceptions import (
    ConflictException,
    NotFoundException,
    ValidationException,
)
from app.models.document import Document, DocumentStatus
from app.models.document_tag import DocumentTag
from app.models.folder import Folder
from app.models.space import Space
from app.services.document_service import DocumentService


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def mock_db():
    """Create a mock async database session."""
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.delete = AsyncMock()
    db.refresh = AsyncMock()
    return db


@pytest.fixture
def service(mock_db):
    """Create a DocumentService instance with mocked DB."""
    return DocumentService(db=mock_db)


# ─── Helper Functions ──────────────────────────────────────────────────


def make_space(
    name: str = "Test Space",
    description: str | None = None,
    created_by: uuid.UUID | None = None,
) -> MagicMock:
    """Create a mock Space object."""
    space = MagicMock(spec=Space)
    space.id = uuid.uuid4()
    space.name = name
    space.description = description
    space.created_by = created_by or uuid.uuid4()
    space.created_at = datetime.now(timezone.utc)
    space.updated_at = datetime.now(timezone.utc)
    return space


def make_folder(
    name: str = "Test Folder",
    space_id: uuid.UUID | None = None,
    parent_id: uuid.UUID | None = None,
    depth: int = 1,
) -> MagicMock:
    """Create a mock Folder object."""
    folder = MagicMock(spec=Folder)
    folder.id = uuid.uuid4()
    folder.space_id = space_id or uuid.uuid4()
    folder.parent_id = parent_id
    folder.name = name
    folder.depth = depth
    folder.created_at = datetime.now(timezone.utc)
    return folder


def make_document(
    title: str = "Test Doc",
    space_id: uuid.UUID | None = None,
    folder_id: uuid.UUID | None = None,
) -> MagicMock:
    """Create a mock Document object."""
    doc = MagicMock(spec=Document)
    doc.id = uuid.uuid4()
    doc.space_id = space_id or uuid.uuid4()
    doc.folder_id = folder_id
    doc.title = title
    doc.file_type = "pdf"
    doc.file_size = 1024
    doc.storage_path = "/test/path.pdf"
    doc.status = DocumentStatus.pending
    doc.created_at = datetime.now(timezone.utc)
    doc.updated_at = datetime.now(timezone.utc)
    return doc


def make_tag(document_id: uuid.UUID, tag_name: str) -> MagicMock:
    """Create a mock DocumentTag object."""
    tag = MagicMock(spec=DocumentTag)
    tag.id = uuid.uuid4()
    tag.document_id = document_id
    tag.tag_name = tag_name
    return tag


# ─── Space CRUD Tests ──────────────────────────────────────────────────


class TestSpaceCRUD:
    """Tests for space create, read, update, delete operations."""

    @pytest.mark.asyncio
    async def test_create_space_success(self, service, mock_db):
        """Successfully create a space with valid name."""
        user_id = uuid.uuid4()

        # No existing space with same name
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        space = await service.create_space(
            name="My Space", description="A test space", created_by=user_id
        )

        # create_space 现在播种 Space + owner Permission, 共 2 次 add
        assert mock_db.add.call_count == 2
        # flush 也会被调 2 次 (Space 一次, Permission 一次)
        assert mock_db.flush.call_count >= 1

    @pytest.mark.asyncio
    async def test_create_space_duplicate_name(self, service, mock_db):
        """Creating a space with duplicate name raises ConflictException."""
        user_id = uuid.uuid4()
        existing_space = make_space(name="Existing")

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_space
        mock_db.execute = AsyncMock(return_value=mock_result)

        with pytest.raises(ConflictException, match="已存在"):
            await service.create_space(
                name="Existing", description=None, created_by=user_id
            )

    @pytest.mark.asyncio
    async def test_create_space_empty_name(self, service, mock_db):
        """Creating a space with empty name raises ValidationException."""
        user_id = uuid.uuid4()

        with pytest.raises(ValidationException, match="不能为空"):
            await service.create_space(name="", description=None, created_by=user_id)

    @pytest.mark.asyncio
    async def test_create_space_name_too_long(self, service, mock_db):
        """Creating a space with name > 50 chars raises ValidationException."""
        user_id = uuid.uuid4()

        with pytest.raises(ValidationException, match="50"):
            await service.create_space(
                name="x" * 51, description=None, created_by=user_id
            )

    @pytest.mark.asyncio
    async def test_get_space_not_found(self, service, mock_db):
        """Getting a non-existent space raises NotFoundException."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        with pytest.raises(NotFoundException):
            await service.get_space(uuid.uuid4())

    @pytest.mark.asyncio
    async def test_update_space_name(self, service, mock_db):
        """Successfully update a space name."""
        space = make_space(name="Old Name")

        # First call: get_space, second call: check uniqueness
        get_result = MagicMock()
        get_result.scalar_one_or_none.return_value = space
        unique_result = MagicMock()
        unique_result.scalar_one_or_none.return_value = None

        mock_db.execute = AsyncMock(side_effect=[get_result, unique_result])

        updated = await service.update_space(space.id, name="New Name")
        assert space.name == "New Name"

    @pytest.mark.asyncio
    async def test_create_space_concurrent_integrity_error(self, service, mock_db):
        """业务层去重通过但底层唯一约束触发时，转为 ConflictException。"""
        from sqlalchemy.exc import IntegrityError

        user_id = uuid.uuid4()

        # 第一次查询：业务层去重未发现重名
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        # flush 时模拟并发竞争抛出 IntegrityError
        mock_db.flush = AsyncMock(
            side_effect=IntegrityError("INSERT", {}, Exception("unique violation"))
        )

        with pytest.raises(ConflictException, match="已存在"):
            await service.create_space(
                name="Race", description=None, created_by=user_id
            )
        mock_db.rollback.assert_awaited()


# ─── Space List Tests ──────────────────────────────────────────────────


class TestSpaceList:
    """Tests for listing spaces."""

    @pytest.mark.asyncio
    async def test_list_spaces_empty(self, service, mock_db):
        """无空间时返回空列表。"""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db.execute = AsyncMock(return_value=mock_result)

        spaces = await service.list_spaces()
        assert spaces == []

    @pytest.mark.asyncio
    async def test_list_spaces_returns_all(self, service, mock_db):
        """返回所有空间。"""
        spaces = [make_space(name=f"S{i}") for i in range(3)]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = spaces
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await service.list_spaces()
        assert len(result) == 3
        assert {s.name for s in result} == {"S0", "S1", "S2"}


# ─── Folder CRUD Tests ─────────────────────────────────────────────────


class TestFolderCRUD:
    """Tests for folder create, list, delete operations."""

    @pytest.mark.asyncio
    async def test_create_folder_root_level(self, service, mock_db):
        """Create a root-level folder (no parent)."""
        space = make_space()

        # get_space, check_folder_name_unique
        space_result = MagicMock()
        space_result.scalar_one_or_none.return_value = space
        unique_result = MagicMock()
        unique_result.scalar_one_or_none.return_value = None

        mock_db.execute = AsyncMock(side_effect=[space_result, unique_result])

        folder = await service.create_folder(
            space_id=space.id, name="Root Folder", parent_id=None
        )

        mock_db.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_folder_with_parent(self, service, mock_db):
        """Create a folder with a parent folder."""
        space = make_space()
        parent = make_folder(space_id=space.id, depth=3)

        # get_space, get_folder (parent), check_folder_name_unique
        space_result = MagicMock()
        space_result.scalar_one_or_none.return_value = space
        parent_result = MagicMock()
        parent_result.scalar_one_or_none.return_value = parent
        unique_result = MagicMock()
        unique_result.scalar_one_or_none.return_value = None

        mock_db.execute = AsyncMock(
            side_effect=[space_result, parent_result, unique_result]
        )

        folder = await service.create_folder(
            space_id=space.id, name="Child Folder", parent_id=parent.id
        )

        mock_db.add.assert_called_once()
        # Verify the added folder has correct depth
        added_folder = mock_db.add.call_args[0][0]
        assert added_folder.depth == 4

    @pytest.mark.asyncio
    async def test_create_folder_exceeds_nesting_limit(self, service, mock_db):
        """Creating a folder at depth > 10 raises ValidationException."""
        space = make_space()
        parent = make_folder(space_id=space.id, depth=10)

        space_result = MagicMock()
        space_result.scalar_one_or_none.return_value = space
        parent_result = MagicMock()
        parent_result.scalar_one_or_none.return_value = parent

        mock_db.execute = AsyncMock(side_effect=[space_result, parent_result])

        with pytest.raises(ValidationException, match="10 级"):
            await service.create_folder(
                space_id=space.id, name="Too Deep", parent_id=parent.id
            )

    @pytest.mark.asyncio
    async def test_create_folder_duplicate_sibling_name(self, service, mock_db):
        """Creating a folder with duplicate name in same parent raises ConflictException."""
        space = make_space()
        existing_folder = make_folder(name="Existing", space_id=space.id)

        space_result = MagicMock()
        space_result.scalar_one_or_none.return_value = space
        unique_result = MagicMock()
        unique_result.scalar_one_or_none.return_value = existing_folder

        mock_db.execute = AsyncMock(side_effect=[space_result, unique_result])

        with pytest.raises(ConflictException, match="已存在"):
            await service.create_folder(
                space_id=space.id, name="Existing", parent_id=None
            )

    @pytest.mark.asyncio
    async def test_create_folder_empty_name(self, service, mock_db):
        """Creating a folder with empty name raises ValidationException."""
        with pytest.raises(ValidationException, match="不能为空"):
            await service.create_folder(
                space_id=uuid.uuid4(), name="", parent_id=None
            )

    @pytest.mark.asyncio
    async def test_create_folder_name_too_long(self, service, mock_db):
        """Creating a folder with name > 50 chars raises ValidationException."""
        with pytest.raises(ValidationException, match="50"):
            await service.create_folder(
                space_id=uuid.uuid4(), name="x" * 51, parent_id=None
            )

    @pytest.mark.asyncio
    async def test_create_folder_parent_wrong_space(self, service, mock_db):
        """Creating a folder with parent in different space raises ValidationException."""
        space = make_space()
        parent = make_folder(space_id=uuid.uuid4(), depth=1)  # Different space

        space_result = MagicMock()
        space_result.scalar_one_or_none.return_value = space
        parent_result = MagicMock()
        parent_result.scalar_one_or_none.return_value = parent

        mock_db.execute = AsyncMock(side_effect=[space_result, parent_result])

        with pytest.raises(ValidationException, match="不属于该空间"):
            await service.create_folder(
                space_id=space.id, name="Child", parent_id=parent.id
            )

    @pytest.mark.asyncio
    async def test_delete_folder(self, service, mock_db):
        """Successfully delete a folder."""
        folder = make_folder()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = folder
        mock_db.execute = AsyncMock(return_value=mock_result)

        await service.delete_folder(folder.id)
        mock_db.delete.assert_called_once_with(folder)


# ─── Folder Tree Tests ─────────────────────────────────────────────────


class TestFolderTree:
    """Tests for folder tree query."""

    @pytest.mark.asyncio
    async def test_get_folder_tree_empty(self, service, mock_db):
        """Get tree for a space with no folders returns empty list."""
        space = make_space()

        space_result = MagicMock()
        space_result.scalar_one_or_none.return_value = space
        folders_result = MagicMock()
        folders_result.scalars.return_value.all.return_value = []

        mock_db.execute = AsyncMock(side_effect=[space_result, folders_result])

        tree = await service.get_folder_tree(space.id)
        assert tree == []

    @pytest.mark.asyncio
    async def test_get_folder_tree_nested(self, service, mock_db):
        """Get tree builds correct parent-child relationships."""
        space = make_space()
        root = make_folder(name="Root", space_id=space.id, depth=1)
        root.parent_id = None
        child = make_folder(name="Child", space_id=space.id, parent_id=root.id, depth=2)

        space_result = MagicMock()
        space_result.scalar_one_or_none.return_value = space
        folders_result = MagicMock()
        folders_result.scalars.return_value.all.return_value = [root, child]

        mock_db.execute = AsyncMock(side_effect=[space_result, folders_result])

        tree = await service.get_folder_tree(space.id)
        assert len(tree) == 1
        assert tree[0]["name"] == "Root"
        assert len(tree[0]["children"]) == 1
        assert tree[0]["children"][0]["name"] == "Child"


# ─── Tag Management Tests ──────────────────────────────────────────────


class TestTagManagement:
    """Tests for tag add/remove operations."""

    @pytest.mark.asyncio
    async def test_add_tag_success(self, service, mock_db):
        """Successfully add a tag to a document."""
        doc = make_document()

        # get_document, get_tag_count, get_existing_tag
        doc_result = MagicMock()
        doc_result.scalar_one_or_none.return_value = doc
        count_result = MagicMock()
        count_result.scalar.return_value = 5
        existing_result = MagicMock()
        existing_result.scalar_one_or_none.return_value = None

        mock_db.execute = AsyncMock(
            side_effect=[doc_result, count_result, existing_result]
        )

        tag = await service.add_tag(document_id=doc.id, tag_name="important")
        mock_db.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_tag_exceeds_limit(self, service, mock_db):
        """Adding a tag when document already has 20 tags raises ValidationException."""
        doc = make_document()

        doc_result = MagicMock()
        doc_result.scalar_one_or_none.return_value = doc
        count_result = MagicMock()
        count_result.scalar.return_value = 20

        mock_db.execute = AsyncMock(side_effect=[doc_result, count_result])

        with pytest.raises(ValidationException, match="20"):
            await service.add_tag(document_id=doc.id, tag_name="overflow")

    @pytest.mark.asyncio
    async def test_add_tag_duplicate(self, service, mock_db):
        """Adding a duplicate tag raises ConflictException."""
        doc = make_document()
        existing_tag = make_tag(doc.id, "existing")

        doc_result = MagicMock()
        doc_result.scalar_one_or_none.return_value = doc
        count_result = MagicMock()
        count_result.scalar.return_value = 5
        existing_result = MagicMock()
        existing_result.scalar_one_or_none.return_value = existing_tag

        mock_db.execute = AsyncMock(
            side_effect=[doc_result, count_result, existing_result]
        )

        with pytest.raises(ConflictException, match="已存在"):
            await service.add_tag(document_id=doc.id, tag_name="existing")

    @pytest.mark.asyncio
    async def test_add_tag_empty_name(self, service, mock_db):
        """Adding a tag with empty name raises ValidationException."""
        with pytest.raises(ValidationException, match="不能为空"):
            await service.add_tag(document_id=uuid.uuid4(), tag_name="")

    @pytest.mark.asyncio
    async def test_add_tag_name_too_long(self, service, mock_db):
        """Adding a tag with name > 30 chars raises ValidationException."""
        with pytest.raises(ValidationException, match="30"):
            await service.add_tag(document_id=uuid.uuid4(), tag_name="x" * 31)

    @pytest.mark.asyncio
    async def test_remove_tag_success(self, service, mock_db):
        """Successfully remove a tag from a document."""
        doc_id = uuid.uuid4()
        tag = make_tag(doc_id, "removeme")

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = tag
        mock_db.execute = AsyncMock(return_value=mock_result)

        await service.remove_tag(document_id=doc_id, tag_name="removeme")
        mock_db.delete.assert_called_once_with(tag)

    @pytest.mark.asyncio
    async def test_remove_tag_not_found(self, service, mock_db):
        """Removing a non-existent tag raises NotFoundException."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        with pytest.raises(NotFoundException):
            await service.remove_tag(document_id=uuid.uuid4(), tag_name="nonexistent")


# ─── Document Filtering Tests ──────────────────────────────────────────


class TestDocumentFiltering:
    """Tests for document listing with filters and pagination."""

    @pytest.mark.asyncio
    async def test_list_documents_default_pagination(self, service, mock_db):
        """List documents returns paginated results with defaults."""
        docs = [make_document() for _ in range(3)]

        count_result = MagicMock()
        count_result.scalar.return_value = 3
        docs_result = MagicMock()
        docs_result.scalars.return_value.all.return_value = docs

        mock_db.execute = AsyncMock(side_effect=[count_result, docs_result])

        result = await service.list_documents()
        assert result["total"] == 3
        assert result["page"] == 1
        assert result["page_size"] == 20
        assert len(result["items"]) == 3

    @pytest.mark.asyncio
    async def test_list_documents_with_space_filter(self, service, mock_db):
        """List documents filtered by space_id."""
        space_id = uuid.uuid4()
        docs = [make_document(space_id=space_id)]

        count_result = MagicMock()
        count_result.scalar.return_value = 1
        docs_result = MagicMock()
        docs_result.scalars.return_value.all.return_value = docs

        mock_db.execute = AsyncMock(side_effect=[count_result, docs_result])

        result = await service.list_documents(space_id=space_id)
        assert result["total"] == 1

    @pytest.mark.asyncio
    async def test_list_documents_pagination_calculation(self, service, mock_db):
        """Pagination correctly calculates total pages."""
        count_result = MagicMock()
        count_result.scalar.return_value = 45
        docs_result = MagicMock()
        docs_result.scalars.return_value.all.return_value = []

        mock_db.execute = AsyncMock(side_effect=[count_result, docs_result])

        result = await service.list_documents(page=1, page_size=20)
        assert result["pages"] == 3  # ceil(45/20) = 3


# ─── Document Move Tests ───────────────────────────────────────────────


class TestDocumentMove:
    """Tests for document move operation."""

    @pytest.mark.asyncio
    async def test_move_document_to_different_space(self, service, mock_db):
        """Move a document to a different space."""
        doc = make_document()
        target_space = make_space()

        # get_document, get_space (target)
        doc_result = MagicMock()
        doc_result.scalar_one_or_none.return_value = doc
        space_result = MagicMock()
        space_result.scalar_one_or_none.return_value = target_space

        mock_db.execute = AsyncMock(side_effect=[doc_result, space_result])

        result = await service.move_document(
            document_id=doc.id, target_space_id=target_space.id
        )
        assert doc.space_id == target_space.id
        assert doc.folder_id is None  # Cleared when moving to new space

    @pytest.mark.asyncio
    async def test_move_document_to_folder(self, service, mock_db):
        """Move a document to a specific folder."""
        space_id = uuid.uuid4()
        doc = make_document(space_id=space_id)
        target_folder = make_folder(space_id=space_id)

        # get_document, get_folder (target)
        doc_result = MagicMock()
        doc_result.scalar_one_or_none.return_value = doc
        folder_result = MagicMock()
        folder_result.scalar_one_or_none.return_value = target_folder

        mock_db.execute = AsyncMock(side_effect=[doc_result, folder_result])

        result = await service.move_document(
            document_id=doc.id, target_folder_id=target_folder.id
        )
        assert doc.folder_id == target_folder.id

    @pytest.mark.asyncio
    async def test_move_document_folder_wrong_space(self, service, mock_db):
        """Moving document to folder in different space raises ValidationException."""
        doc = make_document(space_id=uuid.uuid4())
        target_folder = make_folder(space_id=uuid.uuid4())  # Different space

        doc_result = MagicMock()
        doc_result.scalar_one_or_none.return_value = doc
        folder_result = MagicMock()
        folder_result.scalar_one_or_none.return_value = target_folder

        mock_db.execute = AsyncMock(side_effect=[doc_result, folder_result])

        with pytest.raises(ValidationException, match="不属于目标空间"):
            await service.move_document(
                document_id=doc.id, target_folder_id=target_folder.id
            )

    @pytest.mark.asyncio
    async def test_move_document_not_found(self, service, mock_db):
        """Moving a non-existent document raises NotFoundException."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        with pytest.raises(NotFoundException):
            await service.move_document(
                document_id=uuid.uuid4(), target_space_id=uuid.uuid4()
            )


# ─── Cascade Delete Tests ──────────────────────────────────────────────


class TestCascadeDelete:
    """Tests for cascade delete behavior."""

    @pytest.mark.asyncio
    async def test_delete_space_calls_db_delete(self, service, mock_db):
        """Deleting a space calls db.delete (cascade handled by SQLAlchemy)."""
        space = make_space()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = space
        mock_db.execute = AsyncMock(return_value=mock_result)

        await service.delete_space(space.id)
        mock_db.delete.assert_called_once_with(space)
        mock_db.flush.assert_called()

    @pytest.mark.asyncio
    async def test_delete_folder_calls_db_delete(self, service, mock_db):
        """Deleting a folder calls db.delete (cascade handled by SQLAlchemy)."""
        folder = make_folder()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = folder
        mock_db.execute = AsyncMock(return_value=mock_result)

        await service.delete_folder(folder.id)
        mock_db.delete.assert_called_once_with(folder)
        mock_db.flush.assert_called()


# ─── Validation Tests ──────────────────────────────────────────────────


class TestValidation:
    """Tests for input validation edge cases."""

    @pytest.mark.asyncio
    async def test_space_name_whitespace_only(self, service, mock_db):
        """Space name with only whitespace raises ValidationException."""
        with pytest.raises(ValidationException, match="不能为空"):
            await service.create_space(
                name="   ", description=None, created_by=uuid.uuid4()
            )

    @pytest.mark.asyncio
    async def test_folder_name_whitespace_only(self, service, mock_db):
        """Folder name with only whitespace raises ValidationException."""
        with pytest.raises(ValidationException, match="不能为空"):
            await service.create_folder(
                space_id=uuid.uuid4(), name="   ", parent_id=None
            )

    @pytest.mark.asyncio
    async def test_tag_name_whitespace_only(self, service, mock_db):
        """Tag name with only whitespace raises ValidationException."""
        with pytest.raises(ValidationException, match="不能为空"):
            await service.add_tag(document_id=uuid.uuid4(), tag_name="   ")

    @pytest.mark.asyncio
    async def test_space_name_exactly_50_chars(self, service, mock_db):
        """Space name with exactly 50 chars is valid."""
        name = "x" * 50
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        # Should not raise
        await service.create_space(name=name, description=None, created_by=uuid.uuid4())
        # Space + owner Permission = 2 次 add
        assert mock_db.add.call_count == 2

    @pytest.mark.asyncio
    async def test_tag_name_exactly_30_chars(self, service, mock_db):
        """Tag name with exactly 30 chars is valid."""
        doc = make_document()
        tag_name = "x" * 30

        doc_result = MagicMock()
        doc_result.scalar_one_or_none.return_value = doc
        count_result = MagicMock()
        count_result.scalar.return_value = 0
        existing_result = MagicMock()
        existing_result.scalar_one_or_none.return_value = None

        mock_db.execute = AsyncMock(
            side_effect=[doc_result, count_result, existing_result]
        )

        # Should not raise
        await service.add_tag(document_id=doc.id, tag_name=tag_name)
        mock_db.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_folder_depth_exactly_10(self, service, mock_db):
        """Creating a folder at exactly depth 10 is valid."""
        space = make_space()
        parent = make_folder(space_id=space.id, depth=9)

        space_result = MagicMock()
        space_result.scalar_one_or_none.return_value = space
        parent_result = MagicMock()
        parent_result.scalar_one_or_none.return_value = parent
        unique_result = MagicMock()
        unique_result.scalar_one_or_none.return_value = None

        mock_db.execute = AsyncMock(
            side_effect=[space_result, parent_result, unique_result]
        )

        # Should not raise - depth 10 is the max allowed
        await service.create_folder(
            space_id=space.id, name="Level 10", parent_id=parent.id
        )
        added_folder = mock_db.add.call_args[0][0]
        assert added_folder.depth == 10


# ─── Folder List & Tree (multi-level) Tests ────────────────────────────


class TestFolderList:
    """Tests for listing folders within a space."""

    @pytest.mark.asyncio
    async def test_list_folders_filters_by_parent_none(self, service, mock_db):
        """parent_id 为空时仅返回根目录。"""
        space = make_space()
        roots = [make_folder(name="A", space_id=space.id, depth=1)]

        space_result = MagicMock()
        space_result.scalar_one_or_none.return_value = space
        folder_result = MagicMock()
        folder_result.scalars.return_value.all.return_value = roots

        mock_db.execute = AsyncMock(side_effect=[space_result, folder_result])

        result = await service.list_folders(space.id, parent_id=None)
        assert len(result) == 1
        assert result[0].name == "A"

    @pytest.mark.asyncio
    async def test_list_folders_filters_by_parent_id(self, service, mock_db):
        """传入 parent_id 时返回该父目录下的子目录。"""
        space = make_space()
        parent_id = uuid.uuid4()
        children = [
            make_folder(name="C1", space_id=space.id, parent_id=parent_id, depth=2),
            make_folder(name="C2", space_id=space.id, parent_id=parent_id, depth=2),
        ]

        space_result = MagicMock()
        space_result.scalar_one_or_none.return_value = space
        folder_result = MagicMock()
        folder_result.scalars.return_value.all.return_value = children

        mock_db.execute = AsyncMock(side_effect=[space_result, folder_result])

        result = await service.list_folders(space.id, parent_id=parent_id)
        assert len(result) == 2


class TestFolderTreeMultiLevel:
    """更深层级的目录树构建测试。"""

    @pytest.mark.asyncio
    async def test_tree_three_levels(self, service, mock_db):
        """三层嵌套结构构建正确，根列表只包含 parent 为空的节点。"""
        space = make_space()
        root_a = make_folder(name="A", space_id=space.id, depth=1)
        root_a.parent_id = None
        root_b = make_folder(name="B", space_id=space.id, depth=1)
        root_b.parent_id = None
        child_a1 = make_folder(
            name="A-1", space_id=space.id, parent_id=root_a.id, depth=2
        )
        child_a2 = make_folder(
            name="A-2", space_id=space.id, parent_id=root_a.id, depth=2
        )
        grand = make_folder(
            name="A-1-x", space_id=space.id, parent_id=child_a1.id, depth=3
        )

        space_result = MagicMock()
        space_result.scalar_one_or_none.return_value = space
        folder_result = MagicMock()
        folder_result.scalars.return_value.all.return_value = [
            root_a,
            root_b,
            child_a1,
            child_a2,
            grand,
        ]

        mock_db.execute = AsyncMock(side_effect=[space_result, folder_result])

        tree = await service.get_folder_tree(space.id)

        # 根节点恰好 2 个，且都没有 parent_id
        assert len(tree) == 2
        assert {n["name"] for n in tree} == {"A", "B"}
        for n in tree:
            assert n["parent_id"] is None

        node_a = next(n for n in tree if n["name"] == "A")
        assert {c["name"] for c in node_a["children"]} == {"A-1", "A-2"}
        node_a1 = next(c for c in node_a["children"] if c["name"] == "A-1")
        assert len(node_a1["children"]) == 1
        assert node_a1["children"][0]["name"] == "A-1-x"
        assert node_a1["children"][0]["depth"] == 3


# ─── Folder Concurrency Integrity ──────────────────────────────────────


class TestFolderConcurrency:
    @pytest.mark.asyncio
    async def test_create_folder_integrity_error(self, service, mock_db):
        """flush 时 (space_id, parent_id, name) 唯一约束触发 → ConflictException。"""
        from sqlalchemy.exc import IntegrityError

        space = make_space()
        space_result = MagicMock()
        space_result.scalar_one_or_none.return_value = space
        unique_result = MagicMock()
        unique_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(side_effect=[space_result, unique_result])
        mock_db.flush = AsyncMock(
            side_effect=IntegrityError("INSERT", {}, Exception("unique"))
        )

        with pytest.raises(ConflictException, match="已存在"):
            await service.create_folder(
                space_id=space.id, name="Race", parent_id=None
            )
        mock_db.rollback.assert_awaited()


# ─── Tag list & concurrency ────────────────────────────────────────────


class TestTagList:
    @pytest.mark.asyncio
    async def test_list_tags_returns_distinct_sorted(self, service, mock_db):
        """list_tags 返回去重后按字母排序的标签集合。"""
        # 注意：service 内部已经 distinct + order_by；这里只断言传递性。
        tags = ["alpha", "beta", "gamma"]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = tags
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await service.list_tags()
        assert result == tags


class TestTagConcurrency:
    @pytest.mark.asyncio
    async def test_add_tag_integrity_error(self, service, mock_db):
        """flush 时 (document_id, tag_name) 唯一约束触发 → ConflictException。"""
        from sqlalchemy.exc import IntegrityError

        doc = make_document()
        doc_result = MagicMock()
        doc_result.scalar_one_or_none.return_value = doc
        count_result = MagicMock()
        count_result.scalar.return_value = 0
        existing_result = MagicMock()
        existing_result.scalar_one_or_none.return_value = None

        mock_db.execute = AsyncMock(
            side_effect=[doc_result, count_result, existing_result]
        )
        mock_db.flush = AsyncMock(
            side_effect=IntegrityError("INSERT", {}, Exception("unique"))
        )

        with pytest.raises(ConflictException, match="已存在"):
            await service.add_tag(document_id=doc.id, tag_name="dup")
        mock_db.rollback.assert_awaited()


# ─── Document Filtering 扩展 ───────────────────────────────────────────


class TestDocumentFilteringExtra:
    """补齐 folder 筛选、tag 筛选、空集与越界场景。"""

    @pytest.mark.asyncio
    async def test_list_documents_filter_by_folder(self, service, mock_db):
        folder_id = uuid.uuid4()
        docs = [make_document(folder_id=folder_id)]

        count_result = MagicMock()
        count_result.scalar.return_value = 1
        docs_result = MagicMock()
        docs_result.scalars.return_value.all.return_value = docs

        mock_db.execute = AsyncMock(side_effect=[count_result, docs_result])

        result = await service.list_documents(folder_id=folder_id)
        assert result["total"] == 1
        assert len(result["items"]) == 1

    @pytest.mark.asyncio
    async def test_list_documents_filter_by_tag(self, service, mock_db):
        docs = [make_document() for _ in range(2)]

        count_result = MagicMock()
        count_result.scalar.return_value = 2
        docs_result = MagicMock()
        docs_result.scalars.return_value.all.return_value = docs

        mock_db.execute = AsyncMock(side_effect=[count_result, docs_result])

        result = await service.list_documents(tag="important")
        assert result["total"] == 2
        assert result["pages"] == 1

    @pytest.mark.asyncio
    async def test_list_documents_empty_set(self, service, mock_db):
        """无匹配结果时 total=0 且 pages=0。"""
        count_result = MagicMock()
        count_result.scalar.return_value = 0
        docs_result = MagicMock()
        docs_result.scalars.return_value.all.return_value = []

        mock_db.execute = AsyncMock(side_effect=[count_result, docs_result])

        result = await service.list_documents()
        assert result["total"] == 0
        assert result["pages"] == 0
        assert result["items"] == []

    @pytest.mark.asyncio
    async def test_list_documents_page_out_of_range(self, service, mock_db):
        """越界 page 返回空 items 但 total 与 pages 正确。"""
        count_result = MagicMock()
        count_result.scalar.return_value = 5
        docs_result = MagicMock()
        docs_result.scalars.return_value.all.return_value = []

        mock_db.execute = AsyncMock(side_effect=[count_result, docs_result])

        result = await service.list_documents(page=10, page_size=20)
        assert result["total"] == 5
        assert result["pages"] == 1  # ceil(5/20) = 1
        assert result["page"] == 10
        assert result["items"] == []

    @pytest.mark.asyncio
    async def test_list_documents_combined_filters(self, service, mock_db):
        """同时按 space + folder + tag 三种条件过滤。"""
        space_id = uuid.uuid4()
        folder_id = uuid.uuid4()
        docs = [make_document(space_id=space_id, folder_id=folder_id)]

        count_result = MagicMock()
        count_result.scalar.return_value = 1
        docs_result = MagicMock()
        docs_result.scalars.return_value.all.return_value = docs

        mock_db.execute = AsyncMock(side_effect=[count_result, docs_result])

        result = await service.list_documents(
            space_id=space_id, folder_id=folder_id, tag="t"
        )
        assert result["total"] == 1


# ─── Document Move 扩展 ────────────────────────────────────────────────


class TestDocumentMoveExtra:
    """补齐目标目录不存在、跨空间携带目录等场景。"""

    @pytest.mark.asyncio
    async def test_move_target_folder_not_found(self, service, mock_db):
        """目标目录不存在 → NotFoundException。"""
        doc = make_document()
        doc_result = MagicMock()
        doc_result.scalar_one_or_none.return_value = doc
        folder_result = MagicMock()
        folder_result.scalar_one_or_none.return_value = None

        mock_db.execute = AsyncMock(side_effect=[doc_result, folder_result])

        with pytest.raises(NotFoundException):
            await service.move_document(
                document_id=doc.id, target_folder_id=uuid.uuid4()
            )

    @pytest.mark.asyncio
    async def test_move_to_target_space_with_folder(self, service, mock_db):
        """同时指定 target_space_id 与 target_folder_id 时，目录必须属于目标空间。"""
        target_space = make_space()
        doc = make_document(space_id=uuid.uuid4())  # 原空间
        target_folder = make_folder(space_id=target_space.id)

        doc_result = MagicMock()
        doc_result.scalar_one_or_none.return_value = doc
        space_result = MagicMock()
        space_result.scalar_one_or_none.return_value = target_space
        folder_result = MagicMock()
        folder_result.scalar_one_or_none.return_value = target_folder

        mock_db.execute = AsyncMock(
            side_effect=[doc_result, space_result, folder_result]
        )

        result = await service.move_document(
            document_id=doc.id,
            target_space_id=target_space.id,
            target_folder_id=target_folder.id,
        )
        assert doc.space_id == target_space.id
        assert doc.folder_id == target_folder.id

    @pytest.mark.asyncio
    async def test_move_only_space_clears_folder(self, service, mock_db):
        """仅传 target_space_id 时清空 folder_id。"""
        original_folder_id = uuid.uuid4()
        doc = make_document(folder_id=original_folder_id)
        target_space = make_space()

        doc_result = MagicMock()
        doc_result.scalar_one_or_none.return_value = doc
        space_result = MagicMock()
        space_result.scalar_one_or_none.return_value = target_space

        mock_db.execute = AsyncMock(side_effect=[doc_result, space_result])

        await service.move_document(
            document_id=doc.id, target_space_id=target_space.id
        )
        assert doc.space_id == target_space.id
        assert doc.folder_id is None


# ─── 属性测试 (Hypothesis) ─────────────────────────────────────────────


class TestTagPropertyBased:
    """属性测试：标签名称长度边界与每个文档 ≤20 个标签的硬性约束。

    Validates: Requirements 2.3, 2.7
    """

    def test_property_tag_name_length_boundary(self, mock_db):
        """长度 1..30 的标签必须通过校验，>30 必须拒绝。"""
        from hypothesis import given, settings as hyp_settings, strategies as st

        @hyp_settings(max_examples=80, deadline=None)
        @given(
            st.text(
                alphabet=st.characters(
                    min_codepoint=0x4E00,
                    max_codepoint=0x9FFF,  # 中文常用字
                )
                | st.characters(
                    min_codepoint=33, max_codepoint=126, blacklist_categories=("Cs",)
                ),
                min_size=1,
                max_size=60,
            )
        )
        def prop(name: str) -> None:
            # 排除全空白：业务上视为空标签
            if not name.strip():
                return
            service = DocumentService(db=AsyncMock())
            length = len(name)
            if length <= 30:
                # 不应抛 ValidationException（长度合法）
                service._validate_tag_name(name)
            else:
                with pytest.raises(ValidationException):
                    service._validate_tag_name(name)

        prop()

    @pytest.mark.asyncio
    async def test_property_tag_count_limit(self):
        """文档当前标签数 0..19 时新增成功；>=20 时新增必拒。

        以参数化形式遍历关键边界值（hypothesis 异步迭代兼容性较差，
        这里使用确定性边界覆盖：0、1、19、20、21、30）。
        """
        for current_count in [0, 1, 19, 20, 21, 30]:
            db = AsyncMock()
            db.add = MagicMock()
            db.flush = AsyncMock()
            db.refresh = AsyncMock()
            db.delete = AsyncMock()
            db.rollback = AsyncMock()

            service = DocumentService(db=db)
            doc = make_document()
            doc_result = MagicMock()
            doc_result.scalar_one_or_none.return_value = doc
            count_result = MagicMock()
            count_result.scalar.return_value = current_count
            existing_result = MagicMock()
            existing_result.scalar_one_or_none.return_value = None

            db.execute = AsyncMock(
                side_effect=[doc_result, count_result, existing_result]
            )

            if current_count < 20:
                await service.add_tag(document_id=doc.id, tag_name="tag")
                db.add.assert_called_once()
            else:
                with pytest.raises(ValidationException, match="20"):
                    await service.add_tag(document_id=doc.id, tag_name="tag")


class TestFolderDepthPropertyBased:
    """属性测试：父目录 depth 1..9 时创建子目录合法（最终深度 2..10）；depth ≥ 10 必拒。

    Validates: Requirements 2.2

    采用边界值参数化（hypothesis 异步迭代兼容性差，确定性覆盖关键边界）。
    """

    @pytest.mark.asyncio
    async def test_property_folder_depth_boundary(self):
        for parent_depth in [1, 5, 9, 10, 11, 15]:
            db = AsyncMock()
            db.add = MagicMock()
            db.flush = AsyncMock()
            db.refresh = AsyncMock()
            db.delete = AsyncMock()
            db.rollback = AsyncMock()

            service = DocumentService(db=db)
            space = make_space()
            parent = make_folder(space_id=space.id, depth=parent_depth)

            space_result = MagicMock()
            space_result.scalar_one_or_none.return_value = space
            parent_result = MagicMock()
            parent_result.scalar_one_or_none.return_value = parent
            unique_result = MagicMock()
            unique_result.scalar_one_or_none.return_value = None
            db.execute = AsyncMock(
                side_effect=[space_result, parent_result, unique_result]
            )

            if parent_depth + 1 <= 10:
                await service.create_folder(
                    space_id=space.id, name="child", parent_id=parent.id
                )
                added = db.add.call_args[0][0]
                assert added.depth == parent_depth + 1
            else:
                with pytest.raises(ValidationException, match="10 级"):
                    await service.create_folder(
                        space_id=space.id, name="child", parent_id=parent.id
                    )
