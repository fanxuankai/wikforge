"""Upload service for document file upload, URL import, status management, and retry logic.

Provides:
- File upload to MinIO with format/size validation
- URL import via trafilatura with 30s timeout
- Document status state machine management
- Progress tracking via Redis
- Retry logic with permanent failure marking (max 3 retries)
"""

import io
import logging
import time
import uuid
from datetime import datetime, timezone

import trafilatura
from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.exceptions import NotFoundException, ValidationException
from app.core.minio import ensure_bucket_exists, get_minio_client
from app.core.redis import get_redis
from app.models.document import Document, DocumentStatus

# 注: 任务管线入口在 services 层调用,而非 api 层,
# 是为了让 import_url / upload_files / 重处理流程都共用统一的"落库 + 入队"语义。
try:
    from app.tasks.pipeline import submit_pipeline as _submit_pipeline
except Exception:  # pragma: no cover — Celery 不可用时 (单元测试) 不触发任务
    _submit_pipeline = None

settings = get_settings()
logger = logging.getLogger(__name__)

# Supported file formats and their extensions
SUPPORTED_FORMATS = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "txt": "text/plain",
    "md": "text/markdown",
    "html": "text/html",
}

SUPPORTED_EXTENSIONS = set(SUPPORTED_FORMATS.keys())

# Max file size: 100MB
MAX_FILE_SIZE = 100 * 1024 * 1024

# Max batch upload: 50 files
MAX_BATCH_SIZE = 50

# URL import timeout: 30 seconds
URL_IMPORT_TIMEOUT = 30

# Max retry count before permanent failure
MAX_RETRY_COUNT = 3

# Redis key prefix for document status
REDIS_STATUS_PREFIX = "doc:status:"
REDIS_STATUS_TTL = 3600  # 1 hour

# Valid status transitions (state machine)
VALID_TRANSITIONS: dict[DocumentStatus, list[DocumentStatus]] = {
    DocumentStatus.pending: [DocumentStatus.parsing, DocumentStatus.failed],
    DocumentStatus.parsing: [DocumentStatus.cleaning, DocumentStatus.failed],
    DocumentStatus.cleaning: [DocumentStatus.chunking, DocumentStatus.failed],
    DocumentStatus.chunking: [DocumentStatus.embedding, DocumentStatus.failed],
    DocumentStatus.embedding: [DocumentStatus.indexing, DocumentStatus.failed],
    DocumentStatus.indexing: [DocumentStatus.completed, DocumentStatus.failed],
    DocumentStatus.completed: [],
    DocumentStatus.failed: [DocumentStatus.pending],  # retry resets to pending
}


class UploadService:
    """Service for document upload, URL import, status management, and retry."""

    def __init__(self, db: AsyncSession):
        self.db = db

    # ─── File Upload ───────────────────────────────────────────────────

    async def upload_files(
        self,
        files: list[UploadFile],
        space_id: uuid.UUID,
        folder_id: uuid.UUID | None,
        uploaded_by: uuid.UUID,
    ) -> list[Document]:
        """Upload multiple files to MinIO and create document records.

        Args:
            files: List of uploaded files (max 50)
            space_id: Target space ID
            folder_id: Optional target folder ID
            uploaded_by: User ID who uploaded

        Returns:
            List of created Document records

        Raises:
            ValidationException: If batch size, format, or size validation fails
        """
        if len(files) > MAX_BATCH_SIZE:
            raise ValidationException(
                f"批量上传最多支持 {MAX_BATCH_SIZE} 个文件，当前 {len(files)} 个"
            )

        if len(files) == 0:
            raise ValidationException("请至少上传一个文件")

        start_time = time.perf_counter()
        logger.info(
            "upload_files started: count=%d space_id=%s folder_id=%s uploaded_by=%s",
            len(files),
            space_id,
            folder_id,
            uploaded_by,
        )

        documents = []
        for file in files:
            doc = await self._upload_single_file(
                file=file,
                space_id=space_id,
                folder_id=folder_id,
                uploaded_by=uploaded_by,
            )
            documents.append(doc)

        # 落库后提交,保证 worker 拉到的 document 一定能从 DB 读到
        await self.db.commit()
        logger.info(
            "upload_files committed: count=%d document_ids=%s elapsed_ms=%d",
            len(documents),
            [str(doc.id) for doc in documents],
            int((time.perf_counter() - start_time) * 1000),
        )

        # 触发 Celery 处理管线 (parse → profile_match → process → chunk → embed → index)
        if _submit_pipeline is not None:
            for doc in documents:
                try:
                    _submit_pipeline(str(doc.id))
                    logger.info(
                        "upload_files pipeline submitted: document_id=%s",
                        doc.id,
                    )
                except Exception:
                    # 入队失败不阻塞上传响应：文件与 DB 记录已经保存，继续返回
                    # document_id 便于用户或管理员稍后手动重试 pending 文档。
                    logger.warning(
                        "upload_files pipeline submit failed: document_id=%s",
                        doc.id,
                        exc_info=True,
                    )
        else:
            logger.warning(
                "upload_files pipeline skipped: celery submitter unavailable count=%d",
                len(documents),
            )

        return documents

    async def _upload_single_file(
        self,
        file: UploadFile,
        space_id: uuid.UUID,
        folder_id: uuid.UUID | None,
        uploaded_by: uuid.UUID,
    ) -> Document:
        """Upload a single file: validate, store to MinIO, create DB record."""
        # Validate file format
        file_ext = self._get_file_extension(file.filename or "")
        if file_ext not in SUPPORTED_EXTENSIONS:
            raise ValidationException(
                f"不支持的文件格式: {file_ext}。"
                f"支持的格式: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        # Read file content
        content = await file.read()
        file_size = len(content)

        # Validate file size
        if file_size > MAX_FILE_SIZE:
            raise ValidationException(
                f"文件 '{file.filename}' 大小 ({file_size / 1024 / 1024:.1f}MB) "
                f"超出限制，最大允许 100MB"
            )

        if file_size == 0:
            raise ValidationException(f"文件 '{file.filename}' 为空文件")

        # Generate storage path
        storage_path = self._generate_storage_path(space_id, file.filename or "unknown")

        # Upload to MinIO
        self._store_to_minio(content, storage_path, file_ext)

        # Create document record
        document = Document(
            space_id=space_id,
            folder_id=folder_id,
            title=file.filename or "untitled",
            file_type=file_ext,
            file_size=file_size,
            storage_path=storage_path,
            status=DocumentStatus.pending,
            retry_count=0,
            uploaded_by=uploaded_by,
        )
        self.db.add(document)
        await self.db.flush()
        await self.db.refresh(document)

        # Initialize Redis status
        await self._init_redis_status(document.id)

        logger.info(
            "upload_file stored: document_id=%s space_id=%s file_type=%s file_size=%d",
            document.id,
            space_id,
            file_ext,
            file_size,
        )

        return document

    # ─── URL Import ────────────────────────────────────────────────────

    async def import_url(
        self,
        url: str,
        space_id: uuid.UUID,
        folder_id: uuid.UUID | None,
        uploaded_by: uuid.UUID,
    ) -> Document:
        """Import a document from a URL using trafilatura.

        Args:
            url: The URL to fetch content from
            space_id: Target space ID
            folder_id: Optional target folder ID
            uploaded_by: User ID who initiated import

        Returns:
            Created Document record

        Raises:
            ValidationException: If URL is invalid or content cannot be fetched
        """
        if not url or not url.strip():
            raise ValidationException("URL 不能为空")

        if not url.startswith(("http://", "https://")):
            raise ValidationException("URL 必须以 http:// 或 https:// 开头")

        start_time = time.perf_counter()
        logger.info(
            "import_url started: space_id=%s folder_id=%s uploaded_by=%s",
            space_id,
            folder_id,
            uploaded_by,
        )

        # Fetch content with trafilatura (30s timeout)
        try:
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                raise ValidationException(
                    f"无法访问 URL: {url}，请检查 URL 是否正确"
                )

            content = trafilatura.extract(downloaded)
            if not content or not content.strip():
                raise ValidationException(
                    f"无法从 URL 提取有效内容: {url}"
                )
        except ValidationException:
            raise
        except Exception as e:
            raise ValidationException(
                f"URL 导入失败: {url}，原因: {str(e)}"
            )

        # Store extracted content as HTML file in MinIO
        content_bytes = content.encode("utf-8")
        file_size = len(content_bytes)

        # Generate a title from URL
        title = self._extract_title_from_url(url)

        # Generate storage path
        storage_path = self._generate_storage_path(space_id, f"{title}.html")

        # Upload to MinIO
        self._store_to_minio(content_bytes, storage_path, "html")

        # Create document record
        document = Document(
            space_id=space_id,
            folder_id=folder_id,
            title=title,
            file_type="html",
            file_size=file_size,
            storage_path=storage_path,
            status=DocumentStatus.pending,
            retry_count=0,
            uploaded_by=uploaded_by,
        )
        self.db.add(document)
        await self.db.flush()
        await self.db.refresh(document)

        # Initialize Redis status
        await self._init_redis_status(document.id)

        # commit + 触发管线 (与 upload_files 一致语义)
        await self.db.commit()
        logger.info(
            "import_url committed: document_id=%s file_size=%d elapsed_ms=%d",
            document.id,
            file_size,
            int((time.perf_counter() - start_time) * 1000),
        )
        if _submit_pipeline is not None:
            try:
                _submit_pipeline(str(document.id))
                logger.info(
                    "import_url pipeline submitted: document_id=%s",
                    document.id,
                )
            except Exception:
                # 与 upload_files 一致：导入结果已落库，入队失败只影响后续异步处理。
                logger.warning(
                    "import_url pipeline submit failed: document_id=%s",
                    document.id,
                    exc_info=True,
                )
        else:
            logger.warning(
                "import_url pipeline skipped: celery submitter unavailable document_id=%s",
                document.id,
            )

        return document

    # ─── Status Management ─────────────────────────────────────────────

    async def transition_status(
        self,
        document_id: uuid.UUID,
        new_status: DocumentStatus,
        error_detail: str | None = None,
        progress_percent: int = 0,
    ) -> Document:
        """Transition a document to a new status following the state machine.

        Args:
            document_id: Document ID
            new_status: Target status
            error_detail: Error detail if transitioning to failed
            progress_percent: Current progress percentage (0-100)

        Returns:
            Updated Document

        Raises:
            ValidationException: If transition is invalid
            NotFoundException: If document not found
        """
        document = await self._get_document(document_id)
        current_status = document.status

        # Validate transition
        valid_next = VALID_TRANSITIONS.get(current_status, [])
        if new_status not in valid_next:
            raise ValidationException(
                f"无效的状态转换: {current_status.value} → {new_status.value}"
            )

        # Update document
        document.status = new_status
        document.current_stage = new_status.value
        document.progress_percent = progress_percent

        if new_status == DocumentStatus.failed:
            document.error_detail = error_detail
            document.retry_count += 1

        if new_status == DocumentStatus.completed:
            document.progress_percent = 100
            document.error_detail = None

        await self.db.flush()
        await self.db.refresh(document)

        # Update Redis status cache
        await self._update_redis_status(document)

        return document

    async def get_document_progress(self, document_id: uuid.UUID) -> dict:
        """Get document processing progress from Redis cache.

        Args:
            document_id: Document ID

        Returns:
            Dict with stage, progress, and updated_at
        """
        redis = await get_redis()
        key = f"{REDIS_STATUS_PREFIX}{document_id}"

        data = await redis.hgetall(key)
        if not data:
            # Fallback to database
            document = await self._get_document(document_id)
            return {
                "document_id": str(document_id),
                "stage": document.current_stage or document.status.value,
                "progress": document.progress_percent,
                "updated_at": document.updated_at.isoformat() if document.updated_at else None,
            }

        return {
            "document_id": str(document_id),
            "stage": data.get("stage", "unknown"),
            "progress": int(data.get("progress", 0)),
            "updated_at": data.get("updated_at"),
        }

    # ─── Retry Logic ───────────────────────────────────────────────────

    async def retry_document(self, document_id: uuid.UUID) -> Document:
        """Retry processing a failed document.

        Resets status to pending and re-enqueues for processing.
        Raises ValidationException if document has reached max retries.

        Args:
            document_id: Document ID to retry

        Returns:
            Updated Document with reset status

        Raises:
            ValidationException: If document is not in failed state or max retries reached
            NotFoundException: If document not found
        """
        document = await self._get_document(document_id)

        if document.status != DocumentStatus.failed:
            raise ValidationException(
                f"只有状态为'失败'的文档才能重试，当前状态: {document.status.value}"
            )

        # Check permanent failure
        if document.retry_count >= MAX_RETRY_COUNT:
            raise ValidationException(
                f"文档已累计失败 {document.retry_count} 次，已标记为永久失败，需人工介入"
            )

        # Reset status to pending
        document.status = DocumentStatus.pending
        document.current_stage = DocumentStatus.pending.value
        document.progress_percent = 0
        document.error_detail = None

        await self.db.flush()
        await self.db.refresh(document)

        # Update Redis status
        await self._update_redis_status(document)

        # 重新入队到 Celery 管线 (parse → profile_match → process → chunk → embed → index)
        # 与 upload_files / import_url 保持同样语义。
        # 注意:retry_document 没有 await self.db.commit(),依赖外层路由 (FastAPI dep)
        # 在 success 时自动提交,所以入队也放在这里;若 commit 失败任务会丢但 db 状态
        # 会回滚到 failed,语义一致。
        if _submit_pipeline is not None:
            try:
                _submit_pipeline(str(document.id))
                logger.info(
                    "retry_document pipeline submitted: document_id=%s retry_count=%s",
                    document.id,
                    document.retry_count,
                )
            except Exception:
                # 入队失败不阻塞 retry 响应。前端 status=pending 用户可再次点 retry。
                logger.warning(
                    "retry_document pipeline submit failed: document_id=%s retry_count=%s",
                    document.id,
                    document.retry_count,
                    exc_info=True,
                )
        else:
            logger.warning(
                "retry_document pipeline skipped: celery submitter unavailable document_id=%s",
                document.id,
            )

        return document

    def is_permanently_failed(self, document: Document) -> bool:
        """Check if a document has reached permanent failure (3+ retries)."""
        return (
            document.status == DocumentStatus.failed
            and document.retry_count >= MAX_RETRY_COUNT
        )

    # ─── Private Helpers ───────────────────────────────────────────────

    def _get_file_extension(self, filename: str) -> str:
        """Extract file extension from filename (lowercase, no dot)."""
        if "." not in filename:
            return ""
        return filename.rsplit(".", 1)[-1].lower()

    def _generate_storage_path(self, space_id: uuid.UUID, filename: str) -> str:
        """Generate a unique storage path in MinIO.

        Format: {space_id}/{uuid}/{filename}
        """
        unique_id = uuid.uuid4()
        return f"{space_id}/{unique_id}/{filename}"

    def _store_to_minio(self, content: bytes, storage_path: str, file_ext: str) -> None:
        """Store file content to MinIO using streaming upload.

        boto3 ``upload_fileobj`` 自动判断单次 PUT 还是 multipart upload:
        - 默认 8MB 阈值, 大文件自动切片并行上传
        - 失败片段自动重试, 不需要重传整个文件
        - 对内存压力比 ``put_object(Body=bytes)`` 小: bytes 一次性载入,
          但底层 transfer manager 是流式发送

        参数与 boto3 TransferConfig 默认值一致, 不需要显式构造。
        """
        client = get_minio_client()
        ensure_bucket_exists()

        content_type = SUPPORTED_FORMATS.get(file_ext, "application/octet-stream")

        client.upload_fileobj(
            Fileobj=io.BytesIO(content),
            Bucket=settings.MINIO_BUCKET,
            Key=storage_path,
            ExtraArgs={"ContentType": content_type},
        )

    def _extract_title_from_url(self, url: str) -> str:
        """Extract a reasonable title from a URL."""
        # Remove protocol
        title = url.split("://", 1)[-1]
        # Remove query params
        title = title.split("?", 1)[0]
        # Remove trailing slash
        title = title.rstrip("/")
        # Get last path segment or domain
        parts = title.split("/")
        if len(parts) > 1 and parts[-1]:
            title = parts[-1]
        else:
            title = parts[0]
        # Limit length
        if len(title) > 200:
            title = title[:200]
        return title or "imported-url"

    async def _get_document(self, document_id: uuid.UUID) -> Document:
        """Get a document by ID or raise NotFoundException."""
        stmt = select(Document).where(Document.id == document_id)
        result = await self.db.execute(stmt)
        document = result.scalar_one_or_none()
        if not document:
            raise NotFoundException("Document", str(document_id))
        return document

    async def _init_redis_status(self, document_id: uuid.UUID) -> None:
        """Initialize document status in Redis."""
        redis = await get_redis()
        key = f"{REDIS_STATUS_PREFIX}{document_id}"
        now = datetime.now(timezone.utc).isoformat()
        await redis.hset(key, mapping={
            "stage": DocumentStatus.pending.value,
            "progress": "0",
            "updated_at": now,
        })
        await redis.expire(key, REDIS_STATUS_TTL)

    async def _update_redis_status(self, document: Document) -> None:
        """Update document status in Redis cache."""
        redis = await get_redis()
        key = f"{REDIS_STATUS_PREFIX}{document.id}"
        now = datetime.now(timezone.utc).isoformat()
        await redis.hset(key, mapping={
            "stage": document.current_stage or document.status.value,
            "progress": str(document.progress_percent),
            "updated_at": now,
        })
        await redis.expire(key, REDIS_STATUS_TTL)
