"""Celery task chain for document processing pipeline.

Implements the full pipeline orchestration:
parse_document → profile_match → process_document → chunk_document → embed_chunks → index_chunks

Each task has:
- bind=True for access to self (retry, task info)
- max_retries=3
- default_retry_delay=10 (exponential backoff)
- soft_time_limit=55
- time_limit=60
"""

import logging
import os
import tempfile
import time
import uuid

try:
    from celery import chain
    from celery.exceptions import MaxRetriesExceededError, SoftTimeLimitExceeded

    from app.core.celery_app import celery_app
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    celery_app = None

    # Fallback exception classes when celery is not installed
    class MaxRetriesExceededError(Exception):  # type: ignore[no-redef]
        pass

    class SoftTimeLimitExceeded(Exception):  # type: ignore[no-redef]
        pass

logger = logging.getLogger(__name__)


def _get_retry_delay(retry_count: int, base_delay: int = 10) -> int:
    """Calculate exponential backoff delay.

    Args:
        retry_count: Current retry attempt (0-indexed)
        base_delay: Initial delay in seconds

    Returns:
        Delay in seconds with exponential backoff
    """
    return base_delay * (2 ** retry_count)


# 任务 12.8：管线 stage → PostgreSQL ``DocumentStatus`` 枚举映射。
#
# 仅这些 stage 是 ``DocumentStatus`` 中的合法值，写 PG 时才用得上；
# - ``profile_matching``、``universal_parser_check`` 这类「子步骤」不在
#   状态枚举里，所以不映射，仅写 Redis 进度即可。
# - ``done`` 在 PG 侧落为 ``completed``，与 ``DocumentStatus.completed`` 对齐。
_STAGE_TO_DB_STATUS: dict[str, str] = {
    "parsing": "parsing",
    "cleaning": "cleaning",
    "chunking": "chunking",
    "embedding": "embedding",
    "indexing": "indexing",
    "done": "completed",
}


def _update_document_status(document_id: str, stage: str, progress: int = 0) -> None:
    """Update document processing status in Redis and PostgreSQL.

    任务 12.8：
    - **Redis**：每次调用都用最新 ``stage``/``progress``/``updated_at`` 刷一次
      ``doc:status:{document_id}`` Hash，给在线进度查询接口使用。
    - **PostgreSQL**：仅在 ``stage`` 是合法 ``DocumentStatus`` 枚举值且
      ``progress in {0, 100}``（步骤入口/出口）时同步更新 ``documents`` 表，
      避免每次中间进度都写库。子步骤（如 ``profile_matching``）只更新 Redis。

    Redis 与 PG 的失败都被局部 ``try/except`` 吞掉并仅记 WARNING——管线是
    「锦上添花」语义，状态更新失败不应让正在处理的文档崩盘。

    Args:
        document_id: Document UUID
        stage: Current processing stage
        progress: Progress percentage (0-100)
    """
    try:
        import redis as redis_lib

        from app.core.config import get_settings

        settings = get_settings()
        r = redis_lib.Redis.from_url(settings.REDIS_URL)
        key = f"doc:status:{document_id}"
        r.hset(key, mapping={
            "stage": stage,
            "progress": str(progress),
            "updated_at": str(time.time()),
        })
        r.expire(key, 3600)
    except Exception as e:
        logger.warning(f"Failed to update Redis status for {document_id}: {e}")

    # 仅在「步骤进入」(progress=0) 或「步骤结束」(progress=100) 时把状态推到
    # PostgreSQL，覆盖需求 4.8 的「用户可查询所处步骤」。中间进度走 Redis 即可。
    if progress in (0, 100):
        db_status = _STAGE_TO_DB_STATUS.get(stage)
        if db_status:
            try:
                from app.services.indexing_service import update_document_db_status

                update_document_db_status(
                    document_id,
                    db_status,
                    current_stage=stage,
                    progress_percent=progress,
                )
            except Exception as e:
                # ``update_document_db_status`` 自身已包了 try/except，这里再兜一层
                # 是为了防御 import 失败（例如 sqlalchemy 在某些测试环境缺失）。
                logger.warning(
                    f"Failed to update PG status for {document_id} (stage={stage}): {e}"
                )


def _mark_document_failed(document_id: str, stage: str, error: str) -> None:
    """Mark document as failed in Redis and PostgreSQL.

    任务 12.8：失败路径必须把 ``Document.status`` 置为 ``failed`` 并写入
    ``error_detail``（需求 4.7 / 4.9）。Redis 与 PG 任意一边失败都不阻塞调用方
    重新抛出原始异常——管线靠 Celery 的 retry/MaxRetries 控制，状态更新只是
    可观测性的副作用。

    Args:
        document_id: Document UUID
        stage: Stage where failure occurred
        error: Error description
    """
    try:
        import redis as redis_lib

        from app.core.config import get_settings

        settings = get_settings()
        r = redis_lib.Redis.from_url(settings.REDIS_URL)
        key = f"doc:status:{document_id}"
        r.hset(key, mapping={
            "stage": "failed",
            "progress": "0",
            "error": error,
            "failed_stage": stage,
            "updated_at": str(time.time()),
        })
        r.expire(key, 3600)
    except Exception as e:
        logger.warning(f"Failed to update Redis failure status for {document_id}: {e}")

    try:
        from app.services.indexing_service import update_document_db_status

        update_document_db_status(
            document_id,
            "failed",
            current_stage=stage,
            error_detail=error,
        )
    except Exception as e:
        logger.warning(
            f"Failed to update PG failure status for {document_id} (stage={stage}): {e}"
        )


def _download_file_from_minio(storage_path: str) -> str:
    """Download a file from MinIO to a temporary location.

    Args:
        storage_path: MinIO object path

    Returns:
        Local temporary file path
    """
    from app.core.config import get_settings
    from app.core.minio import get_minio_client

    settings = get_settings()
    client = get_minio_client()

    # Create temp file with appropriate extension
    ext = os.path.splitext(storage_path)[1]
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp_path = tmp.name
    tmp.close()

    client.download_file(
        Bucket=settings.MINIO_BUCKET,
        Key=storage_path,
        Filename=tmp_path,
    )

    return tmp_path


def _task_decorator(**kwargs):
    """Decorator that wraps celery_app.task when available, or is a no-op."""
    if CELERY_AVAILABLE and celery_app is not None:
        return celery_app.task(**kwargs)
    else:
        def decorator(func):
            return func
        return decorator


@_task_decorator(
    bind=True,
    name="pipeline.parse_document",
    max_retries=3,
    default_retry_delay=10,
    soft_time_limit=580,
    time_limit=600,
)
def parse_document(self, document_id: str) -> dict:
    """Parse a document using the appropriate parser plugin.

    Downloads the file from MinIO, selects the correct parser based on
    file extension, and returns the parsed intermediate representation.

    Args:
        document_id: UUID of the document to parse

    Returns:
        Dict containing document_id and serialized parse result

    Raises:
        Retries on transient failures, marks as failed on permanent errors
    """
    logger.info(f"Starting parse_document for {document_id}")
    start_time = time.perf_counter()
    _update_document_status(document_id, "parsing", 0)

    try:
        from app.services.parsers.base import ParseError
        from app.services.parsers.registry import get_parser_registry

        # Get document info from database (synchronous for Celery)
        doc_info = _get_document_info(document_id)
        if not doc_info:
            raise ValueError(f"Document not found: {document_id}")

        storage_path = doc_info["storage_path"]
        file_type = doc_info["file_type"]
        logger.info(
            "parse_document loaded document: document_id=%s file_type=%s source_file=%s",
            document_id,
            file_type,
            os.path.basename(storage_path),
        )

        # Download file from MinIO
        _update_document_status(document_id, "parsing", 10)
        local_path = _download_file_from_minio(storage_path)

        try:
            # Select parser
            registry = get_parser_registry()
            _ensure_default_parsers_registered(registry)

            mime_type = _get_mime_type(file_type)
            parser = registry.select(local_path, mime_type)
            logger.info(
                "parse_document selected parser: document_id=%s parser=%s mime_type=%s",
                document_id,
                getattr(parser, "name", parser.__class__.__name__),
                mime_type,
            )

            _update_document_status(document_id, "parsing", 30)

            # Parse the document (run async in sync context)
            import asyncio
            loop = asyncio.new_event_loop()
            try:
                parsed_doc = loop.run_until_complete(parser.parse(local_path))
            finally:
                loop.close()

            _update_document_status(document_id, "parsing", 90)

            # Serialize the result
            result = {
                "document_id": document_id,
                "blocks": [
                    {
                        "type": block.type,
                        "text": block.text,
                        "bbox": block.bbox,
                        "page_number": block.page_number,
                        "style": block.style,
                    }
                    for block in parsed_doc.blocks
                ],
                "metadata": parsed_doc.metadata,
                "asset_count": len(parsed_doc.assets),
            }

            _update_document_status(document_id, "parsing", 100)
            logger.info(
                "Parsed document %s: blocks=%d assets=%d elapsed_ms=%d",
                document_id,
                len(parsed_doc.blocks),
                len(parsed_doc.assets),
                int((time.perf_counter() - start_time) * 1000),
            )
            return result

        finally:
            # Clean up temp file
            if os.path.exists(local_path):
                os.unlink(local_path)

    except ParseError as e:
        # Permanent parse errors (corrupted, password-protected) - don't retry
        if e.reason in ("corrupted", "password_protected", "empty"):
            logger.error(f"Permanent parse error for {document_id}: {e} (reason={e.reason})")
            _mark_document_failed(document_id, "parsing", f"{e.reason}: {str(e)}")
            raise  # Don't retry permanent errors

        # Transient errors - retry with exponential backoff
        logger.warning(f"Transient parse error for {document_id}: {e}, retrying...")
        try:
            retry_delay = _get_retry_delay(self.request.retries)
            raise self.retry(exc=e, countdown=retry_delay)
        except MaxRetriesExceededError:
            _mark_document_failed(document_id, "parsing", str(e))
            raise

    except SoftTimeLimitExceeded:
        logger.error(f"Parse timeout for document {document_id}")
        try:
            retry_delay = _get_retry_delay(self.request.retries)
            raise self.retry(countdown=retry_delay)
        except MaxRetriesExceededError:
            _mark_document_failed(document_id, "parsing", "Timeout exceeded")
            raise

    except (ValueError, ImportError) as e:
        # Non-retryable errors
        _mark_document_failed(document_id, "parsing", str(e))
        raise

    except Exception as e:
        logger.error(f"Unexpected error parsing {document_id}: {e}")
        try:
            retry_delay = _get_retry_delay(self.request.retries)
            raise self.retry(exc=e, countdown=retry_delay)
        except MaxRetriesExceededError:
            _mark_document_failed(document_id, "parsing", str(e))
            raise


@_task_decorator(
    bind=True,
    name="pipeline.profile_match",
    max_retries=3,
    default_retry_delay=10,
    soft_time_limit=55,
    time_limit=60,
)
def profile_match(self, parse_result: dict) -> dict:
    """Match a document profile based on parsed content.

    Uses ProfileMatcher to extract features from the parsed document and
    match against all enabled profiles in the database.

    Args:
        parse_result: Output from parse_document task

    Returns:
        Dict with parse_result + matched profile info
    """
    document_id = parse_result["document_id"]
    logger.info(f"Starting profile_match for {document_id}")
    start_time = time.perf_counter()
    _update_document_status(document_id, "profile_matching", 0)

    try:
        from app.services.parsers.base import Block, ParsedDocument
        from app.services.profile_matcher import ProfileMatcher

        # Reconstruct ParsedDocument from serialized blocks
        blocks = [
            Block(
                type=b["type"],
                text=b["text"],
                bbox=tuple(b["bbox"]) if b.get("bbox") else None,
                page_number=b.get("page_number", 1),
                style=b.get("style", {}),
            )
            for b in parse_result.get("blocks", [])
        ]
        parsed_doc = ParsedDocument(
            blocks=blocks,
            metadata=parse_result.get("metadata", {}),
        )

        _update_document_status(document_id, "profile_matching", 30)

        # Load enabled profiles from database
        profiles = _load_profiles_from_db()
        logger.info(
            "profile_match loaded profiles: document_id=%s profile_count=%d block_count=%d",
            document_id,
            len(profiles),
            len(blocks),
        )

        _update_document_status(document_id, "profile_matching", 60)

        # Get filename from document info
        doc_info = _get_document_info(document_id)
        filename = ""
        if doc_info:
            filename = os.path.basename(doc_info.get("storage_path", ""))

        # Run profile matching
        matcher = ProfileMatcher(profiles=profiles)
        matched_profile = matcher.match(parsed_doc, filename)

        _update_document_status(document_id, "profile_matching", 100)

        result = {
            **parse_result,
            "profile_id": matched_profile.id if matched_profile.id != "default" else None,
            "profile_name": matched_profile.name,
        }

        logger.info(
            "Profile matched for %s: profile=%s priority=%s elapsed_ms=%d",
            document_id,
            matched_profile.name,
            matched_profile.priority,
            int((time.perf_counter() - start_time) * 1000),
        )
        return result

    except SoftTimeLimitExceeded:
        logger.error(f"Profile match timeout for {document_id}")
        try:
            retry_delay = _get_retry_delay(self.request.retries)
            raise self.retry(countdown=retry_delay)
        except MaxRetriesExceededError:
            _mark_document_failed(document_id, "profile_matching", "Timeout exceeded")
            raise

    except Exception as e:
        logger.error(f"Error in profile_match for {document_id}: {e}")
        # On error, fall back to generic-text rather than failing the pipeline
        logger.warning(f"Falling back to generic-text profile for {document_id}")
        _update_document_status(document_id, "profile_matching", 100)
        return {
            **parse_result,
            "profile_id": None,
            "profile_name": "generic-text",
        }


@_task_decorator(
    bind=True,
    name="pipeline.universal_parser_check",
    max_retries=3,
    default_retry_delay=10,
    soft_time_limit=580,
    time_limit=600,
)
def universal_parser_check(self, match_result: dict) -> dict:
    """任务 10.9：在 Profile 匹配后判断是否需要走 LLM 通用兜底解析。

    判定使用 ``universal_parser_trigger.should_run_universal_parser``：

    - ``profile_id is None`` 或 ``profile_name == "generic-text"`` → 触发；
    - ``quality_score < settings.QUALITY_FALLBACK_THRESHOLD`` → 触发（任务 11 接入
      QualityScorer 之前，这里始终传 ``None``，下方有 TODO 标记）。

    触发后：
    1. 重建 ``ParsedDocument``（与 ``profile_match`` 一致的反序列化逻辑）。
    2. 在 ``asyncio.new_event_loop()`` 内调用
       ``run_universal_parser_and_persist_candidate``，得到 LLM 处理结果与候选
       Profile ID。
    3. 用 LLM 处理后的 blocks 替换 ``match_result["blocks"]``，让下游
       ``process_document`` / ``chunk_document`` 跑在「LLM 清洗后」的内容上。
    4. 把 ``ProcessedDocument.metadata["universal_parser"]`` envelope 合并进
       ``match_result["metadata"]``，并打上顶层 marker
       ``universal_parser_triggered`` / ``universal_parser_trigger_reasons`` /
       ``candidate_profile_id``。

    没触发时直接透传 ``match_result``，让 ``process_document`` 按原路径处理。

    Args:
        match_result: ``profile_match`` 的输出。

    Returns:
        透传或被 LLM 处理结果改写过的 ``match_result``。
    """
    document_id = match_result["document_id"]
    profile_id = match_result.get("profile_id")
    profile_name = match_result.get("profile_name")
    start_time = time.perf_counter()

    # TODO(10.11): 等任务 11 的 QualityScorer 上线后，这里需要把真实的
    #              quality_score 传进 should_run_universal_parser，让「质量分低于
    #              阈值」也能触发兜底。当前阶段管线尚未计算 quality_score，所以
    #              传 None。
    from app.services.universal_parser_trigger import (
        run_universal_parser_and_persist_candidate,
        should_run_universal_parser,
    )

    should_run, reasons = should_run_universal_parser(
        profile_id=profile_id,
        profile_name=profile_name,
        quality_score=None,
    )

    if not should_run:
        logger.info(
            "universal_parser skipped for %s: profile=%s elapsed_ms=%d",
            document_id,
            profile_name,
            int((time.perf_counter() - start_time) * 1000),
        )
        return match_result

    logger.info(
        "universal_parser triggered for %s: reasons=%s",
        document_id,
        reasons,
    )

    try:
        from app.services.parsers.base import Block, ParsedDocument

        blocks = [
            Block(
                type=b["type"],
                text=b["text"],
                bbox=tuple(b["bbox"]) if b.get("bbox") else None,
                page_number=b.get("page_number", 1),
                style=b.get("style", {}),
            )
            for b in match_result.get("blocks", [])
        ]
        parsed_doc = ParsedDocument(
            blocks=blocks,
            metadata=dict(match_result.get("metadata") or {}),
        )

        # 在同步 Celery 任务里运行异步编排函数，与 ``parse_document`` 保持同样的
        # ``asyncio.new_event_loop()`` 风格。``db=None`` 走只解析不持久化的路径，
        # 持久化版本在生产部署里由 worker 注入 AsyncSession（下方使用
        # AsyncSessionLocal 自带的事务管理）。
        import asyncio

        async def _run_with_session() -> dict:
            from app.core.database import AsyncSessionLocal

            async with AsyncSessionLocal() as session:
                return await run_universal_parser_and_persist_candidate(
                    parsed_doc,
                    db=session,
                )

        loop = asyncio.new_event_loop()
        try:
            try:
                outcome = loop.run_until_complete(_run_with_session())
            except Exception as exc:  # noqa: BLE001 — defensive: DB might not be available
                logger.warning(
                    "universal_parser DB-backed run failed for %s: %s; retrying without persistence",
                    document_id,
                    exc,
                )
                outcome = loop.run_until_complete(
                    run_universal_parser_and_persist_candidate(
                        parsed_doc,
                        db=None,
                    )
                )
        finally:
            loop.close()

        processed_document = outcome.get("processed_document") or {}
        candidate_profile_id = outcome.get("candidate_profile_id")

        # 把 LLM 处理后的 blocks 透传给下游。``ProcessedBlock`` 的字段集合与
        # 上游 ``Block`` 不完全一致，所以这里做一次 dict 投影，保留下游真正需要
        # 的字段：``type``/``text``/``page_number``/``style``。
        new_blocks_raw = processed_document.get("blocks") or []
        new_blocks: list[dict] = []
        for nb in new_blocks_raw:
            if not isinstance(nb, dict):
                continue
            text_value = nb.get("text", "") or ""
            if not text_value.strip():
                continue
            style = dict(nb.get("style") or {})
            heading_level = nb.get("heading_level") or 0
            if heading_level:
                style["heading_level"] = heading_level
            new_blocks.append(
                {
                    "type": nb.get("type", "paragraph"),
                    "text": text_value,
                    "bbox": None,
                    "page_number": nb.get("page_number", 1),
                    "style": style,
                }
            )

        merged_metadata = dict(match_result.get("metadata") or {})
        processed_metadata = processed_document.get("metadata") or {}
        if isinstance(processed_metadata, dict) and processed_metadata.get("universal_parser"):
            merged_metadata["universal_parser"] = processed_metadata["universal_parser"]

        logger.info(
            "universal_parser completed for %s: input_blocks=%d output_blocks=%d "
            "candidate_profile_id=%s elapsed_ms=%d",
            document_id,
            len(blocks),
            len(new_blocks),
            candidate_profile_id,
            int((time.perf_counter() - start_time) * 1000),
        )

        return {
            **match_result,
            # 仅在 LLM 真的产出了块时才覆盖，避免空降级吞掉所有内容。
            "blocks": new_blocks if new_blocks else match_result.get("blocks", []),
            "metadata": merged_metadata,
            "universal_parser_triggered": True,
            "universal_parser_trigger_reasons": list(reasons),
            "candidate_profile_id": candidate_profile_id,
        }

    except SoftTimeLimitExceeded:
        logger.error(f"Universal parser timeout for {document_id}")
        try:
            retry_delay = _get_retry_delay(self.request.retries)
            raise self.retry(countdown=retry_delay)
        except MaxRetriesExceededError:
            # 兜底：超时后让原始 match_result 继续走原路径，避免文档卡死。
            logger.warning(
                "universal_parser exhausted retries for %s; passing through",
                document_id,
            )
            return match_result

    except Exception as e:  # noqa: BLE001 — pipeline must not crash on optional step
        logger.error(f"Universal parser failed for {document_id}: {e}")
        # LLM 兜底是「锦上添花」步骤，失败时不应阻塞下游。透传原始 match_result。
        return {
            **match_result,
            "universal_parser_triggered": False,
            "universal_parser_trigger_reasons": list(reasons),
            "universal_parser_error": str(e),
        }


@_task_decorator(
    bind=True,
    name="pipeline.process_document",
    max_retries=3,
    default_retry_delay=10,
    soft_time_limit=290,
    time_limit=300,
)
def process_document(self, match_result: dict) -> dict:
    """Process document: cleaning, structural recognition, quality scoring, review-queue gating.

    管线第三步（在 ``profile_match`` / ``universal_parser_check`` 之后、
    ``chunk_document`` 之前）：

    1. 用 ``DocumentProcessor`` 按命中的 Profile 做清洗 + 结构识别（任务 9）。
    2. 用 ``QualityScorer`` 对清洗后的文档打多维度质量分（任务 11.1-11.7）。
    3. 把 ``ParseQualityScore`` 通过 ``to_dict()`` 写到 ``Document.quality_score``
       JSONB 列（持久化），保持 ``ParseQualityScore.from_dict`` 的双向往返。
    4. 当 ``quality_scorer.needs_review(score)`` 为真时（``overall < 0.7``，
       任务 11.8），调用 ``ReviewQueue.enqueue`` 把文档加入人工审核队列；
       入队是幂等的（重复入队仅刷新已存在 pending 行的 quality_score）。

    返回的 ``blocks`` 来自 ``ProcessedDocument``，让下游 ``chunk_document``
    跑在「清洗后」的内容上。``metadata`` 中追加 ``quality_score`` 与
    ``review_enqueued`` 两个 marker，便于排查与下游可观测。

    Args:
        match_result: Output from profile_match / universal_parser_check task

    Returns:
        Dict with cleaned/structured content + scoring + review queue marker
    """
    document_id = match_result["document_id"]
    logger.info(f"Starting process_document for {document_id}")
    start_time = time.perf_counter()
    _update_document_status(document_id, "cleaning", 0)

    try:
        from app.services.document_processor import DocumentProcessor
        from app.services.parsers.base import Block, ParsedDocument
        from app.services.quality_scorer import QualityScorer

        # 1) 重建 ParsedDocument（与 profile_match 的反序列化逻辑保持一致）。
        original_blocks = [
            Block(
                type=b["type"],
                text=b["text"],
                bbox=tuple(b["bbox"]) if b.get("bbox") else None,
                page_number=b.get("page_number", 1),
                style=b.get("style", {}),
            )
            for b in match_result.get("blocks", [])
        ]
        parsed_doc = ParsedDocument(
            blocks=original_blocks,
            metadata=dict(match_result.get("metadata") or {}),
        )

        _update_document_status(document_id, "cleaning", 30)

        # 2) 加载命中的 Profile（``profile_id`` 缺省 → 读 generic-text 兜底）。
        profile = _resolve_profile_for_processing(
            profile_id=match_result.get("profile_id"),
            profile_name=match_result.get("profile_name"),
        )

        # 3) 跑清洗 / 结构识别。``enable_llm_image_description=False`` 与
        #    ``DocumentProcessor`` 默认值一致；多模态图片描述需要在
        #    专门的子任务里启用，避免管线意外消耗 LLM 配额。
        processor = DocumentProcessor(enable_llm_image_description=False)
        processed = processor.process(parsed_doc, profile)

        _update_document_status(document_id, "cleaning", 70)

        # 4) 打多维度质量分。``QualityScorer()`` 会读取默认权重 + 0.7 阈值。
        scorer = QualityScorer()
        score = scorer.score(parsed_doc, processed, profile)
        score_dict = score.to_dict()

        # 5) 持久化质量分到 documents 表。沿用 indexing_service 的同步引擎写法，
        #    避免在 Celery 任务里跑 asyncpg 事件循环。
        _persist_document_quality_score(document_id, score_dict)

        # 6) 审核队列门禁：``score < 0.7`` 时入队。失败不阻塞下游，只记 WARNING；
        #    管线是「锦上添花」语义——评分系统宕机时仍要让文档继续走完。
        #    把 ``ProcessedDocument.markdown`` 一并落进 JSONB（key:
        #    ``parsed_markdown``），任务 11.10 文档并排预览 API 直接读取该字段。
        review_enqueued = False
        if scorer.needs_review(score):
            try:
                review_enqueued = _enqueue_for_review_sync(
                    document_id,
                    score,
                    parsed_markdown=processed.markdown or None,
                )
            except Exception as exc:  # noqa: BLE001 — defensive
                logger.warning(
                    "process_document: failed to enqueue document %s for review: %s",
                    document_id,
                    exc,
                )

        # 7) 把 ProcessedBlock 投影回与上游兼容的 dict 形态。``style`` 用来承载
        #    heading_level，与 universal_parser_check 的输出口径保持一致。
        result_blocks: list[dict] = []
        for block in processed.blocks:
            if block.is_noise:
                continue
            text_value = block.text or ""
            if not text_value.strip():
                continue
            style = {}
            if block.heading_level:
                style["heading_level"] = block.heading_level
            result_blocks.append(
                {
                    "type": block.type,
                    "text": text_value,
                    "bbox": None,
                    "page_number": block.page_number,
                    "style": style,
                    "asset_ids": list(block.asset_ids),
                }
            )

        merged_metadata = dict(match_result.get("metadata") or {})
        merged_metadata["quality_score"] = score_dict
        merged_metadata["review_enqueued"] = review_enqueued

        _update_document_status(document_id, "cleaning", 100)

        result = {
            "document_id": document_id,
            "blocks": result_blocks if result_blocks else match_result.get("blocks", []),
            "metadata": merged_metadata,
            "profile_id": match_result.get("profile_id"),
            "profile_name": match_result.get("profile_name"),
            "quality_score": score_dict,
            "review_enqueued": review_enqueued,
        }

        logger.info(
            "Processed document %s: input_blocks=%d output_blocks=%d overall=%.4f "
            "review_enqueued=%s elapsed_ms=%d",
            document_id,
            len(original_blocks),
            len(result["blocks"]),
            score.overall,
            review_enqueued,
            int((time.perf_counter() - start_time) * 1000),
        )
        return result

    except SoftTimeLimitExceeded:
        logger.error(f"Process timeout for {document_id}")
        try:
            retry_delay = _get_retry_delay(self.request.retries)
            raise self.retry(countdown=retry_delay)
        except MaxRetriesExceededError:
            _mark_document_failed(document_id, "cleaning", "Timeout exceeded")
            raise

    except Exception as e:
        logger.error(f"Error processing {document_id}: {e}")
        try:
            retry_delay = _get_retry_delay(self.request.retries)
            raise self.retry(exc=e, countdown=retry_delay)
        except MaxRetriesExceededError:
            _mark_document_failed(document_id, "cleaning", str(e))
            raise


@_task_decorator(
    bind=True,
    name="pipeline.chunk_document",
    max_retries=3,
    default_retry_delay=10,
    soft_time_limit=290,
    time_limit=300,
)
def chunk_document(self, process_result: dict) -> dict:
    """Chunk document into smaller pieces for embedding.

    Args:
        process_result: Output from process_document task

    Returns:
        Dict with document chunks
    """
    document_id = process_result["document_id"]
    logger.info(f"Starting chunk_document for {document_id}")
    start_time = time.perf_counter()
    _update_document_status(document_id, "chunking", 0)

    try:
        # 当前管线的 chunk 输入只需要文本块摘要，不把正文写入日志，避免泄露内容。
        blocks = process_result["blocks"]
        chunks = []
        for i, block in enumerate(blocks):
            if block["text"].strip():
                chunks.append({
                    "chunk_index": i,
                    "text": block["text"],
                    "type": block["type"],
                    "page_number": block.get("page_number", 1),
                    "style": block.get("style", {}),
                })

        _update_document_status(document_id, "chunking", 100)

        result = {
            "document_id": document_id,
            "chunks": chunks,
            "metadata": process_result["metadata"],
            "profile_id": process_result.get("profile_id"),
        }

        logger.info(
            "Chunked document %s: input_blocks=%d chunks=%d elapsed_ms=%d",
            document_id,
            len(blocks),
            len(chunks),
            int((time.perf_counter() - start_time) * 1000),
        )
        return result

    except SoftTimeLimitExceeded:
        logger.error(f"Chunk timeout for {document_id}")
        try:
            retry_delay = _get_retry_delay(self.request.retries)
            raise self.retry(countdown=retry_delay)
        except MaxRetriesExceededError:
            _mark_document_failed(document_id, "chunking", "Timeout exceeded")
            raise

    except Exception as e:
        logger.error(f"Error chunking {document_id}: {e}")
        try:
            retry_delay = _get_retry_delay(self.request.retries)
            raise self.retry(exc=e, countdown=retry_delay)
        except MaxRetriesExceededError:
            _mark_document_failed(document_id, "chunking", str(e))
            raise


@_task_decorator(
    bind=True,
    name="pipeline.embed_chunks",
    max_retries=3,
    default_retry_delay=10,
    soft_time_limit=580,
    time_limit=600,
)
def embed_chunks(self, chunk_result: dict) -> dict:
    """Generate embeddings for document chunks.

    Calls EmbeddingService to generate dense (1024-dim via LiteLLM) and
    sparse (TF-IDF based) vectors for each chunk.

    Args:
        chunk_result: Output from chunk_document task

    Returns:
        Dict with chunks, their embeddings, and metadata
    """
    document_id = chunk_result["document_id"]
    logger.info(f"Starting embed_chunks for {document_id}")
    start_time = time.perf_counter()
    _update_document_status(document_id, "embedding", 0)

    try:
        import asyncio

        from app.services.embedding_service import EmbeddingService

        chunks = chunk_result["chunks"]
        if not chunks:
            _update_document_status(document_id, "embedding", 100)
            logger.info(
                "embed_chunks skipped for %s: no chunks elapsed_ms=%d",
                document_id,
                int((time.perf_counter() - start_time) * 1000),
            )
            return {
                "document_id": document_id,
                "chunks": [],
                "embeddings": [],
                "metadata": chunk_result.get("metadata", {}),
                "profile_id": chunk_result.get("profile_id"),
            }

        # Prepare chunks for embedding (need 'id' and 'text' fields)
        embed_input = []
        for chunk in chunks:
            chunk_id = chunk.get("id") or str(uuid.uuid4())
            chunk["id"] = chunk_id  # Ensure each chunk has an ID
            embed_input.append({"id": chunk_id, "text": chunk["text"]})

        _update_document_status(document_id, "embedding", 20)

        # Generate embeddings
        service = EmbeddingService()
        loop = asyncio.new_event_loop()
        try:
            results = loop.run_until_complete(service.embed_chunks(embed_input))
        finally:
            loop.close()

        _update_document_status(document_id, "embedding", 90)

        # Serialize embedding results
        embeddings = []
        for result in results:
            embeddings.append({
                "chunk_id": result.chunk_id,
                "dense_vector": result.dense_vector,
                "sparse_indices": result.sparse_indices,
                "sparse_values": result.sparse_values,
            })

        _update_document_status(document_id, "embedding", 100)

        output = {
            "document_id": document_id,
            "chunks": chunks,
            "embeddings": embeddings,
            "metadata": chunk_result.get("metadata", {}),
            "profile_id": chunk_result.get("profile_id"),
        }

        logger.info(
            "Embedded chunks for document %s: chunks=%d embeddings=%d elapsed_ms=%d",
            document_id,
            len(chunks),
            len(embeddings),
            int((time.perf_counter() - start_time) * 1000),
        )
        return output

    except SoftTimeLimitExceeded:
        logger.error(f"Embed timeout for {document_id}")
        try:
            retry_delay = _get_retry_delay(self.request.retries)
            raise self.retry(countdown=retry_delay)
        except MaxRetriesExceededError:
            _mark_document_failed(document_id, "embedding", "Timeout exceeded")
            raise

    except Exception as e:
        logger.error(f"Error embedding {document_id}: {e}")
        try:
            retry_delay = _get_retry_delay(self.request.retries)
            raise self.retry(exc=e, countdown=retry_delay)
        except MaxRetriesExceededError:
            _mark_document_failed(document_id, "embedding", str(e))
            raise


@_task_decorator(
    bind=True,
    name="pipeline.index_chunks",
    max_retries=3,
    default_retry_delay=10,
    soft_time_limit=290,
    time_limit=300,
)
def index_chunks(self, embed_result: dict) -> dict:
    """Index chunks into Qdrant and OpenSearch with dual-write transaction.

    Writes chunks with their vectors to Qdrant and full-text content to
    OpenSearch. Both writes must succeed or both are rolled back.

    Args:
        embed_result: Output from embed_chunks task

    Returns:
        Dict with indexing result summary
    """
    document_id = embed_result["document_id"]
    logger.info(f"Starting index_chunks for {document_id}")
    start_time = time.perf_counter()
    _update_document_status(document_id, "indexing", 0)

    try:
        from app.core.opensearch import ensure_index_exists
        from app.core.qdrant import ensure_collection_exists
        from app.services.embedding_service import EmbeddingResult
        from app.services.indexing_service import (
            ChunkPayload,
            IndexingService,
            update_document_db_status,
        )

        chunks = embed_result.get("chunks", [])
        embeddings_data = embed_result.get("embeddings", [])
        logger.info(
            "index_chunks preparing: document_id=%s chunks=%d embeddings=%d",
            document_id,
            len(chunks),
            len(embeddings_data),
        )

        if not chunks:
            _update_document_status(document_id, "done", 100)
            update_document_db_status(document_id, "completed", "done", 100)
            logger.info(
                "index_chunks skipped for %s: no chunks elapsed_ms=%d",
                document_id,
                int((time.perf_counter() - start_time) * 1000),
            )
            return {
                "document_id": document_id,
                "indexed_chunks": 0,
                "status": "completed",
            }

        _update_document_status(document_id, "indexing", 10)

        # Ensure storage backends are ready
        ensure_collection_exists()
        ensure_index_exists()

        _update_document_status(document_id, "indexing", 20)

        # Get document info for permissions
        doc_info = _get_document_info(document_id)
        space_id = ""
        source_file = ""
        if doc_info:
            source_file = os.path.basename(doc_info.get("storage_path", ""))
            # Get space_id from document record
            space_id = _get_document_space_id(document_id) or ""

        # Build payloads
        payloads = []
        for chunk in chunks:
            payload = ChunkPayload(
                chunk_id=chunk.get("id", str(uuid.uuid4())),
                document_id=document_id,
                space_id=chunk.get("space_id", space_id),
                chunk_index=chunk.get("chunk_index", 0),
                title_chain=chunk.get("title_chain", ""),
                source_file=chunk.get("source_file", source_file),
                page_number=chunk.get("page_number", 1),
                content=chunk.get("text", ""),
                parent_chunk_id=chunk.get("parent_id"),
                depth=chunk.get("depth", 1),
                token_count=chunk.get("token_count", 0),
                allowed_user_ids=chunk.get("permission_ids", []),
                access_level="read",
            )
            payloads.append(payload)

        # Build embedding results
        embedding_results = []
        for emb_data in embeddings_data:
            embedding_results.append(EmbeddingResult(
                chunk_id=emb_data.get("chunk_id", ""),
                dense_vector=emb_data.get("dense_vector", []),
                sparse_indices=emb_data.get("sparse_indices", []),
                sparse_values=emb_data.get("sparse_values", []),
            ))

        # Embedding 数量不足属于上游降级场景：补零向量让索引链路可完成，
        # 同时用日志暴露数量差，便于后续排查召回质量问题。
        padded_embeddings = 0
        while len(embedding_results) < len(payloads):
            embedding_results.append(EmbeddingResult(
                chunk_id=payloads[len(embedding_results)].chunk_id,
                dense_vector=[0.0] * 1024,
                sparse_indices=[],
                sparse_values=[],
            ))
            padded_embeddings += 1

        if padded_embeddings:
            logger.warning(
                "index_chunks padded embeddings: document_id=%s padded=%d payloads=%d",
                document_id,
                padded_embeddings,
                len(payloads),
            )

        _update_document_status(document_id, "indexing", 40)

        # Perform dual-write
        service = IndexingService()
        result = service.index_chunks(payloads, embedding_results)

        _update_document_status(document_id, "done", 100)

        # Update PostgreSQL document status
        update_document_db_status(document_id, "completed", "done", 100)

        output = {
            "document_id": document_id,
            "indexed_chunks": result["qdrant_count"],
            "status": "completed",
        }

        logger.info(
            "Indexed document %s: qdrant_chunks=%d opensearch_docs=%d elapsed_ms=%d",
            document_id,
            result["qdrant_count"],
            result["opensearch_count"],
            int((time.perf_counter() - start_time) * 1000),
        )
        return output

    except SoftTimeLimitExceeded:
        logger.error(f"Index timeout for {document_id}")
        try:
            retry_delay = _get_retry_delay(self.request.retries)
            raise self.retry(countdown=retry_delay)
        except MaxRetriesExceededError:
            _mark_document_failed(document_id, "indexing", "Timeout exceeded")
            raise

    except Exception as e:
        logger.error(f"Error indexing {document_id}: {e}")
        try:
            retry_delay = _get_retry_delay(self.request.retries)
            raise self.retry(exc=e, countdown=retry_delay)
        except MaxRetriesExceededError:
            _mark_document_failed(document_id, "indexing", str(e))
            raise


def submit_pipeline(document_id: str) -> None:
    """Submit a document to the processing pipeline.

    Creates and dispatches the full Celery task chain:
    parse → profile_match → process → chunk → embed → index

    Args:
        document_id: UUID of the document to process

    Raises:
        RuntimeError: If Celery is not available
    """
    if not CELERY_AVAILABLE:
        raise RuntimeError("Celery is not available. Cannot submit pipeline.")

    pipeline = chain(
        parse_document.s(document_id),
        profile_match.s(),
        universal_parser_check.s(),
        process_document.s(),
        chunk_document.s(),
        embed_chunks.s(),
        index_chunks.s(),
    )
    pipeline.apply_async()
    logger.info(f"Submitted pipeline for document {document_id}")


# ─── 任务 11.11: 人工修正后重新触发分块/向量化 ─────────────────────────


@_task_decorator(
    bind=True,
    name="pipeline.cleanup_document_indices",
    max_retries=3,
    default_retry_delay=10,
    soft_time_limit=290,
    time_limit=300,
)
def cleanup_document_indices(self, process_result: dict) -> dict:
    """删除文档在 Qdrant / OpenSearch 中的旧 chunk（任务 11.11）。

    人工修正流程（``submit_reprocess_from_markdown``）的链头：把同一
    ``document_id`` 在搜索后端里的旧切片清掉，避免修正前的解析结果
    继续残留在召回结果里。失败时仅 WARNING，让链路继续走完——
    ``chunk_document`` 在生成新切片时会用全新的 UUID，下游
    ``index_chunks`` 的 upsert 也能覆盖大多数残留场景。

    Args:
        process_result: 合成的 ``process_result``-shape dict，``chunk_document``
            会消费它。透传不变以让 Celery chain 的下一节点继续使用。

    Returns:
        透传的 ``process_result``，便于 chain 链式传递。
    """
    document_id = process_result["document_id"]
    logger.info("Starting cleanup_document_indices for %s", document_id)

    try:
        from app.services.indexing_service import IndexingService

        service = IndexingService()
        service.delete_document_chunks(document_id)
        logger.info("Deleted prior chunks for document %s", document_id)
    except Exception as exc:  # noqa: BLE001 — best-effort cleanup
        # 清理失败不应阻塞 reprocess。新切片会用新的 UUID，多数情况下
        # ``index_chunks`` 的 upsert 也能写新数据；遗留点只是召回噪声，
        # 后续运维清理即可。
        logger.warning(
            "Cleanup failed for document %s: %s; continuing with reindex",
            document_id,
            exc,
        )

    return process_result


def _markdown_to_pipeline_blocks(markdown: str) -> list[dict]:
    """把修正后的 Markdown 转成 ``chunk_document`` 期望的 block 列表。

    ``chunk_document`` 消费的是形如 ``{type, text, page_number, style}`` 的
    dict 列表（见 :func:`chunk_document` 的实现）。修正后的 Markdown 已经
    是「干净」的，所以这里只做最小拆分：

    - ATX 标题（``#`` 到 ``######``）→ ``type='heading'``，
      ``style={'heading_level': N}``。
    - 空行作为段落分隔符 → 每段成为一个 ``type='paragraph'`` block。
    - Fenced code block（```` ``` ````）保持为一个段落 block，
      内容含围栏字符，不再做语法识别。
    - 表格 / 列表等不再单独识别——分块器（``chunk_document``）下游会按
      token 数兜底切分，避免我们在这里做重复且可能出错的解析。

    Args:
        markdown: 用户提交的修正后 Markdown 文本。

    Returns:
        ``chunk_document`` 可直接消费的 block 列表。空字符串返回 ``[]``。
    """
    import re as _re

    blocks: list[dict] = []
    if not markdown or not markdown.strip():
        return blocks

    current: list[str] = []
    in_fence = False

    def _flush_paragraph() -> None:
        if not current:
            return
        text = "\n".join(current).strip()
        current.clear()
        if text:
            blocks.append(
                {
                    "type": "paragraph",
                    "text": text,
                    "page_number": 1,
                    "style": {},
                }
            )

    for line in markdown.split("\n"):
        stripped = line.strip()

        # 围栏代码块：进入 / 退出时切换状态，块内容（含围栏）累积成段。
        if stripped.startswith("```"):
            in_fence = not in_fence
            current.append(line)
            if not in_fence:
                # 围栏闭合，整个代码块作为一个段落落盘。
                _flush_paragraph()
            continue
        if in_fence:
            current.append(line)
            continue

        # 标题：先冲洗当前段落，再单独追加 heading block。
        heading_m = _re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading_m:
            _flush_paragraph()
            level = len(heading_m.group(1))
            blocks.append(
                {
                    "type": "heading",
                    "text": heading_m.group(2).strip(),
                    "page_number": 1,
                    "style": {"heading_level": level},
                }
            )
            continue

        # 空行：段落分隔。
        if not stripped:
            _flush_paragraph()
            continue

        current.append(line)

    _flush_paragraph()
    return blocks


def submit_reprocess_from_markdown(
    document_id: str, corrected_markdown: str
) -> bool:
    """跳过 parse/profile/process 阶段，用修正后的 Markdown 直接重跑分块/向量化。

    用于人工修正流程（任务 11.11）：管理员在审核详情页提交修正后的
    Markdown 后，不再需要重新跑解析——内容已经是 reviewer 认可的
    cleaned Markdown 了。链路：

        cleanup_document_indices → chunk_document → embed_chunks → index_chunks

    ``cleanup_document_indices`` 先清掉旧的 Qdrant/OpenSearch 切片，
    随后 ``chunk_document`` 用 ``_markdown_to_pipeline_blocks`` 把 Markdown
    切回 block 列表，下游与正常管线复用同一套 chunk → embed → index 实现。

    Args:
        document_id: 文档 UUID 字符串。
        corrected_markdown: reviewer 提交的修正后 Markdown 全文。

    Returns:
        ``True`` 表示 Celery chain 已成功 apply_async；``False`` 表示
        Celery 不可用 / 提交失败（已记 WARNING）。调用方应据此构造提示
        信息并继续把 review 标记为 ``corrected``，不能因为消息队列宕机而
        让管理员卡在审核流程上。
    """
    if not CELERY_AVAILABLE:
        logger.warning(
            "Celery not available; cannot submit reprocess for document %s. "
            "Re-chunking/vectorization will be skipped.",
            document_id,
        )
        return False

    blocks = _markdown_to_pipeline_blocks(corrected_markdown)
    process_result = {
        "document_id": document_id,
        "blocks": blocks,
        "metadata": {"reprocess_source": "manual_correction"},
        "profile_id": None,
        "profile_name": None,
        "quality_score": None,
        "review_enqueued": False,
    }

    try:
        pipeline = chain(
            cleanup_document_indices.s(process_result),
            chunk_document.s(),
            embed_chunks.s(),
            index_chunks.s(),
        )
        pipeline.apply_async()
        logger.info(
            "Submitted reprocess pipeline for document %s (%d blocks from corrected markdown)",
            document_id,
            len(blocks),
        )
        return True
    except Exception as exc:  # noqa: BLE001 — broker may be unreachable
        logger.warning(
            "Failed to submit reprocess pipeline for document %s: %s",
            document_id,
            exc,
        )
        return False


# ─── 任务 17.9：批量重处理进度追踪 ─────────────────────────────────────
#
# ``FeedbackService.trigger_reprocessing`` 在入队前已经把 ``total`` /
# ``status='running'`` / ``created_at`` 写到 Redis hash
# ``reprocess:task:{task_id}``。本节负责让每个 worker 处理完单篇文档后
# 把 ``processed`` 字段原子递增 1，并在 ``processed >= total`` 时把
# ``status`` 推进到 ``completed``（全部成功）或 ``failed``
# （存在失败计数），从而满足需求 9.9：
#
# - 进度查询 API（``GET /api/admin/feedback/reprocess/{task_id}``）能在
#   任意时刻读到最新 ``processed_documents`` 与 ``progress_percent``。
# - ``status`` 状态转换：``pending → running → completed`` 或
#   ``pending → running → failed``（失败累计 ≥1 即 failed）。
# - ``progress_percent`` 始终落在 ``[0, 100]``，``total == 0`` 兜底为 0.0。


def _reprocess_progress_key(task_id: str) -> str:
    """返回 Redis 中重处理任务进度 hash 的 key 名。"""
    return f"reprocess:task:{task_id}"


def _increment_reprocess_progress(task_id: str, *, errored: bool = False) -> None:
    """递增批量重处理任务的进度计数（同步，供 Celery worker 调用）。

    - 用 ``HINCRBY processed 1`` 原子自增；
    - 失败计数额外用 ``HINCRBY errors 1`` 单独累计，便于诊断；
    - 处理完一篇就比较 ``processed`` 与 ``total``：
      * ``processed >= total`` 且 ``errors == 0`` → ``status = completed``；
      * ``processed >= total`` 且 ``errors > 0``  → ``status = failed``；
      * 否则保持 ``running``。

    Redis 不可用时仅记录 WARNING，不阻塞 worker——进度追踪是「锦上添花」
    的可观测性能力，缺失时 API 仍能返回入队时落盘的 ``total`` /
    ``status`` 快照，不会让正在跑的文档管线崩盘。

    Args:
        task_id: 批量重处理任务 ID（由 ``FeedbackService.trigger_reprocessing``
            生成）。
        errored: 单篇文档是否处理失败。``True`` 会同时递增 ``errors`` 计数，
            在 ``processed >= total`` 时把 ``status`` 收敛到 ``failed``。
    """
    try:
        import redis as redis_lib

        from app.core.config import get_settings

        settings = get_settings()
        client = redis_lib.Redis.from_url(
            settings.REDIS_URL, decode_responses=True
        )
        key = _reprocess_progress_key(task_id)

        # 1) 原子递增 processed（与可选的 errors）
        processed = client.hincrby(key, "processed", 1)
        errors = client.hincrby(key, "errors", 1) if errored else int(
            client.hget(key, "errors") or 0
        )

        # 2) 读 total 决定是否收敛 status
        try:
            total = int(client.hget(key, "total") or 0)
        except (TypeError, ValueError):
            total = 0

        if total > 0 and processed >= total:
            new_status = "failed" if errors > 0 else "completed"
            client.hset(key, "status", new_status)
            logger.info(
                "Reprocess task %s reached terminal state status=%s "
                "(processed=%d/%d, errors=%d)",
                task_id,
                new_status,
                processed,
                total,
                errors,
            )
    except Exception as exc:  # noqa: BLE001 — best-effort progress tracking
        logger.warning(
            "Failed to increment reprocess progress for task %s: %s",
            task_id,
            exc,
        )


@_task_decorator(
    bind=True,
    name="pipeline.mark_reprocess_progress",
    max_retries=0,
    soft_time_limit=10,
    time_limit=15,
)
def mark_reprocess_progress(self, prev_result, task_id: str) -> dict:
    """Celery chain 末尾节点：把单篇文档处理的成败汇总到批量任务进度。

    作为 ``reprocess_document`` 提交的 chain 的最后一步，无论 chain 是
    全部成功（透传 ``index_chunks`` 的结果 dict）还是某一步抛出异常
    （由 ``link_error`` 路径走另一个回调），都会调用本任务把进度推进 1。

    成功路径：
    - ``prev_result`` 含 ``status == 'completed'`` 视为成功。
    - 任何其它形态 / 缺字段都视为失败，让 ``status`` 收敛到 ``failed``。

    Args:
        prev_result: 上一步任务（通常是 ``index_chunks``）的返回值。
        task_id: 批量重处理任务 ID。

    Returns:
        透传 ``prev_result``，便于 Celery 链上调试。
    """
    errored = not (
        isinstance(prev_result, dict) and prev_result.get("status") == "completed"
    )
    _increment_reprocess_progress(task_id, errored=errored)
    return prev_result if isinstance(prev_result, dict) else {}


@_task_decorator(
    bind=True,
    name="app.tasks.reprocess_document",
    max_retries=0,
    soft_time_limit=10,
    time_limit=15,
)
def reprocess_document(self, document_id: str, task_id: str) -> dict:
    """批量重处理调度任务（任务 17.9 / 需求 9.9）。

    ``FeedbackService.trigger_reprocessing`` 为每篇受影响文档投递一条本任务，
    队列固定为 ``reprocessing``，与主 pipeline 队列隔离避免互相争抢。

    本任务不做实际解析工作，只负责构造完整的处理 chain 并附加进度回调：

        parse → profile_match → universal_parser_check → process →
        chunk → embed → index → mark_reprocess_progress(task_id)

    末尾的 ``mark_reprocess_progress`` 会把 ``processed`` 计数 +1，并在
    ``processed >= total`` 时把 ``status`` 推进到 ``completed`` 或
    ``failed``，这是需求 9.9 的核心约束。

    错误兜底：
    - chain 中任一步骤抛异常 → ``link_error`` 也会驱动一次进度递增，
      避免失败文档把整个批量任务的 ``status`` 卡在 ``running``。
    - Celery 不可用 → 直接以「失败」标记当前文档的进度，让 API 仍能
      返回最终态（不会无限轮询）。

    Args:
        document_id: 要重处理的文档 UUID。
        task_id: 批量重处理任务 ID（由调用方写入 Redis 的 hash key）。

    Returns:
        ``{"document_id": ..., "task_id": ..., "submitted": bool}``。
    """
    if not CELERY_AVAILABLE:
        # 没有 Celery 时无法构造 chain；直接把进度记为「该文档失败」，
        # 让批量任务能在所有文档处理后正常收敛到 failed。
        logger.warning(
            "Celery not available; marking reprocess of %s as failed "
            "(task_id=%s)",
            document_id,
            task_id,
        )
        _increment_reprocess_progress(task_id, errored=True)
        return {
            "document_id": document_id,
            "task_id": task_id,
            "submitted": False,
        }

    try:
        pipeline = chain(
            parse_document.s(document_id),
            profile_match.s(),
            universal_parser_check.s(),
            process_document.s(),
            chunk_document.s(),
            embed_chunks.s(),
            index_chunks.s(),
            mark_reprocess_progress.s(task_id),
        )
        # link_error 让 chain 失败时仍能推进进度计数（errored=True）。
        pipeline.apply_async(
            link_error=mark_reprocess_progress.s(task_id),
        )
        logger.info(
            "Submitted reprocess chain for document %s (task_id=%s)",
            document_id,
            task_id,
        )
        return {
            "document_id": document_id,
            "task_id": task_id,
            "submitted": True,
        }
    except Exception as exc:  # noqa: BLE001 — broker may be unreachable
        logger.warning(
            "Failed to submit reprocess chain for %s (task_id=%s): %s",
            document_id,
            task_id,
            exc,
        )
        _increment_reprocess_progress(task_id, errored=True)
        return {
            "document_id": document_id,
            "task_id": task_id,
            "submitted": False,
        }


# ─── Helper Functions ──────────────────────────────────────────────────


def _load_profiles_from_db() -> list:
    """Load all enabled profiles from the database (synchronous for Celery).

    Returns:
        List of DocumentProfileConfig instances
    """
    try:
        from sqlalchemy import create_engine, text

        from app.core.config import get_settings
        from app.services.profile_matcher import profile_from_dict

        settings = get_settings()
        sync_url = settings.DATABASE_URL.replace("+asyncpg", "+psycopg2")
        engine = create_engine(sync_url)

        with engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT id, name, description, priority, enabled, "
                    "match_rules, heading_rules, boilerplate, tables, chunking, "
                    "domain_dictionary_id, version, created_at, updated_at "
                    "FROM document_profiles WHERE enabled = true "
                    "ORDER BY priority DESC"
                )
            )
            rows = result.fetchall()

        profiles = []
        for row in rows:
            profile_data = {
                "id": str(row[0]),
                "name": row[1],
                "description": row[2] or "",
                "priority": row[3],
                "enabled": row[4],
                "match_rules": row[5] or {},
                "heading_rules": row[6] or [],
                "boilerplate": row[7] or {},
                "tables": row[8] or {},
                "chunking": row[9] or {},
                "domain_dictionary_id": str(row[10]) if row[10] else None,
                "version": row[11],
                "created_at": row[12],
                "updated_at": row[13],
            }
            profiles.append(profile_from_dict(profile_data))

        return profiles

    except Exception as e:
        logger.error(f"Failed to load profiles from database: {e}")
        return []


def _get_document_info(document_id: str) -> dict | None:
    """Get document info from database (synchronous).

    Args:
        document_id: Document UUID

    Returns:
        Dict with storage_path and file_type, or None if not found
    """
    try:
        from sqlalchemy import create_engine, text

        from app.core.config import get_settings

        settings = get_settings()
        # Use synchronous connection for Celery tasks
        sync_url = settings.DATABASE_URL.replace("+asyncpg", "+psycopg2")
        engine = create_engine(sync_url)

        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT storage_path, file_type FROM documents WHERE id = :id"),
                {"id": document_id},
            )
            row = result.fetchone()
            if row:
                return {"storage_path": row[0], "file_type": row[1]}
            return None
    except Exception as e:
        logger.error(f"Failed to get document info for {document_id}: {e}")
        return None


def _get_mime_type(file_type: str) -> str:
    """Map file type extension to MIME type."""
    mime_map = {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "txt": "text/plain",
        "md": "text/markdown",
        "html": "text/html",
    }
    return mime_map.get(file_type, "application/octet-stream")


def _get_document_space_id(document_id: str) -> str | None:
    """Get the space_id for a document from the database (synchronous).

    Args:
        document_id: Document UUID

    Returns:
        Space ID string or None if not found
    """
    try:
        from sqlalchemy import create_engine, text

        from app.core.config import get_settings

        settings = get_settings()
        sync_url = settings.DATABASE_URL.replace("+asyncpg", "+psycopg2")
        engine = create_engine(sync_url)

        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT space_id FROM documents WHERE id = :id"),
                {"id": document_id},
            )
            row = result.fetchone()
            if row and row[0]:
                return str(row[0])
            return None
    except Exception as e:
        logger.error(f"Failed to get space_id for {document_id}: {e}")
        return None


def _ensure_default_parsers_registered(registry) -> None:
    """Ensure default parsers are registered if registry is empty."""
    if registry.plugins:
        return

    from app.services.parsers.docx_parser import DocxParser
    from app.services.parsers.html_parser import HtmlParser
    from app.services.parsers.pdf_parser import PdfParser
    from app.services.parsers.pptx_parser import PptxParser
    from app.services.parsers.text_parser import TextParser

    for parser_class in [PdfParser, DocxParser, PptxParser, HtmlParser, TextParser]:
        try:
            registry.register(parser_class())
        except ValueError:
            pass  # Already registered


# ─── 任务 11: Quality Scoring + Review Queue helpers ──────────────────


def _resolve_profile_for_processing(
    profile_id: str | None,
    profile_name: str | None,
):
    """加载用于 ``process_document`` 的 ``DocumentProfileConfig``。

    优先按 ``profile_id`` 精确加载；找不到时退化为按 ``name='generic-text'``
    加载兜底 Profile；再失败则返回一个最小可用的 in-memory generic-text，
    保证 ``DocumentProcessor`` 总有 Profile 可用、不会让管线在评分阶段卡死。
    """
    from sqlalchemy import create_engine, text

    from app.core.config import get_settings
    from app.services.profile_matcher import (
        BoilerplateConfig,
        ChunkingConfig,
        DocumentProfileConfig,
        MatchRules,
        TableConfig,
        profile_from_dict,
    )

    def _row_to_profile(row):
        return profile_from_dict(
            {
                "id": str(row[0]),
                "name": row[1],
                "description": row[2] or "",
                "priority": row[3],
                "enabled": row[4],
                "match_rules": row[5] or {},
                "heading_rules": row[6] or [],
                "boilerplate": row[7] or {},
                "tables": row[8] or {},
                "chunking": row[9] or {},
                "domain_dictionary_id": str(row[10]) if row[10] else None,
                "version": row[11],
            }
        )

    select_sql = (
        "SELECT id, name, description, priority, enabled, "
        "match_rules, heading_rules, boilerplate, tables, chunking, "
        "domain_dictionary_id, version "
        "FROM document_profiles WHERE {where} LIMIT 1"
    )

    try:
        settings = get_settings()
        sync_url = settings.DATABASE_URL.replace("+asyncpg", "+psycopg2")
        engine = create_engine(sync_url)
        with engine.connect() as conn:
            row = None
            if profile_id:
                row = conn.execute(
                    text(select_sql.format(where="id = :pid")),
                    {"pid": profile_id},
                ).fetchone()
            if row is None:
                target_name = profile_name or "generic-text"
                row = conn.execute(
                    text(select_sql.format(where="name = :pname")),
                    {"pname": target_name},
                ).fetchone()
            if row is None:
                row = conn.execute(
                    text(select_sql.format(where="name = :pname")),
                    {"pname": "generic-text"},
                ).fetchone()
        if row is not None:
            return _row_to_profile(row)
    except Exception as exc:  # noqa: BLE001 — defensive: DB might not be reachable in tests
        logger.warning(
            "process_document: failed to load profile from DB (id=%s name=%s): %s; "
            "falling back to in-memory generic-text",
            profile_id,
            profile_name,
            exc,
        )

    # 数据库不可达或没有任何 generic-text 行——构造一个最小可用 Profile。
    return DocumentProfileConfig(
        id="default",
        name="generic-text",
        description="in-memory fallback",
        priority=0,
        enabled=True,
        match_rules=MatchRules(),
        heading_rules=[],
        boilerplate=BoilerplateConfig(),
        tables=TableConfig(),
        chunking=ChunkingConfig(),
    )


def _persist_document_quality_score(document_id: str, score_dict: dict) -> None:
    """把 ``ParseQualityScore.to_dict()`` 写入 documents.quality_score JSONB 列。

    与 ``update_document_db_status`` 保持同样的「同步 psycopg2 + 自管事务」
    风格：Celery worker 跑在同步上下文里，不引入 asyncpg 事件循环。
    """
    try:
        import json as _json

        from sqlalchemy import create_engine, text

        from app.core.config import get_settings

        settings = get_settings()
        sync_url = settings.DATABASE_URL.replace("+asyncpg", "+psycopg2")
        engine = create_engine(sync_url)

        with engine.connect() as conn:
            conn.execute(
                text(
                    "UPDATE documents SET quality_score = CAST(:payload AS JSONB), "
                    "last_status_update = NOW() WHERE id = :id"
                ),
                {"id": document_id, "payload": _json.dumps(score_dict)},
            )
            conn.commit()
    except Exception as exc:  # noqa: BLE001 — defensive
        logger.warning(
            "process_document: failed to persist quality_score for %s: %s",
            document_id,
            exc,
        )


def _enqueue_for_review_sync(
    document_id: str,
    score,
    parsed_markdown: str | None = None,
) -> bool:
    """同步包装 ``ReviewQueue.enqueue``，在 Celery worker 里跑。

    与 ``universal_parser_check`` 一致，使用 ``asyncio.new_event_loop()`` +
    ``AsyncSessionLocal`` 临时打开一次 AsyncSession；事务由本函数显式
    ``commit()`` 收尾，与 FastAPI ``get_db`` 的语义保持一致。

    Args:
        document_id: 文档 UUID（字符串）。
        score: ``ParseQualityScore`` 实例（任务里直接传进来，避免反序列化）。
        parsed_markdown: 可选，``ProcessedDocument.markdown`` 的完整内容。任务
            11.10「文档并排预览 API」用它把清洗后的 Markdown 落进
            ``DocumentReview.quality_score`` JSONB（key: ``parsed_markdown``），
            让审核详情页无需再跑一次解析就能渲染右栏。

    Returns:
        是否成功入队（包括「幂等命中已存在 pending 行」也算成功）。
    """
    import asyncio

    from app.services.review_queue import ReviewQueue

    extra_payload: dict | None = None
    if parsed_markdown:
        extra_payload = {"parsed_markdown": parsed_markdown}

    async def _run() -> bool:
        from app.core.database import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            queue = ReviewQueue(session)
            await queue.enqueue(document_id, score, extra_payload=extra_payload)
            await session.commit()
            return True

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_run())
    finally:
        loop.close()
