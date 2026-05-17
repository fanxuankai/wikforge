"""文档处理流水线的看门狗任务。

定期扫描 ``documents`` 表,把长时间停留在中间状态(parsing/cleaning/...)
但 ``last_status_update`` 已超过阈值的记录标记成 ``failed``。
适用场景:
- worker 进程被 SIGKILL (OOM / time_limit) 后状态没机会回滚
- broker 短暂不可用导致 retry 链断裂
- 容器被强制重启

使用方式:
- Celery beat 周期触发 (``app.tasks.watchdog.reap_stuck_documents``)
- 也可以直接 ``celery -A app.core.celery_app call ...`` 手动触发

不会触碰 ``status='completed'`` / ``'failed'`` / ``'pending'`` 的文档。
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from app.core.celery_app import celery_app
from app.core.config import get_settings
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


# 超过这个时间没更新状态就视为卡住 (单位: 秒)
# 当前最长任务 (parse_document/embed_chunks) 是 600s,
# 加上 retry 3 次的最坏情况 (10 + 20 + 40 + 600*3 ≈ 30 分钟),
# 留 5 分钟缓冲,统一用 35 分钟。
DEFAULT_STUCK_THRESHOLD_SECONDS = 35 * 60


# 中间状态: 进了 worker 流水线但还没到 completed/failed
INTERMEDIATE_STATES = (
    "parsing",
    "cleaning",
    "chunking",
    "embedding",
    "indexing",
)


@celery_app.task(
    name="watchdog.reap_stuck_documents",
    soft_time_limit=50,
    time_limit=60,
)
def reap_stuck_documents(threshold_seconds: int | None = None) -> dict:
    """扫描并把长时间卡住的文档标记成 ``failed``。

    Args:
        threshold_seconds: 卡住超过多久判定为僵尸,默认 35 分钟。

    Returns:
        ``{"reaped": <count>, "ids": [<uuid>, ...]}``
    """
    threshold = threshold_seconds or DEFAULT_STUCK_THRESHOLD_SECONDS
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=threshold)

    # Celery 任务里用同步 engine, 比 asyncpg 简单
    settings = get_settings()
    sync_url = (
        f"postgresql+psycopg2://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    )
    sync_engine = create_engine(sync_url, pool_pre_ping=True)

    sql = text(
        """
        UPDATE documents
           SET status = 'failed',
               error_detail = COALESCE(error_detail, '')
                              || CASE WHEN error_detail IS NULL OR error_detail = ''
                                      THEN ''
                                      ELSE ' | ' END
                              || 'watchdog: stuck in ' || status::text
                              || ' for >' || :threshold_min || 'min',
               last_status_update = NOW()
         WHERE status = ANY(:states)
           AND last_status_update < :cutoff
        RETURNING id, title
        """
    )

    with sync_engine.begin() as conn:
        result = conn.execute(
            sql,
            {
                "threshold_min": threshold // 60,
                "states": list(INTERMEDIATE_STATES),
                "cutoff": cutoff,
            },
        )
        rows = result.fetchall()

    reaped_ids = [str(r[0]) for r in rows]
    if reaped_ids:
        logger.warning(
            "Watchdog reaped %d stuck documents: %s", len(reaped_ids), reaped_ids
        )
    else:
        logger.debug("Watchdog: no stuck documents found (threshold=%ds)", threshold)

    return {"reaped": len(reaped_ids), "ids": reaped_ids}
