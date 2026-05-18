# Wikforge Roadmap

> 共 55 项, 按层级 A-G 分类。✅ 已完成 / 🚧 进行中 / ⬜ 待做。

总工时估算: **约 6-8 个工作日** 全做完。
内网生产推荐顺序: **A → B → C → D → F → G** (≈ 2.5 工作日)。

---

## A. Bug / 不一致 (必修)

| ID | 项目 | 工时 | 状态 |
|---|---|---|---|
| A1 | `list_spaces` 不做权限过滤 | 15 分钟 | ✅ |
| A2 | `list_documents` / `list_folders` 是否漏过滤 | 30 分钟 | ✅ |
| A3 | 旧 chunk score 不一致 (重建索引) | 15 分钟 | ✅ (随 IK 装时一起重建) |
| A4 | EmbeddingService 走 LiteLLM Proxy 统一管控 | 30 分钟 | ✅ (验证通过 — `LITELLM_API_BASE=http://litellm:4000` 已生效) |
| A5 | LiteLLM Proxy 的 Redis 缓存生效验证 | 15 分钟 | ✅ (验证通过 — Redis DB 1 有 cache key, 重复请求秒返) |
| A6 | upload_files 的 commit 应该让 dependency 兜底 | 30 分钟 | ⬜ |
| A7 | smoke-test.sh "创建空间失败" 提示优化 | 5 分钟 | ⬜ |

## B. 检索质量 (中文核心)

| ID | 项目 | 工时 | 状态 |
|---|---|---|---|
| B1 | OpenSearch 装 IK 中文分词器 | 30 分钟 | ⬜ |
| B2 | 真 Cross-Encoder reranker (BAAI/bge-reranker-base) | 1.5 小时 | ⬜ |
| B3 | Bigram fallback 加日志告警 | 15 分钟 | ✅ (已做) |
| B4 | Profile 匹配实测 (国标 + 扫描版 PDF) | 30 分钟 | ⬜ |
| B5 | Universal Parser LLM 兜底实测 | 30 分钟 | ⬜ |
| B6 | 查询增强 (rewrite/HyDE/decomp) 实测 | 20 分钟 | ⬜ |

## C. 工程加固

| ID | 项目 | 工时 | 状态 |
|---|---|---|---|
| C1 | storage_path 改 presigned URL | 30 分钟 | ⬜ |
| C2 | MinIO 大文件 multipart 上传 | 30 分钟 | ⬜ |
| C3 | Celery 任务统一时间限制策略 | 30 分钟 | ⬜ |
| C4 | api 容器 `--reload` 在生产关闭 | 5 分钟 | ⬜ |
| C5 | api/worker .env 注入逻辑统一 | 15 分钟 | ⬜ |
| C6 | 升级到 query_points, 解锁 qdrant-client 1.15+ | 1.5 小时 | ⬜ |
| C7 | LiteLLM Proxy 健康检查 start_period 加宽 | 15 分钟 | ⬜ |
| C8 | next.config eslint dirs 改 ignoreDuringBuilds | 5 分钟 | ⬜ |
| C9 | qdrant collection / opensearch index 启动时自动 ensure | 15 分钟 | ⬜ |

## D. 部署 / 运维

| ID | 项目 | 工时 | 状态 |
|---|---|---|---|
| D1 | Docker registry mirror 配置 | 5 分钟 | ⬜ |
| D2 | 备份 cron 脚本 (postgres + minio + qdrant) | 30 分钟 | ⬜ |
| D3 | nginx 反代配置 (终止 TLS + SSE proxy_buffering) | 30 分钟 | ⬜ |
| D4 | SSE proxy_buffering off 示例 (合并到 D3) | - | - |
| D5 | docker-compose 加 logging driver max-size | 15 分钟 | ⬜ |
| D6 | Makefile 加 rebuild / reset-data / log-tail | 10 分钟 | ⬜ |
| D7 | docs/deploy.md 补充本次踩坑修复 | 30 分钟 | ⬜ |
| D8 | docs/architecture.md 系统架构图 + 调用链 | 1 小时 | ⬜ |
| D9 | 全局 health-check API (前端 dashboard 显示状态) | 1 小时 | ⬜ |

## E. 用户体验 / 功能完善

| ID | 项目 | 工时 | 状态 |
|---|---|---|---|
| E1 | 修改密码功能 | 1 小时 | ⬜ |
| E2 | 修改邮箱 / 显示名 | 30 分钟 | ⬜ |
| E3 | 文档版本管理 / 回滚 | 1 天 | ⬜ |
| E4 | 文档操作审计日志 | 4 小时 | ⬜ |
| E5 | RAG 答案 赞/踩 反馈 (打通 feedback 表) | 4 小时 | ⬜ |
| E6 | 搜索结果 CSV 导出 | 30 分钟 | ⬜ |
| E7 | 文档批量操作 (批量删除/移动/标签) | 4 小时 | ⬜ |
| E8 | 移动端响应式补强 | 4 小时 | ⬜ |
| E9 | 国际化 (i18n) | 1 天 | ⬜ |
| E10 | 暗色模式细节修整 | 2 小时 | ⬜ |
| E11 | 搜索结果 "在文档中查看" 高亮跳转 | 1 小时 | ⬜ |
| E12 | RAG 流式中途停止按钮 (abort signal) | 30 分钟 | ⬜ |

## F. 性能 / 扩展性

| ID | 项目 | 工时 | 状态 |
|---|---|---|---|
| F1 | api 多 worker (gunicorn / uvicorn workers) | 15 分钟 | ⬜ |
| F2 | Celery worker concurrency 调优 | 15 分钟 | ⬜ |
| F3 | Postgres 连接池调优 | 15 分钟 | ⬜ |
| F4 | Redis 持久化 (AOF) | 15 分钟 | ⬜ |
| F5 | OpenSearch JVM heap 调优 | 15 分钟 | ⬜ |
| F6 | Qdrant HNSW 参数调优 | 30 分钟 | ⬜ |
| F7 | LiteLLM Proxy 限流 / 配额 | 30 分钟 | ⬜ |

## G. 测试 / 质量保障

| ID | 项目 | 工时 | 状态 |
|---|---|---|---|
| G1 | smoke-test.sh 改进 (清理脏数据 + 更多 API) | 30 分钟 | ⬜ |
| G2 | 集成测试在 CI 中跑通 | 2 小时 | ⬜ |
| G3 | 检索质量评估脚本跑一遍 (Recall@K / MRR / NDCG) | 30 分钟 | ⬜ |
| G4 | Playwright E2E 在容器里跑 | 1 小时 | ⬜ |
| G5 | 后端单元测试覆盖率核查 | 30 分钟 | ⬜ |

---

## 已完成 (本轮)

> 2026-05-17 启动初期踩坑 + 修复

- ✅ 12 个部署坑修复 (Redis 端口冲突 / Postgres 5432 / Frontend build / Qdrant healthcheck / Alembic ENUM / OpenSearch SSL / qdrant-client / IK fallback / admin email / batch_size / CORS / env_file)
- ✅ LiteLLM Proxy 独立部署 + Admin UI
- ✅ 自动迁移 + 自动 init_db (api-entrypoint.sh)
- ✅ Makefile 14 个命令
- ✅ docs/deploy.md 完整部署手册
- ✅ 冒烟测试脚本
- ✅ 创建空间自动播种 owner write 权限
- ✅ 中文 bigram fallback rerank
- ✅ Embedding 自动加 openai/ 前缀
- ✅ LLMGateway drop_params=True (gpt-5 兼容)
- ✅ Logo + favicon (区分 Dify)
- ✅ 用户管理 API (`/api/admin/users`)
- ✅ 系统监控 API (`/api/admin/monitoring`)
- ✅ LLM 配置页改为跳转 LiteLLM Admin UI
- ✅ Git 仓库初始化 + 推送 GitHub (private)

> 2026-05-18 PDF 解析 OOM 修复 (夜班作业 ABCD 收尾)

- ✅ **PDF 解析 OOM 死循环修复**: marker / surya 模型权重 (~3GB) 在 Docker VM (默认 7.65GB) 单 worker 加载就压死 OOM SIGKILL,导致解析永远卡在 "parsing" 状态。
  - `worker_max_memory_per_child` 512MB → 4GB (graceful 上限,避免软杀)
  - 模型缓存目录: `MODEL_CACHE_DIR=/models/datalab/models` (BaseSettings 字段名,之前的 `DATALAB_CACHE_DIR` 完全无效),配合 `models_cache` volume 让 1.34GB+1.34GB+1.4GB 模型只下载一次
  - api 容器同步加 HF / SENTENCE_TRANSFORMERS / models_cache,让 Cross-Encoder reranker (BAAI/bge-reranker-base) 也走共享缓存
  - **新增 `PDF_PARSER_MODE=auto` 模式**: 文件 <2MB (默认) 走 PyMuPDF/fitz (内存 <100MB,几秒解析完),大文件才走 marker;低内存环境可显式 `PDF_PARSER_MODE=fitz`
  - worker concurrency: 4 → 1 (marker fork 多并发会内存翻倍)
- ✅ **`POST /api/documents/{id}/retry` 真的 enqueue Celery**: 之前的 retry 只改 db status=pending,从来没把任务塞进 Celery 队列,导致 retry 后文档卡死 pending。现在调 `submit_pipeline()` 与 upload_files / import_url 行为一致。
- ✅ **OpenSearch 磁盘水位线放宽**: low/high/flood_stage 调到 95/97/99% (开发场景),避免 30G/32G 满了就触发 flood_stage 把 index 设为 read-only
- ✅ **watchdog SQL enum cast bug**: `WHERE status = ANY(:states)` 在 PostgreSQL 里 enum vs text[] 不匹配,加 `status::text` 强转,watchdog 任务恢复正常
- ✅ **端到端验证**: 两个真实 PDF (简历 222KB / 技术规范 661KB) 全程 parse → profile_match → universal_parser_check → process → chunk → embed → index 跑通,status=completed,progress=100%
