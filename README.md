<div align="center">

<img src="docs/assets/banner.svg" alt="Wikforge" width="100%"/>

<br/>

<p>
  <a href="#-quick-start"><img src="https://img.shields.io/badge/部署-make_first--run-7C3AED?style=for-the-badge&logo=docker&logoColor=white" alt="Quick Start"/></a>
  <a href="docs/deploy.md"><img src="https://img.shields.io/badge/文档-deploy.md-4F46E5?style=for-the-badge&logo=readthedocs&logoColor=white" alt="Docs"/></a>
  <a href="#-roadmap"><img src="https://img.shields.io/badge/Roadmap-55_项-22C55E?style=for-the-badge&logo=todoist&logoColor=white" alt="Roadmap"/></a>
  <a href="https://github.com/fanxuankai/wikforge/actions/workflows/ci.yml"><img src="https://github.com/fanxuankai/wikforge/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"/></a>
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python 3.11"/>
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Celery-5.4-37814A?logo=celery&logoColor=white" alt="Celery"/>
  <img src="https://img.shields.io/badge/Next.js-14-000000?logo=nextdotjs&logoColor=white" alt="Next.js 14"/>
  <img src="https://img.shields.io/badge/TypeScript-5.x-3178C6?logo=typescript&logoColor=white" alt="TypeScript"/>
  <img src="https://img.shields.io/badge/Tailwind-3.4-06B6D4?logo=tailwindcss&logoColor=white" alt="Tailwind"/>
</p>

<p>
  <img src="https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white" alt="PostgreSQL"/>
  <img src="https://img.shields.io/badge/Redis-7-DC382D?logo=redis&logoColor=white" alt="Redis"/>
  <img src="https://img.shields.io/badge/OpenSearch-2.17-005EB8?logo=opensearch&logoColor=white" alt="OpenSearch"/>
  <img src="https://img.shields.io/badge/Qdrant-1.14-DC382D?logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0iI0RDMzgyRCIgZD0iTTEyIDJsOSA1djEwbC05IDUtOS01VjdsOS01eiIvPjwvc3ZnPg==&logoColor=white" alt="Qdrant"/>
  <img src="https://img.shields.io/badge/MinIO-S3-C72E49?logo=minio&logoColor=white" alt="MinIO"/>
  <img src="https://img.shields.io/badge/LiteLLM-Proxy-8B5CF6" alt="LiteLLM"/>
</p>

<p>
  <strong>企业级 RAG 知识库 · 文档解析 · 智能检索 · 流式问答 · 反馈闭环</strong>
</p>

</div>

<br/>

## ✨ 核心能力

```
📄 文档解析           插件式架构: PDF / DOCX / Markdown / HTML / 源代码 + LLM 视觉兜底
🔍 复合搜索           BM25 + Dense Vector + Sparse Vector + RRF 融合 + Cross-Encoder 重排
🎯 Profile 系统       自动匹配文档类型 (通用文本 / 中式技术规范 / 扫描版 PDF)
💡 查询增强           LLM 改写 / HyDE 假设文档 / 多子查询分解, 三档独立开关
🤖 流式 RAG           SSE 输出 + 引用标注 + 会话记忆, 首 token < 5s
🔁 反馈闭环           错误模式聚合 → 优化建议 → 一键应用 → 批量重处理
📚 领域词典           术语标准化 + 同义词扩展 + 候选词审核
🔐 权限隔离           Pre-Filtering 在向量层与全文层同时生效, 50ms 内完成判定
🛡️ 审核队列           解析质量评分 + 人工修正 + Profile 反向优化
📊 后台管理           空间 / 用户 / 权限 / Profile / 词典 / 反馈 / 监控 / LLM 网关
```

## 📸 产品截图

<table>
  <tr>
    <td width="50%">
      <img src="docs/screenshots/01-landing.png" alt="落地页"/>
      <p align="center"><sub><b>落地页</b> · 紫色品牌色 + 三入口</sub></p>
    </td>
    <td width="50%">
      <img src="docs/screenshots/02-login.png" alt="登录"/>
      <p align="center"><sub><b>登录</b> · 邮箱密码 / OIDC 单点</sub></p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <img src="docs/screenshots/03-documents.png" alt="文档管理"/>
      <p align="center"><sub><b>文档管理</b> · 多文件上传 + 实时进度 + 标签/移动/批量</sub></p>
    </td>
    <td width="50%">
      <img src="docs/screenshots/04-chat.png" alt="RAG 流式问答"/>
      <p align="center"><sub><b>RAG 流式问答</b> · Markdown 渲染 + 引用标注 + 会话记忆</sub></p>
    </td>
  </tr>
  <tr>
    <td colspan="2">
      <img src="docs/screenshots/05-monitoring.png" alt="系统监控"/>
      <p align="center"><sub><b>系统监控</b> · 文档队列状态 + CPU / 内存 / 磁盘实时</sub></p>
    </td>
  </tr>
</table>

## 🚀 Quick Start

```bash
# 1. 配置环境变量
cp .env.example .env
make secrets         # 生成强随机密钥, 拷贝到 .env
vim .env             # 至少填: CPA_API_KEY (Chat) / DASHSCOPE_API_KEY (Embedding)

# 2. 一键拉起 9 个服务
make first-run

# 3. 验证 (走一遍 登录→上传→搜索→RAG)
./scripts/smoke-test.sh

# 4. 访问
open http://localhost                # 前端 (admin@wikforge.com / Admin@123)
open http://localhost:8000/docs      # API 文档 (Swagger)
open http://localhost:4000/ui        # LiteLLM Admin
open http://localhost:9001           # MinIO Console
open http://localhost:6333/dashboard # Qdrant Dashboard
```

完整部署 / 升级 / 备份 / 排错请看 [`docs/deploy.md`](docs/deploy.md)。

## 🏗️ 架构

```mermaid
flowchart LR
    User([👤 用户])
    Browser[🌐 浏览器]

    subgraph Wikforge["🚀 Wikforge"]
        FE[Next.js 14<br/>前端]
        API[FastAPI<br/>API Server]
        Worker[Celery<br/>Worker × 4]
        LiteLLM[LiteLLM Proxy<br/>多模型网关]
    end

    subgraph Storage["💾 存储层"]
        PG[(PostgreSQL<br/>元数据)]
        Redis[(Redis<br/>缓存/Broker)]
        OS[(OpenSearch<br/>BM25)]
        QD[(Qdrant<br/>向量)]
        MinIO[(MinIO<br/>对象存储)]
    end

    subgraph Upstream["☁️ 上游模型"]
        CPA[CPA 网关<br/>gpt-5.5 / claude / qwen]
        DS[阿里百炼<br/>text-embedding-v4]
    end

    User --> Browser
    Browser <--> FE
    FE <--> API
    API <--> PG & Redis & OS & QD & MinIO & LiteLLM
    API -.触发任务.-> Worker
    Worker <--> PG & Redis & OS & QD & MinIO & LiteLLM
    LiteLLM --> CPA & DS

    classDef fe fill:#3B82F6,stroke:#1E40AF,color:#fff
    classDef api fill:#10B981,stroke:#047857,color:#fff
    classDef worker fill:#F59E0B,stroke:#B45309,color:#fff
    classDef llm fill:#8B5CF6,stroke:#5B21B6,color:#fff
    classDef store fill:#1F2937,stroke:#111827,color:#fff
    classDef upstream fill:#EC4899,stroke:#9D174D,color:#fff

    class FE fe
    class API api
    class Worker worker
    class LiteLLM llm
    class PG,Redis,OS,QD,MinIO store
    class CPA,DS upstream
```

## 📥 文档处理管线

```mermaid
flowchart LR
    Upload[📤 上传<br/>API] --> Parse[🔍 parse_document<br/>原生解析]
    Parse --> Profile[🎯 profile_match<br/>规则匹配]
    Profile --> Universal[🤖 universal_parser_check<br/>LLM 兜底]
    Universal --> Process[⚙️ process_document<br/>清洗+评分]
    Process -- 质量低 --> Review[👁️ 审核队列]
    Process --> Chunk[✂️ chunk_document<br/>分块]
    Chunk --> Embed[🧮 embed_chunks<br/>向量化]
    Embed --> Index[📦 index_chunks<br/>双索引入库]
    Index --> Done([✅ completed])

    Review -. 人工修正 .-> Process

    classDef ok fill:#22C55E,stroke:#15803D,color:#fff
    classDef llm fill:#8B5CF6,stroke:#5B21B6,color:#fff
    classDef warn fill:#F97316,stroke:#9A3412,color:#fff

    class Done ok
    class Universal,Embed llm
    class Review warn
```

## 🔍 检索与问答管线

```mermaid
flowchart TB
    Q[用户查询] --> Enh{查询增强}
    Enh -->|改写| R1[改写×N]
    Enh -->|HyDE| R2[假设文档]
    Enh -->|分解| R3[子查询×N]

    R1 & R2 & R3 --> Multi{多路召回}
    Multi -->|BM25| OS[(OpenSearch)]
    Multi -->|Dense| QD1[(Qdrant Dense)]
    Multi -->|Sparse| QD2[(Qdrant Sparse)]

    OS & QD1 & QD2 --> RRF[🔀 RRF Fusion<br/>k=60]
    RRF --> Rerank[🎯 Cross-Encoder<br/>BGE-Reranker]
    Rerank --> Filter[阈值过滤]
    Filter -->|有结果| LLM[💬 LLM 合成<br/>+ 引用标注]
    Filter -->|无结果| Fallback[📭 兜底回复]
    LLM --> Stream[📡 SSE 流式输出]

    classDef enh fill:#A855F7,stroke:#6B21A8,color:#fff
    classDef rec fill:#3B82F6,stroke:#1E40AF,color:#fff
    classDef llm fill:#8B5CF6,stroke:#5B21B6,color:#fff

    class Enh,R1,R2,R3 enh
    class Multi,RRF,Rerank rec
    class LLM,Stream llm
```

## 📦 技术栈

| 层 | 组件 | 版本 | 职责 |
|---|---|---|---|
| **API** | FastAPI | 0.115+ | 异步 HTTP / OpenAPI 文档 |
| **ORM** | SQLAlchemy | 2.0 (async) | 类型安全的 DB 访问 |
| **Worker** | Celery | 5.4 | 文档处理流水线 |
| **迁移** | Alembic | 1.13+ | 数据库版本管理 |
| **前端** | Next.js | 14 (App Router) | React Server Components |
| **样式** | Tailwind + shadcn/ui | 3.4 | 设计系统 |
| **状态** | Zustand | 5 | 轻量状态管理 |
| **数据库** | PostgreSQL | 16 | 主数据 + JSONB 配置 |
| **缓存** | Redis | 7 | Celery broker / 进度 / 会话 |
| **全文** | OpenSearch | 2.17 | BM25 + 中文分词 (IK 可选) |
| **向量** | Qdrant | 1.14 | Dense (1024d) + Sparse (TF-IDF) |
| **存储** | MinIO | 2025 | S3 兼容对象存储 |
| **LLM** | LiteLLM Proxy | latest | 100+ provider 统一网关 |

## 🧰 常用命令

```bash
make help            # 查看全部命令 (14 个)
make ps              # 服务状态
make logs            # 跟踪所有日志
make logs-api        # 只看 api
make logs-worker     # 只看 worker
make psql            # 进 PostgreSQL CLI
make shell-api       # 进 api 容器 bash
make migrate         # 手动跑 alembic upgrade head
make seed            # 手动 init_db (admin + 默认 Profile)
make verify          # 跑 verify_compose 完整健康检查
make secrets         # 生成一组强随机密钥
make first-run       # 新机器: 启动 + 等待 healthy + 提示访问
make down            # 停止 (保留 volume)
make reset           # 完全清理 (会丢数据!)
```

## 🗂️ 目录结构

```
wikforge/
├── backend/              # Python / FastAPI
│   ├── app/
│   │   ├── api/          # 路由层 (auth/documents/search/qa/admin_*)
│   │   ├── services/     # 业务逻辑
│   │   ├── tasks/        # Celery 任务 (pipeline.py 是核心)
│   │   ├── models/       # SQLAlchemy ORM
│   │   ├── core/         # 基础设施 (db/redis/qdrant/opensearch/minio)
│   │   └── scripts/      # init_db / api-entrypoint
│   ├── alembic/          # 数据库迁移
│   ├── tests/            # 单元 + 集成测试 (1981 个)
│   └── eval/             # 检索质量评估 (Recall@K / MRR / NDCG)
├── frontend/             # Next.js 14
│   ├── src/app/          # App Router 页面
│   ├── src/components/   # UI 组件
│   ├── src/lib/          # api-client / utils
│   └── src/stores/       # Zustand stores
├── litellm/config.yaml   # LiteLLM Proxy 模型路由配置
├── scripts/              # verify_compose / smoke-test / postgres-init
├── docs/                 # 部署文档 / 架构图 / 资源
└── docker-compose.yml    # 9 服务编排
```

## 🛣️ Roadmap

> 总计 55 项, 详情参见 [`docs/ROADMAP.md`](docs/ROADMAP.md)。

### 近期 (P1)

- [ ] **A1** `list_spaces` / `list_documents` 加权限过滤
- [ ] **B1** OpenSearch 装 IK 中文分词器 (中文召回 +30~50%)
- [ ] **B2** 真 Cross-Encoder reranker (BAAI/bge-reranker-base)
- [ ] **B4-B5** Profile 自动匹配 / LLM 兜底实测

### 中期 (P2)

- [ ] **A4** Embedding 走 LiteLLM Proxy 统一管控
- [ ] **A5** LiteLLM Redis 缓存验证
- [ ] **A6** UploadService commit 边界重构
- [ ] **C1** 文档下载用 presigned URL
- [ ] **C2** 大文件 multipart 上传
- [ ] **C6** 升级到 query_points API, 解锁 qdrant-client 1.15+

### 远期 (P3)

- [ ] **D2-D3** 备份 cron + nginx 反代 (TLS / SSE)
- [ ] **E1-E12** 用户体验扩展 (改密 / 版本管理 / 审计 / 批量操作 / i18n / 移动端)
- [ ] **F1-F7** 性能 / HA (gunicorn / Redis Sentinel / Qdrant HNSW 调优)
- [ ] **G1-G5** CI/CD 集成测试 / 检索质量自动评估

## 📚 文档

| 文档 | 说明 |
|---|---|
| [`docs/deploy.md`](docs/deploy.md) | 部署 / 升级 / 备份 / 排错完整手册 |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | 55 项修复 / 优化清单 |
| [`backend/tests/integration/README.md`](backend/tests/integration/README.md) | 集成测试运行说明 |
| [`frontend/e2e/README.md`](frontend/e2e/README.md) | Playwright E2E 说明 |
| [`scripts/README.md`](scripts/README.md) | 运维脚本说明 |

## 🔐 安全

私有仓库, 内网部署。

- ✅ `.env` 在 `.gitignore` 第 1 行, 凭证不入库
- ✅ JWT 签发 / 轮换 / 失败次数锁定
- ✅ Permission Pre-Filtering 在向量库 + 全文库 + Cache 三层一致
- ✅ Profile / 解析失败队列 + 重试上限 + 审核闭环
- ⚠️ 默认 admin 密码 `Admin@123`, **首次登录立即修改**
- ⚠️ Master Key / Secret Key 占位符为 `change-me-*`, 部署前必须替换
- ❌ 当前无 HTTPS 终止, 部署到公网前请加反向代理

## 📜 License

私有项目, All rights reserved by Jolc.

---

<div align="center">

由 ❤️ 与 ☕ 在 macOS 上锻造

<sub>Wiki + Forge — 锻造企业知识库</sub>

</div>
