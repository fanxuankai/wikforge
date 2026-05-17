# =============================================================================
# Wikforge - Makefile
#
# 常用命令:
#   make up              启动所有服务 (后台)
#   make down            停止 + 清理容器(保留 volume)
#   make reset           完全重置(清理 volume,慎用)
#   make logs            跟踪所有服务日志
#   make logs-api        跟踪 api 日志
#   make ps              查看服务状态
#   make migrate         手动跑数据库迁移 (api 启动时已自动运行)
#   make seed            手动跑 init_db (api 启动时已自动运行)
#   make verify          走一遍 verify_compose.sh 健康检查
#   make psql            进入 postgres 交互式 SQL
#   make shell-api       进入 api 容器 shell
#   make secrets         打印一组随机生成的密钥, 复制到 .env 用
#   make first-run       新机器首次部署: 检查 .env -> 启动 -> 等待健康
# =============================================================================

SHELL := /bin/bash
COMPOSE := docker compose

.PHONY: help up down reset logs logs-api logs-worker ps migrate seed verify psql shell-api shell-worker secrets first-run check-env clean clean-data prune-images smoke

help:
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?##"}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

check-env: ## 检查 .env 是否存在
	@if [ ! -f .env ]; then \
		echo "[ERROR] .env 不存在。运行: cp .env.example .env 然后编辑"; \
		exit 1; \
	fi

up: check-env ## 启动所有服务 (后台模式)
	$(COMPOSE) up -d
	@echo ""
	@echo "服务已启动。常用入口:"
	@echo "  Frontend:    http://localhost:$${FRONTEND_PORT:-80}"
	@echo "  API Docs:    http://localhost:$${API_PORT:-8000}/docs"
	@echo "  LiteLLM UI:  http://localhost:$${LITELLM_PORT:-4000}/ui"
	@echo "  MinIO Web:   http://localhost:$${MINIO_CONSOLE_PORT:-9001}"
	@echo "  Qdrant Web:  http://localhost:$${QDRANT_PORT:-6333}/dashboard"

down: ## 停止并删除容器 (保留 volume)
	$(COMPOSE) down --remove-orphans

reset: ## 完全重置(包括 volume,会丢失所有数据!)
	@read -p "确认清空所有数据? (yes/no): " ans && [ "$$ans" = "yes" ] || (echo "已取消"; exit 1)
	$(COMPOSE) down -v --remove-orphans

logs: ## 跟踪所有服务日志
	$(COMPOSE) logs -f --tail=100

logs-api: ## 跟踪 api 日志
	$(COMPOSE) logs -f --tail=200 api

logs-worker: ## 跟踪 worker 日志
	$(COMPOSE) logs -f --tail=200 worker

ps: ## 查看服务状态
	$(COMPOSE) ps

migrate: ## 手动跑数据库迁移
	$(COMPOSE) exec api alembic upgrade head

seed: ## 手动播种初始 admin + 默认 Profile
	$(COMPOSE) exec api python -m app.scripts.init_db

verify: check-env ## 跑 verify_compose.sh 完整健康检查
	./scripts/verify_compose.sh

psql: ## 进入 postgres 交互式 SQL
	$(COMPOSE) exec postgres psql -U $${POSTGRES_USER:-wikforge} -d $${POSTGRES_DB:-wikforge}

shell-api: ## 进入 api 容器 shell
	$(COMPOSE) exec api bash

shell-worker: ## 进入 worker 容器 shell
	$(COMPOSE) exec worker bash

secrets: ## 生成一组强随机密钥, 复制到 .env
	@echo "# 复制下面的值替换 .env 中对应字段:"
	@echo ""
	@echo "POSTGRES_PASSWORD=$$(openssl rand -base64 24 | tr -d '/+=' | head -c 32)"
	@echo "MINIO_SECRET_KEY=$$(openssl rand -base64 24 | tr -d '/+=' | head -c 32)"
	@echo "JWT_SECRET_KEY=$$(openssl rand -hex 32)"
	@echo "LITELLM_MASTER_KEY=sk-$$(openssl rand -hex 24)"
	@echo "LITELLM_SALT_KEY=sk-salt-$$(openssl rand -hex 16)"
	@echo "LITELLM_UI_PASSWORD=$$(openssl rand -base64 18 | tr -d '/+=' | head -c 20)"
	@echo "OPENSEARCH_PASSWORD=ChangeMe-In-Prod-$$(openssl rand -hex 4)!"

first-run: check-env ## 新机器首次部署: 启动 + 等待全部 healthy + 提示访问
	@echo "[1/3] 启动所有服务 (首次需拉镜像并构建,5-15 分钟)..."
	$(COMPOSE) up -d
	@echo "[2/3] 等待全部服务 healthy..."
	@for i in $$(seq 1 60); do \
		unhealthy=$$($(COMPOSE) ps --format json 2>/dev/null | python3 -c "import sys,json; n=0; [n:=n+1 for l in sys.stdin if l.strip() and json.loads(l).get('Health','') not in ('healthy','')]; print(n)" 2>/dev/null || echo 9); \
		if [ "$$unhealthy" = "0" ]; then \
			echo "[OK] 全部服务就绪"; \
			break; \
		fi; \
		echo "  $$i: 还有服务未 healthy, 继续等待..."; \
		sleep 10; \
	done
	@$(COMPOSE) ps
	@echo ""
	@echo "[3/3] 完成。"
	@echo ""
	@echo "管理员账号 (首次播种): $${INITIAL_ADMIN_EMAIL:-admin@wikforge.com} / $${INITIAL_ADMIN_PASSWORD:-Admin@123}"
	@echo "请访问 http://localhost:$${FRONTEND_PORT:-80}/login 登录后立即修改密码。"


# =============================================================================
# 维护类命令
# =============================================================================

smoke: ## 跑端到端冒烟 (登录 → 上传 → 处理 → 搜索 → RAG)
	./scripts/smoke-test.sh

prune-images: ## 清理 docker 悬挂镜像 / 构建缓存 (不动 wikforge volume)
	@echo "回收前:"
	@docker system df
	docker system prune -af
	@echo ""
	@echo "回收后:"
	@docker system df

clean: prune-images ## 同 prune-images, 别名

clean-data: ## 清空 wikforge 业务数据 (DB / OpenSearch / Qdrant / MinIO 文档), 保留 admin 账号
	@read -p "确认清空所有业务数据? 会删除文档/向量/索引,但保留 admin/Profile/词典 (yes/no): " ans && [ "$$ans" = "yes" ] || (echo "已取消"; exit 1)
	@echo "[1/4] 清 PostgreSQL documents..."
	$(COMPOSE) exec -T postgres psql -U $${POSTGRES_USER:-wikforge} -d $${POSTGRES_DB:-wikforge} -c "DELETE FROM documents;" || true
	@echo "[2/4] 清 OpenSearch chunks..."
	@curl -sS -X POST "http://localhost:$${OPENSEARCH_PORT:-9200}/chunks/_delete_by_query" -H "Content-Type: application/json" -d '{"query":{"match_all":{}}}' > /dev/null || true
	@echo "[3/4] 清 Qdrant document_chunks..."
	@curl -sS -X DELETE "http://localhost:$${QDRANT_PORT:-6333}/collections/document_chunks/points" -H "Content-Type: application/json" -d '{"filter":{"must_not":[]}}' > /dev/null || true
	@echo "[4/4] 清 MinIO documents bucket..."
	@$(COMPOSE) exec -T minio sh -c 'mc alias set local http://localhost:9000 "$${MINIO_ROOT_USER:-minioadmin}" "$${MINIO_ROOT_PASSWORD:-minioadmin}" >/dev/null 2>&1; mc rm --recursive --force --quiet local/$${MINIO_BUCKET:-wikforge-documents}/ 2>/dev/null' || true
	@echo "✓ 业务数据已清空 (admin/Profile/词典 保留)"
