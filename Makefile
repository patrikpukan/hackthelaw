# Legal RAG Agent Makefile

.PHONY: help build up down test test-local clean logs restart

# Variables
COMPOSE_FILE = docker-compose.yml
TEST_COMPOSE_FILE = docker-compose.test.yml
PROJECT_NAME = legalrag

help: ## Show this help message
	@echo "Legal RAG Agent - Development Commands"
	@echo "======================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

build: ## Build all services
	docker-compose -f $(COMPOSE_FILE) build

up: ## Start all services
	docker-compose -f $(COMPOSE_FILE) up -d

down: ## Stop all services
	docker-compose -f $(COMPOSE_FILE) down

logs: ## View logs from all services
	docker-compose -f $(COMPOSE_FILE) logs -f

restart: down up ## Restart all services

status: ## Show status of all services
	docker-compose -f $(COMPOSE_FILE) ps

# Testing commands
test: ## Run all tests in Docker containers
	@echo "🧪 Running tests in Docker environment..."
	docker-compose -f $(TEST_COMPOSE_FILE) down --volumes
	docker-compose -f $(TEST_COMPOSE_FILE) build test-runner
	docker-compose -f $(TEST_COMPOSE_FILE) up --abort-on-container-exit test-runner
	@echo "📊 Test results available in ./test-results/"

test-unit: ## Run only unit tests
	@echo "🧪 Running unit tests..."
	docker-compose -f $(TEST_COMPOSE_FILE) run --rm test-runner pytest -m unit -v

test-integration: ## Run only integration tests  
	@echo "🧪 Running integration tests..."
	docker-compose -f $(TEST_COMPOSE_FILE) run --rm test-runner pytest -m integration -v

test-api: ## Run only API tests
	@echo "🧪 Running API tests..."
	docker-compose -f $(TEST_COMPOSE_FILE) run --rm test-runner pytest -m api -v

test-local: ## Run tests locally (requires local Python environment)
	@echo "🧪 Running tests locally..."
	cd app && pytest --cov=. --cov-report=html:../htmlcov --cov-report=term-missing -v

test-coverage: ## Generate test coverage report
	@echo "📊 Generating coverage report..."
	docker-compose -f $(TEST_COMPOSE_FILE) run --rm test-runner pytest --cov=app --cov-report=html:/app/test-results/htmlcov --cov-report=term-missing
	@echo "Coverage report available at ./test-results/htmlcov/index.html"

test-clean: ## Clean test environment
	docker-compose -f $(TEST_COMPOSE_FILE) down --volumes --remove-orphans
	docker system prune -f

# Development commands
dev-setup: ## Set up development environment
	@echo "🚀 Setting up development environment..."
	cp .env.example .env || touch .env
	docker-compose -f $(COMPOSE_FILE) build
	docker-compose -f $(COMPOSE_FILE) up -d postgres redis
	@echo "⏳ Waiting for services to be ready..."
	sleep 10
	@echo "✅ Development environment ready!"

dev-shell: ## Access development shell in API container
	docker-compose -f $(COMPOSE_FILE) exec api bash

dev-db-shell: ## Access PostgreSQL shell
	docker-compose -f $(COMPOSE_FILE) exec postgres psql -U user -d legalrag

dev-redis-shell: ## Access Redis CLI
	docker-compose -f $(COMPOSE_FILE) exec redis redis-cli

# Database commands
db-migrate: ## Run database migrations
	docker-compose -f $(COMPOSE_FILE) exec api alembic upgrade head

db-reset: ## Reset database (WARNING: This will delete all data!)
	@echo "⚠️  This will delete all data! Press Ctrl+C to cancel..."
	@sleep 5
	docker-compose -f $(COMPOSE_FILE) down postgres
	docker volume rm $(PROJECT_NAME)_postgres_data || true
	docker-compose -f $(COMPOSE_FILE) up -d postgres
	@echo "✅ Database reset complete"

# Monitoring commands
monitor-logs: ## Monitor application logs
	docker-compose -f $(COMPOSE_FILE) logs -f api worker

monitor-flower: ## Open Celery Flower monitoring (http://localhost:5555)
	@echo "🌸 Opening Celery Flower at http://localhost:5555"
	@command -v open >/dev/null 2>&1 && open http://localhost:5555 || echo "Please open http://localhost:5555 in your browser"

# Cleanup commands
clean: ## Clean up Docker resources
	docker-compose -f $(COMPOSE_FILE) down --volumes --remove-orphans
	docker-compose -f $(TEST_COMPOSE_FILE) down --volumes --remove-orphans
	docker system prune -f

clean-all: ## Clean up everything including images
	docker-compose -f $(COMPOSE_FILE) down --volumes --remove-orphans --rmi all
	docker-compose -f $(TEST_COMPOSE_FILE) down --volumes --remove-orphans --rmi all
	docker system prune -af

# Production commands  
prod-build: ## Build for production
	docker-compose -f docker-compose.prod.yml build

prod-deploy: ## Deploy to production
	docker-compose -f docker-compose.prod.yml up -d

# Quality commands
lint: ## Run code linting
	docker-compose -f $(COMPOSE_FILE) exec api flake8 .
	docker-compose -f $(COMPOSE_FILE) exec api black --check .

format: ## Format code
	docker-compose -f $(COMPOSE_FILE) exec api black .

# Documentation
docs-serve: ## Serve API documentation
	@echo "📚 API documentation available at:"
	@echo "  - Swagger UI: http://localhost:8000/docs"
	@echo "  - ReDoc: http://localhost:8000/redoc"

# Quick start
quickstart: dev-setup test ## Quick start: setup environment and run tests
	@echo "🎉 Quick start complete!"
	@echo "🌐 API available at: http://localhost:8000"
	@echo "📚 Documentation at: http://localhost:8000/docs"
	@echo "🌸 Celery Flower at: http://localhost:5555"

# Health checks
health: ## Check health of all services
	@echo "🏥 Checking service health..."
	@curl -s http://localhost:8000/health || echo "❌ API service not responding"
	@curl -s http://localhost:5555 > /dev/null && echo "✅ Flower service healthy" || echo "❌ Flower service not responding"
	@docker-compose -f $(COMPOSE_FILE) exec postgres pg_isready -U user -d legalrag > /dev/null && echo "✅ PostgreSQL healthy" || echo "❌ PostgreSQL not responding"
	@docker-compose -f $(COMPOSE_FILE) exec redis redis-cli ping > /dev/null && echo "✅ Redis healthy" || echo "❌ Redis not responding" 