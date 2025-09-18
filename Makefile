# Makefile for AI Content Creator
# Provides convenient shortcuts for common development and deployment tasks

.PHONY: help install dev clean test lint format docker-build docker-dev docker-prod

# Default target
help: ## Show this help message
	@echo "AI Content Creator - Available Commands:"
	@echo "========================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Development
install: ## Install dependencies
	pip install -r requirements.txt

dev: ## Install in development mode
	pip install -e .

clean: ## Clean up Python cache files and build artifacts
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf .tox

# Testing
test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Code Quality
format: ## Format code with black and isort
	black src/ tests/ examples/
	isort src/ tests/ examples/

format-check: ## Check code formatting
	black --check src/ tests/ examples/
	isort --check-only src/ tests/ examples/

lint: ## Lint code with flake8 and mypy
	flake8 src/ tests/ examples/
	mypy src/

# API Server
run-api: ## Run API server with hot reload
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-worker: ## Run background worker
	python -m celery worker -A src.worker.celery_app --loglevel=info

# Docker Commands
docker-build: ## Build Docker images
	docker-compose build

docker-dev: ## Start development environment
	@echo "Starting development environment..."
	@./scripts/deploy.sh dev

docker-prod: ## Deploy production environment
	@echo "Deploying production environment..."
	@./scripts/deploy.sh deploy

docker-stop: ## Stop all Docker services
	docker-compose down

docker-clean: ## Clean up Docker containers and images
	@./scripts/deploy.sh cleanup

docker-logs: ## Show Docker logs
	docker-compose logs -f

# GPU Setup
gpu-setup: ## Setup GPU support for Docker
	@./scripts/setup-gpu.sh setup

gpu-test: ## Test GPU in Docker
	@./scripts/setup-gpu.sh test

gpu-monitor: ## Monitor GPU usage
	@./scripts/setup-gpu.sh monitor

# Database
db-init: ## Initialize database
	docker-compose exec postgres psql -U ai_user -d ai_content_db -f /docker-entrypoint-initdb.d/init.sql

db-backup: ## Backup database
	@./scripts/deploy.sh backup

db-restore: ## Restore database (usage: make db-restore BACKUP=backup_name)
	@./scripts/deploy.sh restore $(BACKUP)

# Monitoring
monitor: ## Monitor all services
	@./scripts/deploy.sh monitor

status: ## Show service status
	@./scripts/deploy.sh status

health: ## Run health checks
	@./scripts/deploy.sh health

# Examples
run-example: ## Run example workflow (usage: make run-example FILE=input.mp4)
	python examples/complete_workflow.py $(FILE)

# Documentation
docs-serve: ## Serve documentation locally
	@echo "Starting documentation server..."
	@echo "Main README: http://localhost:8080/README.md"
	@echo "API Docs: http://localhost:8080/docs/API.md"
	@echo "Docker Guide: http://localhost:8080/docs/DOCKER.md"
	@echo "Usage Guide: http://localhost:8080/docs/USAGE_GUIDE.md"
	@python -m http.server 8080

# Quick Setup Commands
setup-dev: install dev ## Quick development setup
	@echo "Development environment ready!"
	@echo "Run 'make run-api' to start the API server"

setup-docker: docker-build docker-dev ## Quick Docker development setup
	@echo "Docker development environment ready!"
	@echo "API: http://localhost:8000"
	@echo "Jupyter: http://localhost:8888"

setup-prod: docker-build docker-prod ## Quick production setup
	@echo "Production environment ready!"
	@echo "Web Interface: http://localhost"
	@echo "API Docs: http://localhost/docs"
	@echo "Monitoring: http://localhost:3000"

# Utility Commands
models-download: ## Pre-download AI models
	@echo "Downloading AI models..."
	python -c "import whisper; whisper.load_model('tiny'); whisper.load_model('base')"
	python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); YOLO('yolov8s.pt')"
	@echo "Models downloaded successfully"

check-deps: ## Check if all dependencies are available
	@echo "Checking dependencies..."
	@python -c "import torch; print(f'PyTorch: {torch.__version__}')"
	@python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
	@python -c "import whisper; print('Whisper: Available')"
	@python -c "from ultralytics import YOLO; print('YOLO: Available')"
	@echo "All dependencies available!"

benchmark: ## Run performance benchmarks
	@echo "Running performance benchmarks..."
	python -c "
import time
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    # Simple GPU benchmark
    x = torch.randn(1000, 1000).cuda()
    start = time.time()
    for _ in range(100):
        y = torch.mm(x, x)
    torch.cuda.synchronize()
    end = time.time()
    print(f'GPU Performance: {(end-start)*1000:.2f}ms for 100 matrix multiplications')
"

# Environment Info
info: ## Show environment information
	@echo "AI Content Creator - Environment Information"
	@echo "==========================================="
	@echo "Python: $$(python --version)"
	@echo "Pip: $$(pip --version)"
	@echo "Docker: $$(docker --version 2>/dev/null || echo 'Not installed')"
	@echo "Docker Compose: $$(docker-compose --version 2>/dev/null || echo 'Not installed')"
	@echo "NVIDIA-SMI: $$(nvidia-smi --version 2>/dev/null | head -1 || echo 'Not available')"
	@echo "Current Directory: $$(pwd)"
	@echo "Git Branch: $$(git branch --show-current 2>/dev/null || echo 'Not a git repository')"
	@echo "Disk Space: $$(df -h . | tail -1 | awk '{print $$4}') available"

# All-in-one commands
all-tests: format-check lint test ## Run all code quality checks and tests

all-setup: check-deps models-download setup-docker ## Complete setup (dependencies + models + Docker)

# Aliases for convenience
start: docker-dev ## Alias for docker-dev
stop: docker-stop ## Alias for docker-stop
build: docker-build ## Alias for docker-build
deploy: docker-prod ## Alias for docker-prod
logs: docker-logs ## Alias for docker-logs