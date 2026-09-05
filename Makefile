.PHONY: help install docker-up docker-down test clean

help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make docker-up    - Start all Docker services"
	@echo "  make docker-down  - Stop all Docker services"
	@echo "  make docker-build - Rebuild Docker images"
	@echo "  make test         - Run tests"
	@echo "  make clean        - Clean temporary files"

install:
	pip install -r requirements.txt

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-build:
	docker-compose build --no-cache

docker-logs:
	docker-compose logs -f

test:
	pytest tests/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache .coverage htmlcov