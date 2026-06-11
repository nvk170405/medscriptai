.PHONY: install lint test train serve docker-build docker-up docker-down clean

# ── Setup ────────────────────────────────────────────────────────────────────
install:
	pip install -e ".[dev]"
	pre-commit install

install-prod:
	pip install -e .

# ── Quality ──────────────────────────────────────────────────────────────────
lint:
	ruff check src/ api/ tests/ scripts/
	ruff format --check src/ api/ tests/ scripts/

format:
	ruff check --fix src/ api/ tests/ scripts/
	ruff format src/ api/ tests/ scripts/

typecheck:
	mypy src/ api/

# ── Tests ────────────────────────────────────────────────────────────────────
test:
	pytest tests/unit/ -v --cov=src/medscript --cov-report=term-missing

test-integration:
	pytest tests/integration/ -v

test-all:
	pytest tests/ -v --cov=src/medscript --cov-report=html

# ── Data ─────────────────────────────────────────────────────────────────────
download-data:
	python scripts/download_datasets.py

generate-synthetic:
	python scripts/generate_synthetic.py --count 20000 --output data/synthetic/

prepare-data:
	python scripts/prepare_data.py

# ── Training ─────────────────────────────────────────────────────────────────
train:
	python scripts/train.py --config configs/training_config.yaml

evaluate:
	python scripts/evaluate.py --checkpoint checkpoints/best.ckpt

# ── API ──────────────────────────────────────────────────────────────────────
serve:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

serve-prod:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

# ── Frontend ─────────────────────────────────────────────────────────────────
web-install:
	cd web && npm install

web-dev:
	cd web && npm run dev

web-build:
	cd web && npm run build

# ── Docker ───────────────────────────────────────────────────────────────────
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# ── Cleanup ──────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage dist/ build/ *.egg-info/
