.PHONY: start logs stop

# ── start ──────────────────────────────────────────────────────────────────
# Build images and bring everything up.
# Rust runs once (pipeline), dashboard stays up at http://localhost:8501
start:
	docker compose up --build -d || true
	@echo ""
	@echo "  Rust pipeline:  docker compose logs rust"
	@echo "  Dashboard:      http://localhost:8501"
	@echo ""
	@docker compose ps

# ── logs ───────────────────────────────────────────────────────────────────
# Stream logs from all services (Ctrl-C to stop watching)
logs:
	docker compose logs -f

# ── stop ───────────────────────────────────────────────────────────────────
# Stop all containers, remove images, volumes, networks, and local caches
# Only touches cpcm-* resources — leaves other Docker workloads untouched
stop:
	@echo "Stopping containers..."
	docker compose down --timeout 10

	@echo "Removing project images..."
	docker rmi cpcm-rust cpcm-python 2>/dev/null || true

	@echo "Removing project volumes..."
	docker volume rm cpcm-rust-results cpcm-dashboard-cache 2>/dev/null || true

	@echo "Removing project network..."
	docker network rm cpcm_default 2>/dev/null || true

	@echo "Clearing local caches..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf causal_portfolio/data/cache/*.parquet 2>/dev/null || true
	rm -rf causal_model/cache/ 2>/dev/null || true

	@echo ""
	@echo "Clean."
