# Convenience targets for Docker Compose in infra/docker
SHELL := /bin/bash

COMPOSE_FILE := infra/docker/docker-compose.yml
# Pass root .env to Compose so we have a single source of truth
DC := docker compose --env-file .env -f $(COMPOSE_FILE)

.PHONY: dc-up dc-down up down api

# Start services in detached mode
dc-up:
	$(DC) up -d

# Stop and remove services
dc-down:
	$(DC) down

# Aliases
up: dc-up
down: dc-down

# Run the FastAPI server (uvicorn)
API_HOST ?= 127.0.0.1
API_PORT ?= 8000
api:
	uv run uvicorn src.app.main:app --host $(API_HOST) --port $(API_PORT) --reload
