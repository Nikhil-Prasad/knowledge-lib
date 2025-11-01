# Convenience targets for Docker Compose in infra/docker
SHELL := /bin/bash

COMPOSE_FILE := infra/docker/docker-compose.yml
# Pass root .env to Compose so we have a single source of truth
DC := docker compose --env-file .env -f $(COMPOSE_FILE)

.PHONY: dc-up dc-down up down

# Start services in detached mode
dc-up:
	$(DC) up -d

# Stop and remove services
dc-down:
	$(DC) down

# Aliases
up: dc-up
down: dc-down
