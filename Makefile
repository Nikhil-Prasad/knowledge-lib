# Convenience targets for Docker Compose in infra/docker
SHELL := /bin/bash

COMPOSE_FILE := infra/docker/docker-compose.yml
DC := docker compose -f $(COMPOSE_FILE)

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
