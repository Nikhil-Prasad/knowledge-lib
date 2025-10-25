-- runs automatically on first boot
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS unaccent;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

--optional: create an app role separate from the superuser
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'kl_app') THEN
        CREATE ROLE kl_app LOGIN PASSWORD 'klapp' NOSUPERUSER NOCREATEDB NOCREATEROLE;
        GRANT CONNECT ON DATABASE knowledge TO kl_app;
    END IF;
END
$$;
