#!/usr/bin/env bash
# Create multiple PostgreSQL databases on first startup
set -euo pipefail

function create_database() {
    local database=$1
    echo "Creating database: $database"
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
        SELECT 'CREATE DATABASE $database'
        WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$database')\gexec
EOSQL
}

create_database "airflow_db"
create_database "metabase_db"

echo "✓ All databases initialized"
