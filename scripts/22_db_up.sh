#!/bin/bash
# Start the PostgreSQL that stores the agent's history, wait until it accepts connections, print the URL to use.
#
#   bash scripts/22_db_up.sh
#   export AUTOSCALER_DATABASE_URL=...     (printed below)  or pass --database-url to scripts/19 / scripts/20
set -e
cd "$(dirname "$0")/.."
docker compose -f docker-compose.db.yml up -d --wait
PORT="${AUTOSCALER_DB_PORT:-5432}"; PASS="${AUTOSCALER_DB_PASSWORD:-autoscaler-local}"
echo
echo "PostgreSQL is up. Use:"
echo "  export AUTOSCALER_DATABASE_URL=postgresql://autoscaler:${PASS}@127.0.0.1:${PORT}/autoscaler"
echo "Look inside:  docker exec -it autoscaler-postgres psql -U autoscaler -d autoscaler"
