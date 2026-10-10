#!/bin/sh
# Supervisor entrypoint for the single-service deployment (Render).
#
# Starts the FastAPI API on a loopback port with nginx in front on the
# platform's public port (Render injects PORT, default 10000). nginx serves the
# built React dashboard SPA as static files and proxies /health, /api/*, /docs
# and /redoc to the API -- see deploy/nginx.conf.template.
#
# The loop is the crash policy: if either process dies, the container exits so
# the platform restarts the whole stack, rather than serving a half-broken site
# (e.g. a dashboard whose /api/* calls 502).

set -u

# export: envsubst is a child process and only sees exported variables --
# without this, an unexported PORT renders as `listen ;` and nginx aborts.
PORT="${PORT:-10000}"
export PORT
API_HOST=127.0.0.1
API_PORT=8000

# Render the nginx config from the template with the platform port.
envsubst '${PORT}' < /app/deploy/nginx.conf.template > /etc/nginx/conf.d/default.conf

# The backend binds loopback only; the public surface is nginx alone.
python -m uvicorn conftest.api.main:app --host "$API_HOST" --port "$API_PORT" &
API_PID=$!

nginx -g 'daemon off;' &
NGINX_PID=$!

shutdown() {
    kill "$NGINX_PID" "$API_PID" 2>/dev/null || true
    wait "$NGINX_PID" "$API_PID" 2>/dev/null || true
    exit 0
}
trap shutdown TERM INT

while kill -0 "$API_PID" 2>/dev/null \
    && kill -0 "$NGINX_PID" 2>/dev/null; do
    sleep 2
done

echo "entrypoint: a backend process exited; shutting the stack down" >&2
shutdown
