"""Container health probe: GET /healthz/ from the local server.

The request carries the first allowed host name, since Django answers 400 to any other.
"""

import http.client
import os
import sys
import urllib.request

hosts = [h.strip() for h in os.environ.get("ALLOWED_HOSTS", "").split(",") if h.strip() and h.strip() != "*"]
host = hosts[0].lstrip(".") if hosts else "localhost"
request = urllib.request.Request("http://127.0.0.1:8000/healthz/", headers={"Host": host})
try:
    with urllib.request.urlopen(request, timeout=8) as response:
        sys.exit(0 if response.status == 200 else 1)
except (OSError, http.client.HTTPException) as exc:
    print(f"health check failed: {exc}", file=sys.stderr)
    sys.exit(1)
