"""Frames stored in the database (AUTOSCALER_STORE_FRAMES=1) as a list of payloads, for ReplayEngine."""

from datetime import datetime, timedelta, timezone

from src import config
from src.persistence import db, queries


def load_frames(cluster, hours=24.0):
    if not config.DATABASE_URL:
        raise SystemExit("--replay-db needs AUTOSCALER_DATABASE_URL")
    with db.connect(config.DATABASE_URL, autocommit=True) as conn:
        cid = queries.resolve_cluster(conn, cluster)
        if cid is None:
            raise SystemExit(f"no such cluster {cluster!r}; known: {', '.join(c['name'] for c in queries.clusters(conn)) or 'none'}")
        until = datetime.now(timezone.utc)
        frames = queries.frames(conn, cid, until - timedelta(hours=hours), until)
    if not frames:
        raise SystemExit(f"no stored frames for {cluster!r} in the last {hours:g} h (they are kept only with AUTOSCALER_STORE_FRAMES=1)")
    return frames
