"""
Export one cluster's history for the research (training data, plots, tables).

    python -m src.persistence.export --cluster tt-pc --out results/export --hours 72            # one CSV per table
    python -m src.persistence.export --cluster tt-pc --table actions --format jsonl --out -     # to stdout

Always one cluster at a time: histories of different clusters are never merged into a file.
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timedelta, timezone

from src import config
from src.persistence import db, queries

TABLES = {"steps": queries.steps, "metrics": queries.metrics, "actions": queries.actions, "violations": queries.violations,
          "critical_paths": queries.critical_paths, "topology": queries.topology, "events": queries.events}


def _cell(v):
    return json.dumps(v, default=str) if isinstance(v, (dict, list)) else v


def _write(rows, fmt, fh):
    if fmt == "jsonl":
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
        return
    if not rows:
        return
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    for r in rows:
        w.writerow({k: _cell(v) for k, v in r.items()})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cluster", required=True, help="cluster name or id")
    ap.add_argument("--table", choices=list(TABLES), default=None, help="one table (default: all)")
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--format", choices=["csv", "jsonl"], default="csv")
    ap.add_argument("--out", default="export", help="directory, or - for stdout (needs --table)")
    a = ap.parse_args(argv)
    if not config.DATABASE_URL:
        ap.error("set AUTOSCALER_DATABASE_URL")
    if a.out == "-" and not a.table:
        ap.error("--out - needs --table")
    until = datetime.now(timezone.utc)
    since = until - timedelta(hours=a.hours)
    with db.connect(config.DATABASE_URL, autocommit=True) as conn:
        cid = queries.resolve_cluster(conn, a.cluster)
        if cid is None:
            ap.error(f"no such cluster {a.cluster!r}")
        for name in ([a.table] if a.table else TABLES):
            rows = TABLES[name](conn, cid, since, until, limit=queries.MAX_LIMIT)
            if a.out == "-":
                _write(rows, a.format, sys.stdout)
                continue
            os.makedirs(a.out, exist_ok=True)
            path = os.path.join(a.out, f"{name}.{a.format}")
            with open(path, "w", newline="") as fh:
                _write(rows, a.format, fh)
            print(f"{path}: {len(rows)} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
