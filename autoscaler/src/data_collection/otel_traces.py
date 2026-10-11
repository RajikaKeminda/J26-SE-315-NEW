"""
otel_traces.py

Per-service request statistics from standard OpenTelemetry spans (as stored by Jaeger), for
applications that do NOT export Prometheus request metrics - Train-Ticket's Java services only
emit traces. Nothing here is specific to one application: it relies on the OTel conventions

    span.kind = server    a service handling one request (its duration = INCLUSIVE latency)
    span.kind = client    an outbound call (HTTP to another service, or a database query)
    CHILD_OF references   parent/child, across process (service) boundaries via propagation

For every `server` span S of service X:

    inclusive latency   duration(S)
    own latency         duration(S) minus the time X spent WAITING for other services, i.e. the
                        union of the intervals of its outbound calls to other services (parallel
                        calls are counted once). Database time stays in X's own latency: it is
                        part of what the service's callers experience from X.

The request rate of a service is its server-span count over the window, corrected for the
trace sampling ratio. When the query hit the trace limit (so the window was truncated) the rate
is computed over the span of time the returned traces actually cover.
"""

from collections import defaultdict


def _tag(span, key, default=None):
    for t in span.get("tags", ()):
        if t.get("key") == key:
            return t.get("value")
    return default


def _is_error(span):
    code = _tag(span, "http.status_code")
    if code is not None:
        try:
            return int(code) >= 500
        except (TypeError, ValueError):
            pass
    err = _tag(span, "error")
    return err is True or str(err).lower() == "true"


def _pct(sorted_vals, q):
    if not sorted_vals:
        return 0.0
    return sorted_vals[min(len(sorted_vals) - 1, int(q * (len(sorted_vals) - 1) + 0.5))]


def _union_length(intervals):
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def analyse_trace(trace):
    """-> list of {service, parent_service, start_us, incl_ms, own_ms, error} for each server span,
    where parent_service is the calling service (None for the root)."""
    processes = trace.get("processes", {})
    spans = trace.get("spans", [])
    by_id = {s["spanID"]: s for s in spans}
    svc = {s["spanID"]: processes.get(s.get("processID"), {}).get("serviceName") for s in spans}
    children = defaultdict(list)
    parent = {}
    for s in spans:
        for r in s.get("references", ()):
            if r.get("refType") == "CHILD_OF" and r.get("spanID") in by_id:
                children[r["spanID"]].append(s["spanID"])
                parent[s["spanID"]] = r["spanID"]
                break

    def waits(span_id, owner, out):
        """Collect the intervals during which `owner` waits on OTHER services below `span_id`."""
        for c in children.get(span_id, ()):
            if svc[c] != owner:
                p = parent.get(c)
                # the caller's own client span (it wraps the callee plus the network hop)
                ref = by_id[p] if p and svc.get(p) == owner and _tag(by_id[p], "span.kind") == "client" else by_id[c]
                out.append((ref["startTime"], ref["startTime"] + ref["duration"]))
            else:
                waits(c, owner, out)

    rows = []
    for s in spans:
        if _tag(s, "span.kind") != "server" or not svc[s["spanID"]]:
            continue
        sid, owner = s["spanID"], svc[s["spanID"]]
        lo, hi = s["startTime"], s["startTime"] + s["duration"]
        iv = []
        waits(sid, owner, iv)
        waited = _union_length([(max(a, lo), min(b, hi)) for a, b in iv if b > lo and a < hi])
        incl = s["duration"] / 1000.0
        # the caller of this server span = service owning the nearest ancestor from another service
        caller, p = None, parent.get(sid)
        while p is not None:
            if svc[p] != owner:
                caller = svc[p]
                break
            p = parent.get(p)
        rows.append({"service": owner, "parent_service": caller, "start_us": s["startTime"],
                     "incl_ms": incl, "own_ms": max(0.0, incl - waited / 1000.0), "error": _is_error(s)})
    return rows


def window_stats(traces, limit=None, window_s=None, sample_rate=1.0):
    """Aggregate a window of traces into per-service stats keyed like prometheus_client.collect():

        {service: {requests, error_rate, own_latency_ms, own_p95_ms, incl_latency_ms,
                   incl_p50_ms, incl_p95_ms, incl_p99_ms, samples}}

    `limit`: the limit used in the Jaeger query (a result of exactly `limit` traces is truncated).
    `window_s`: the requested window; replaced by the time the returned traces cover if truncated."""
    per = defaultdict(list)
    starts = []
    for t in traces:
        rows = analyse_trace(t)
        root_starts = [r["start_us"] for r in rows if r["parent_service"] is None]
        if root_starts:
            starts.append(min(root_starts))
        for r in rows:
            per[r["service"]].append(r)

    covered = window_s or 0.0
    if limit and len(traces) >= limit and len(starts) > 1:          # truncated: shrink the window
        covered = max((max(starts) - min(starts)) / 1e6, 1e-3)
    covered = max(covered, 1e-3)
    rate_scale = 1.0 / (covered * max(sample_rate, 1e-6))

    out = {}
    for service, rows in per.items():
        incl = sorted(r["incl_ms"] for r in rows)
        own = sorted(r["own_ms"] for r in rows)
        n = len(rows)
        out[service] = {
            "requests": n * rate_scale,
            "error_rate": sum(1 for r in rows if r["error"]) / n,
            "own_latency_ms": sum(own) / n,
            "own_p95_ms": _pct(own, 0.95),
            "incl_latency_ms": sum(incl) / n,
            "incl_p50_ms": _pct(incl, 0.50),
            "incl_p95_ms": _pct(incl, 0.95),
            "incl_p99_ms": _pct(incl, 0.99),
            "samples": n,
        }
    return out


# ---------------------------------------------------------------------------------------------------
# Ground truth for the critical path: who was the request actually waiting for?
# ---------------------------------------------------------------------------------------------------
def critical_time_by_service(trace):
    """Attributes the end-to-end time of ONE request to services, walking the span tree backwards from the
    end of the request (the standard critical-path analysis of a trace):

      * at a service's server span, the last-finishing call to another service is blamed for the time it
        was running; the time before it started is blamed on whatever ran before it, recursively;
      * calls that overlap an already-blamed later call (parallel calls) are not on the critical path;
      * time not covered by any downstream call (own processing, database, network hops) is the service's own.

    Returns ({service: milliseconds}, total_ms); the values sum to the duration of the root server span.
    This is observed behaviour of the request, independent of the agent's graph - it is what the agent's
    critical path is evaluated against (evaluation/critical_path_eval.py)."""
    processes = trace.get("processes", {})
    spans = trace.get("spans", [])
    by_id = {s["spanID"]: s for s in spans}
    svc = {s["spanID"]: processes.get(s.get("processID"), {}).get("serviceName") for s in spans}
    children = defaultdict(list)
    has_parent = set()
    for s in spans:
        for r in s.get("references", ()):
            if r.get("refType") == "CHILD_OF" and r.get("spanID") in by_id:
                children[r["spanID"]].append(s["spanID"])
                has_parent.add(s["spanID"])
                break

    def callees(span_id, owner, out):
        for c in children.get(span_id, ()):
            if svc[c] != owner:
                if _tag(by_id[c], "span.kind") == "server":
                    out.append(c)
                else:
                    callees(c, owner, out)
            else:
                callees(c, owner, out)

    def blame(span_id, hi, acc):
        s = by_id[span_id]
        owner = svc[span_id]
        start, end = s["startTime"], min(s["startTime"] + s["duration"], hi)
        if end <= start:
            return
        cs = []
        callees(span_id, owner, cs)
        cs = sorted(((by_id[c]["startTime"], by_id[c]["startTime"] + by_id[c]["duration"], c) for c in cs),
                    key=lambda x: -x[1])
        cur = end
        for cstart, cend, cid in cs:
            if cstart >= cur or cend <= start:
                continue                                      # overlapped by a later call / outside the span
            eff_end = min(cend, cur)
            acc[owner] += max(0, cur - eff_end)               # time between this call's end and what came after
            blame(cid, eff_end, acc)
            cur = max(start, cstart)
        acc[owner] += max(0, cur - start)

    roots = [s["spanID"] for s in spans if _tag(s, "span.kind") == "server" and s["spanID"] not in has_parent
             or (_tag(s, "span.kind") == "server" and svc[s["spanID"]] and not _has_cross_parent(s, by_id, svc))]
    if not roots:
        return {}, 0.0
    root = max(roots, key=lambda r: by_id[r]["duration"])
    acc = defaultdict(float)
    blame(root, by_id[root]["startTime"] + by_id[root]["duration"], acc)
    total = by_id[root]["duration"] / 1000.0
    return {k: v / 1000.0 for k, v in acc.items() if k and v > 0}, total


def _has_cross_parent(span, by_id, svc):
    """True if some ancestor of `span` belongs to another service (so it is not a request root)."""
    me = svc[span["spanID"]]
    seen, cur = set(), span
    while cur is not None and cur["spanID"] not in seen:
        seen.add(cur["spanID"])
        parent = None
        for r in cur.get("references", ()):
            if r.get("refType") == "CHILD_OF" and r.get("spanID") in by_id:
                parent = by_id[r["spanID"]]
                break
        if parent is not None and svc[parent["spanID"]] != me:
            return True
        cur = parent
    return False
