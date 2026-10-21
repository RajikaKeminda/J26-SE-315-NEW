-- History of what the autoscaler saw and did, one set of rows per monitored cluster.
-- Every fact table carries cluster_id and run_id; (cluster_id, run_id) is a composite foreign key to runs, so a row can
-- never belong to a run of a different cluster.

CREATE TABLE clusters (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name          text NOT NULL,
    namespace     text NOT NULL,
    kube_uid      text,                 -- UID of the kube-system namespace: a fingerprint of the cluster itself, when readable
    api_server    text,
    entry_service text,
    created_at    timestamptz NOT NULL DEFAULT now(),
    last_seen_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (name, namespace)
);
CREATE UNIQUE INDEX clusters_uid_ns ON clusters (kube_uid, namespace) WHERE kube_uid IS NOT NULL;

CREATE TABLE runs (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    cluster_id  uuid NOT NULL REFERENCES clusters (id) ON DELETE CASCADE,
    started_at  timestamptz NOT NULL DEFAULT now(),
    ended_at    timestamptz,
    mode        text NOT NULL,
    shadow      boolean NOT NULL,
    controller  text,
    policy      text,
    slo_ms      double precision,
    git_sha     text,
    settings    jsonb NOT NULL DEFAULT '{}',
    UNIQUE (cluster_id, id)
);
CREATE INDEX runs_cluster ON runs (cluster_id, started_at DESC);

-- one row per control step: the end-to-end view, SLO and breaker (kept long: it is small)
CREATE TABLE steps (
    cluster_id     uuid NOT NULL,
    run_id         uuid NOT NULL,
    step           integer NOT NULL,
    ts             timestamptz NOT NULL,
    rps            double precision,
    p50_ms         double precision,
    p95_ms         double precision,
    p99_ms         double precision,
    mean_ms        double precision,
    error_rate     double precision,
    slo_ms         double precision,
    slo_breached   boolean NOT NULL,
    breaker_open   boolean NOT NULL,
    total_replicas integer,
    cost           double precision,
    budget         double precision,
    stale          boolean NOT NULL,
    stale_reason   text,
    shadow         boolean NOT NULL,
    controller     text,
    PRIMARY KEY (run_id, step),
    FOREIGN KEY (cluster_id, run_id) REFERENCES runs (cluster_id, id) ON DELETE CASCADE
);
CREATE INDEX steps_cluster_ts ON steps (cluster_id, ts);

-- per service, per step (short retention)
CREATE TABLE service_metrics (
    cluster_id       uuid NOT NULL,
    run_id           uuid NOT NULL,
    step             integer NOT NULL,
    ts               timestamptz NOT NULL,
    service          text NOT NULL,
    replicas         integer,
    ready            integer,
    pending          integer,
    cpu_limit        double precision,
    mem_limit_mb     double precision,
    cpu_util         double precision,
    mem_util         double precision,
    latency_ms       double precision,
    p95_ms           double precision,
    rps              double precision,
    on_critical_path boolean,
    controlled       boolean,
    PRIMARY KEY (run_id, step, service),
    FOREIGN KEY (cluster_id, run_id) REFERENCES runs (cluster_id, id) ON DELETE CASCADE
);
CREATE INDEX service_metrics_cluster_ts ON service_metrics (cluster_id, ts);
CREATE INDEX service_metrics_service ON service_metrics (cluster_id, service, ts);

-- what the agent proposed and what happened to it (only steps where something was proposed or blocked)
CREATE TABLE actions (
    cluster_id      uuid NOT NULL,
    run_id          uuid NOT NULL,
    step            integer NOT NULL,
    ts              timestamptz NOT NULL,
    service         text NOT NULL,
    shadow          boolean NOT NULL,          -- true: the action was only decided, nothing was sent to the cluster
    outcome         text NOT NULL,             -- applied | modified | vetoed | k8s_error
    proposed_dr     integer,
    guard_dr        integer,                   -- after the per-service guard, before the cluster-level vetoes
    applied_dr      integer,
    proposed_cpu    double precision,
    applied_cpu     double precision,
    proposed_mem_mb double precision,
    applied_mem_mb  double precision,
    replicas_before integer,
    replicas_after  integer,
    reason          text,
    reason_codes    text[] NOT NULL DEFAULT '{}',
    cpu_util        double precision,
    p95_ms          double precision,
    PRIMARY KEY (run_id, step, service),
    FOREIGN KEY (cluster_id, run_id) REFERENCES runs (cluster_id, id) ON DELETE CASCADE
);
CREATE INDEX actions_cluster_ts ON actions (cluster_id, ts);
CREATE INDEX actions_service ON actions (cluster_id, service, ts);

CREATE TABLE critical_paths (
    cluster_id       uuid NOT NULL,
    run_id           uuid NOT NULL,
    step             integer NOT NULL,
    ts               timestamptz NOT NULL,
    source           text NOT NULL,            -- dfs | crisp
    chain            text[] NOT NULL,
    chain_latency_ms double precision,
    changed          boolean NOT NULL,         -- the chain differs from the previous stored one
    evidence         jsonb,                    -- coverage / top-1 / random baseline / per-service share of the window
    profile          jsonb,                    -- critical shares: overall, tail, per request type
    slowest_trace    text,
    PRIMARY KEY (run_id, step),
    FOREIGN KEY (cluster_id, run_id) REFERENCES runs (cluster_id, id) ON DELETE CASCADE
);
CREATE INDEX critical_paths_cluster_ts ON critical_paths (cluster_id, ts);

CREATE TABLE topology_snapshots (
    cluster_id uuid NOT NULL,
    run_id     uuid NOT NULL,
    step       integer NOT NULL,
    ts         timestamptz NOT NULL,
    edges_hash text NOT NULL,
    n_services integer NOT NULL,
    edges      jsonb NOT NULL,                 -- [{source, target, share}]
    PRIMARY KEY (run_id, edges_hash),
    FOREIGN KEY (cluster_id, run_id) REFERENCES runs (cluster_id, id) ON DELETE CASCADE
);
CREATE INDEX topology_cluster_ts ON topology_snapshots (cluster_id, ts);

-- one row per episode in which the end-to-end p95 stayed above the SLO
CREATE TABLE violations (
    cluster_id      uuid NOT NULL,
    run_id          uuid NOT NULL,
    start_step      integer NOT NULL,
    started_at      timestamptz NOT NULL,
    ended_at        timestamptz,               -- null while the episode is open
    end_step        integer,
    steps_breached  integer NOT NULL,
    peak_p95_ms     double precision NOT NULL,
    slo_ms          double precision NOT NULL,
    top_services    jsonb NOT NULL DEFAULT '[]',   -- [{service, share}] services most often on the critical path during the episode
    actions         integer NOT NULL DEFAULT 0,    -- scaling actions applied during the episode
    breaker_opened  boolean NOT NULL DEFAULT false,
    slowest_trace   text,
    PRIMARY KEY (run_id, start_step),
    FOREIGN KEY (cluster_id, run_id) REFERENCES runs (cluster_id, id) ON DELETE CASCADE
);
CREATE INDEX violations_cluster_ts ON violations (cluster_id, started_at);

CREATE TABLE events (
    id         bigserial PRIMARY KEY,
    cluster_id uuid NOT NULL,
    run_id     uuid NOT NULL,
    step       integer,
    ts         timestamptz NOT NULL,
    type       text NOT NULL,                  -- agent_start | agent_stop | breaker_open | breaker_closed | hold | recovered | k8s_error | persistence
    severity   text NOT NULL DEFAULT 'info',
    detail     jsonb NOT NULL DEFAULT '{}',
    FOREIGN KEY (cluster_id, run_id) REFERENCES runs (cluster_id, id) ON DELETE CASCADE
);
CREATE INDEX events_cluster_ts ON events (cluster_id, ts);

-- full payloads, only when AUTOSCALER_STORE_FRAMES=1 (replay from the database)
CREATE TABLE frames (
    cluster_id uuid NOT NULL,
    run_id     uuid NOT NULL,
    step       integer NOT NULL,
    ts         timestamptz NOT NULL,
    payload    jsonb NOT NULL,
    PRIMARY KEY (run_id, step),
    FOREIGN KEY (cluster_id, run_id) REFERENCES runs (cluster_id, id) ON DELETE CASCADE
);
CREATE INDEX frames_cluster_ts ON frames (cluster_id, ts);
