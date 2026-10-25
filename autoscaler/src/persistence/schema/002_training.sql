-- What the model was trained on, so it can be shown and audited: every training run with its settings and learning curve, a summary of
-- every simulated episode, full step-level trajectories of a sample of episodes, and the real measurements the simulator is built from.
-- Kept for ever (not subject to retention). source_cluster_id links a run to the cluster whose measurements the regime came from.

CREATE TABLE training_inputs (
    id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_cluster_id uuid REFERENCES clusters (id) ON DELETE SET NULL,
    kind              text NOT NULL,            -- traces | calibration | sim_params | topology | profile | regime
    name              text NOT NULL,
    sha256            text NOT NULL,
    created_at        timestamptz NOT NULL DEFAULT now(),
    payload           jsonb NOT NULL,
    UNIQUE (kind, sha256)
);

CREATE TABLE training_runs (
    id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_cluster_id uuid REFERENCES clusters (id) ON DELETE SET NULL,
    started_at        timestamptz NOT NULL DEFAULT now(),
    ended_at          timestamptz,
    status            text NOT NULL DEFAULT 'running',     -- running | finished | failed
    arm               text NOT NULL,
    init_from         text,
    seed              integer NOT NULL,
    episodes          integer NOT NULL,
    hparams           jsonb NOT NULL DEFAULT '{}',
    regime            jsonb NOT NULL DEFAULT '{}',          -- the environment settings the simulator ran with (step length, pod start-up, bounds ...)
    topology_options  jsonb NOT NULL DEFAULT '{}',
    input_ids         uuid[] NOT NULL DEFAULT '{}',         -- training_inputs this run used
    git_sha           text,
    checkpoint_path   text,
    checkpoint_sha256 text,
    best_val          double precision,
    final_val         double precision,
    note              text
);
CREATE INDEX training_runs_started ON training_runs (started_at DESC);

CREATE TABLE training_curve (
    run_id         uuid NOT NULL REFERENCES training_runs (id) ON DELETE CASCADE,
    episode        integer NOT NULL,
    val_score      double precision,
    val_p95_ms     double precision,
    val_cost       double precision,
    val_violation  double precision,
    seconds        double precision,
    losses         jsonb NOT NULL DEFAULT '{}',
    PRIMARY KEY (run_id, episode)
);

-- one row per simulated episode
CREATE TABLE training_episodes (
    run_id           uuid NOT NULL REFERENCES training_runs (id) ON DELETE CASCADE,
    episode          integer NOT NULL,
    topology         text NOT NULL,
    n_services       integer NOT NULL,
    pattern_rps_peak double precision,
    preference       double precision[] NOT NULL,           -- [slo, cost, balance] the agent was steered with
    steps            integer NOT NULL,
    mean_p95_ms      double precision,
    max_p95_ms       double precision,
    slo_ms           double precision,
    violation_steps  integer,
    mean_cost        double precision,
    mean_replicas    double precision,
    scale_actions    integer,                               -- executed replica changes
    mean_penalty     double precision[],                    -- [slo, cost, balance]
    PRIMARY KEY (run_id, episode)
);

-- full step-level record of a sample of the episodes: what the agent saw, did and was scored, per step and per service
CREATE TABLE training_trajectories (
    run_id     uuid NOT NULL REFERENCES training_runs (id) ON DELETE CASCADE,
    episode    integer NOT NULL,
    topology   text NOT NULL,
    steps      jsonb NOT NULL,                              -- [{step, rps, p95_ms, cost, penalty:[3], services:{name:{replicas,cpu,mem,util,p95_ms,rps,delta}}}]
    PRIMARY KEY (run_id, episode)
);
