-- ClickHouse Cloud schema for the Station Director.
-- Run against a ClickHouse Cloud service once the account exists (Day 1).

CREATE TABLE IF NOT EXISTS events (
    ts DateTime64(3) DEFAULT now64(3),
    channel String,
    event_type String,      -- 'channel_dwell' | 'session_end'
    dwell_seconds Float64,
    source String DEFAULT 'live'  -- 'live' | 'synthetic' (seeded history)
) ENGINE = MergeTree
ORDER BY ts;

CREATE TABLE IF NOT EXISTS directives (
    ts DateTime64(3) DEFAULT now64(3),
    channel String,
    action String,           -- 'reorder' | 'interstitial' | 'commission_ingest'
    payload String,          -- JSON, schema-validated at the service layer
    reasoning String         -- the agent's stated rationale, for the demo console
) ENGINE = MergeTree
ORDER BY ts;
