# 8. Database Schema

## Core Tables

```sql
-- Task family groupings
CREATE TABLE task_families (
    family_id TEXT PRIMARY KEY,
    family_name TEXT NOT NULL,
    description TEXT,
    common_patterns TEXT, -- JSON array of pattern descriptions
    avg_difficulty REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ARC tasks storage
CREATE TABLE arc_tasks (
    task_id TEXT PRIMARY KEY,
    task_source TEXT NOT NULL DEFAULT 'training',
    difficulty_level TEXT CHECK (difficulty_level IN ('easy', 'medium', 'hard', 'unknown')),
    train_examples TEXT NOT NULL, -- JSON array
    test_input TEXT NOT NULL, -- JSON 2D array
    test_output TEXT, -- JSON 2D array (null for test tasks)
    family_id TEXT,
    metadata_json TEXT, -- JSON object with extracted features
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (family_id) REFERENCES task_families(family_id)
);

-- Task submissions and results
CREATE TABLE task_submissions (
    submission_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    predicted_output TEXT NOT NULL, -- JSON 2D array
    strategy_used TEXT NOT NULL,
    confidence_score REAL NOT NULL,
    is_correct BOOLEAN,
    processing_time_ms INTEGER NOT NULL,
    resource_usage_json TEXT NOT NULL, -- JSON object
    metadata_json TEXT, -- Additional metadata
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (task_id) REFERENCES arc_tasks(task_id),
    INDEX idx_task_user (task_id, user_id),
    INDEX idx_submitted_at (submitted_at)
);

-- Strategy performance tracking
CREATE TABLE strategy_metrics (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_type TEXT NOT NULL,
    task_id TEXT NOT NULL,
    success BOOLEAN NOT NULL,
    execution_time_ms INTEGER NOT NULL,
    confidence_score REAL,
    resource_cost REAL,
    error_type TEXT,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (task_id) REFERENCES arc_tasks(task_id),
    INDEX idx_strategy_task (strategy_type, task_id)
);

-- LLM response cache
CREATE TABLE llm_cache (
    cache_id TEXT PRIMARY KEY,
    prompt_hash TEXT NOT NULL UNIQUE,
    model_name TEXT NOT NULL,
    temperature REAL NOT NULL,
    response_text TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP,
    access_count INTEGER DEFAULT 1,
    INDEX idx_prompt_model (prompt_hash, model_name)
);

-- Resource usage tracking
CREATE TABLE resource_usage (
    usage_id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    strategy_type TEXT NOT NULL,
    cpu_seconds REAL NOT NULL,
    memory_mb REAL NOT NULL,
    gpu_memory_mb REAL,
    api_calls_json TEXT NOT NULL, -- JSON object
    total_tokens INTEGER,
    estimated_cost REAL NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (task_id) REFERENCES arc_tasks(task_id),
    INDEX idx_task_timestamp (task_id, timestamp)
);

-- Experiment tracking
CREATE TABLE experiments (
    experiment_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    hypothesis TEXT,
    strategy_configs_json TEXT NOT NULL, -- JSON object
    task_selection_json TEXT, -- JSON criteria
    status TEXT DEFAULT 'created',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_by TEXT NOT NULL
);

-- Experiment results
CREATE TABLE experiment_results (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    submission_id TEXT NOT NULL,
    metrics_json TEXT NOT NULL, -- JSON object with custom metrics
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id),
    FOREIGN KEY (task_id) REFERENCES arc_tasks(task_id),
    FOREIGN KEY (submission_id) REFERENCES task_submissions(submission_id)
);

-- TTT adaptations storage
CREATE TABLE ttt_adaptations (
    adaptation_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    base_model_checkpoint TEXT NOT NULL,
    adapted_weights_path TEXT NOT NULL,
    training_examples_json TEXT NOT NULL,
    adaptation_metrics_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used TIMESTAMP,
    use_count INTEGER DEFAULT 1,
    FOREIGN KEY (task_id) REFERENCES arc_tasks(task_id)
);

-- System events and errors
CREATE TABLE system_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    event_source TEXT NOT NULL,
    severity TEXT CHECK (severity IN ('debug', 'info', 'warning', 'error', 'critical')),
    message TEXT NOT NULL,
    context_json TEXT, -- JSON object with additional context
    stack_trace TEXT,
    occurred_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    INDEX idx_severity_time (severity, occurred_at)
);

-- Feature vectors for similarity search
CREATE TABLE task_features (
    task_id TEXT PRIMARY KEY,
    grid_dimensions TEXT NOT NULL, -- JSON array [width, height]
    color_count INTEGER NOT NULL,
    object_count INTEGER,
    has_symmetry BOOLEAN,
    pattern_types TEXT, -- JSON array of detected patterns
    transformation_hints TEXT, -- JSON array of possible transformations
    embedding_vector BLOB, -- Serialized numpy array for similarity
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (task_id) REFERENCES arc_tasks(task_id)
);
```

## Database Triggers

```sql
-- Update timestamp trigger
CREATE TRIGGER update_arc_tasks_timestamp 
AFTER UPDATE ON arc_tasks
FOR EACH ROW
BEGIN
    UPDATE arc_tasks SET updated_at = CURRENT_TIMESTAMP WHERE task_id = NEW.task_id;
END;

-- Cache cleanup trigger
CREATE TRIGGER cleanup_old_cache
AFTER INSERT ON llm_cache
BEGIN
    DELETE FROM llm_cache 
    WHERE last_accessed < datetime('now', '-7 days') 
    AND access_count < 5;
END;

-- Resource usage alert trigger
CREATE TRIGGER resource_usage_alert
AFTER INSERT ON resource_usage
FOR EACH ROW
WHEN NEW.estimated_cost > 0.30
BEGIN
    INSERT INTO system_events (event_type, event_source, severity, message, context_json)
    VALUES ('high_cost_alert', 'resource_monitor', 'warning', 
            'Task exceeded 70% of budget limit',
            json_object('task_id', NEW.task_id, 'cost', NEW.estimated_cost));
END;

-- Experiment completion trigger
CREATE TRIGGER experiment_completion_check
AFTER INSERT ON experiment_results
BEGIN
    UPDATE experiments 
    SET status = 'completed',
        completed_at = CURRENT_TIMESTAMP
    WHERE experiment_id = NEW.experiment_id
    AND NOT EXISTS (
        SELECT 1 FROM arc_tasks t
        WHERE t.task_id IN (
            SELECT json_extract(value, '$') 
            FROM json_each(experiments.task_selection_json, '$.task_ids')
        )
        AND t.task_id NOT IN (
            SELECT task_id FROM experiment_results 
            WHERE experiment_id = NEW.experiment_id
        )
    );
END;
```

## Indexes for Performance

```sql
-- Composite indexes for common queries
CREATE INDEX idx_submissions_strategy_success 
ON task_submissions(strategy_used, is_correct);

CREATE INDEX idx_metrics_strategy_date 
ON strategy_metrics(strategy_type, recorded_at);

CREATE INDEX idx_cache_access 
ON llm_cache(last_accessed, access_count);

-- Full-text search on system events
CREATE VIRTUAL TABLE system_events_fts USING fts5(
    message, context_json, content=system_events
);
```
