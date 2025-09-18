-- PostgreSQL initialization script for AI Content Creator
-- This script sets up the database schema for task tracking and metadata

-- Create database (if running manually)
-- CREATE DATABASE ai_content_db;

-- Connect to the database
\c ai_content_db;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create enum types
CREATE TYPE task_status AS ENUM ('pending', 'processing', 'completed', 'failed', 'cancelled');
CREATE TYPE task_type AS ENUM ('upscale', 'transcribe', 'tts', 'detect_products', 'annotate');
CREATE TYPE file_type AS ENUM ('video', 'audio', 'image', 'text', 'json');

-- Create tasks table
CREATE TABLE IF NOT EXISTS tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_type task_type NOT NULL,
    status task_status DEFAULT 'pending',
    input_path TEXT,
    output_path TEXT,
    parameters JSONB,
    result JSONB,
    error_message TEXT,
    progress INTEGER DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
    priority INTEGER DEFAULT 1 CHECK (priority >= 1 AND priority <= 3),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    user_id TEXT,
    session_id TEXT
);

-- Create files table for metadata
CREATE TABLE IF NOT EXISTS files (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename TEXT NOT NULL,
    original_filename TEXT,
    file_path TEXT NOT NULL,
    file_type file_type NOT NULL,
    file_size BIGINT,
    mime_type TEXT,
    checksum TEXT,
    metadata JSONB,
    task_id UUID REFERENCES tasks(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE
);

-- Create processing_stats table for analytics
CREATE TABLE IF NOT EXISTS processing_stats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id UUID REFERENCES tasks(id) ON DELETE CASCADE,
    task_type task_type NOT NULL,
    processing_time_seconds INTEGER,
    input_file_size BIGINT,
    output_file_size BIGINT,
    gpu_used BOOLEAN DEFAULT FALSE,
    model_used TEXT,
    success BOOLEAN,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create user_sessions table for session management
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id TEXT UNIQUE NOT NULL,
    user_agent TEXT,
    ip_address INET,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() + INTERVAL '24 hours'
);

-- Create system_metrics table for monitoring
CREATE TABLE IF NOT EXISTS system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name TEXT NOT NULL,
    metric_value NUMERIC,
    metric_type TEXT, -- 'counter', 'gauge', 'histogram'
    labels JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_type ON tasks(task_type);
CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks(created_at);
CREATE INDEX IF NOT EXISTS idx_tasks_user_id ON tasks(user_id);
CREATE INDEX IF NOT EXISTS idx_tasks_session_id ON tasks(session_id);

CREATE INDEX IF NOT EXISTS idx_files_task_id ON files(task_id);
CREATE INDEX IF NOT EXISTS idx_files_type ON files(file_type);
CREATE INDEX IF NOT EXISTS idx_files_created_at ON files(created_at);
CREATE INDEX IF NOT EXISTS idx_files_expires_at ON files(expires_at);

CREATE INDEX IF NOT EXISTS idx_processing_stats_task_type ON processing_stats(task_type);
CREATE INDEX IF NOT EXISTS idx_processing_stats_created_at ON processing_stats(created_at);

CREATE INDEX IF NOT EXISTS idx_user_sessions_session_id ON user_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_expires_at ON user_sessions(expires_at);

CREATE INDEX IF NOT EXISTS idx_system_metrics_name_timestamp ON system_metrics(metric_name, timestamp);

-- Create GIN index for JSONB columns
CREATE INDEX IF NOT EXISTS idx_tasks_parameters_gin ON tasks USING GIN(parameters);
CREATE INDEX IF NOT EXISTS idx_tasks_result_gin ON tasks USING GIN(result);
CREATE INDEX IF NOT EXISTS idx_files_metadata_gin ON files USING GIN(metadata);

-- Create functions for automatic timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add updated_at column to tables that need it
ALTER TABLE user_sessions ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();

-- Create trigger for user_sessions
DROP TRIGGER IF EXISTS update_user_sessions_updated_at ON user_sessions;
CREATE TRIGGER update_user_sessions_updated_at
    BEFORE UPDATE ON user_sessions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create function to clean up expired files
CREATE OR REPLACE FUNCTION cleanup_expired_files()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM files
    WHERE expires_at IS NOT NULL
    AND expires_at < NOW();

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Create function to get task statistics
CREATE OR REPLACE FUNCTION get_task_statistics(
    start_date TIMESTAMP WITH TIME ZONE DEFAULT NOW() - INTERVAL '30 days',
    end_date TIMESTAMP WITH TIME ZONE DEFAULT NOW()
)
RETURNS TABLE (
    task_type task_type,
    total_tasks BIGINT,
    completed_tasks BIGINT,
    failed_tasks BIGINT,
    avg_processing_time NUMERIC,
    success_rate NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        t.task_type,
        COUNT(*) as total_tasks,
        COUNT(*) FILTER (WHERE t.status = 'completed') as completed_tasks,
        COUNT(*) FILTER (WHERE t.status = 'failed') as failed_tasks,
        AVG(ps.processing_time_seconds) as avg_processing_time,
        ROUND(
            (COUNT(*) FILTER (WHERE t.status = 'completed')::NUMERIC / COUNT(*)::NUMERIC) * 100,
            2
        ) as success_rate
    FROM tasks t
    LEFT JOIN processing_stats ps ON t.id = ps.task_id
    WHERE t.created_at >= start_date AND t.created_at <= end_date
    GROUP BY t.task_type
    ORDER BY total_tasks DESC;
END;
$$ LANGUAGE plpgsql;

-- Create function to insert system metrics
CREATE OR REPLACE FUNCTION insert_system_metric(
    p_metric_name TEXT,
    p_metric_value NUMERIC,
    p_metric_type TEXT DEFAULT 'gauge',
    p_labels JSONB DEFAULT '{}'::jsonb
)
RETURNS UUID AS $$
DECLARE
    metric_id UUID;
BEGIN
    INSERT INTO system_metrics (metric_name, metric_value, metric_type, labels)
    VALUES (p_metric_name, p_metric_value, p_metric_type, p_labels)
    RETURNING id INTO metric_id;

    RETURN metric_id;
END;
$$ LANGUAGE plpgsql;

-- Create views for common queries
CREATE OR REPLACE VIEW active_tasks AS
SELECT
    id,
    task_type,
    status,
    progress,
    created_at,
    started_at,
    user_id,
    session_id
FROM tasks
WHERE status IN ('pending', 'processing')
ORDER BY priority ASC, created_at ASC;

CREATE OR REPLACE VIEW recent_completed_tasks AS
SELECT
    id,
    task_type,
    status,
    progress,
    created_at,
    started_at,
    completed_at,
    (EXTRACT(EPOCH FROM (completed_at - started_at)))::INTEGER as processing_seconds
FROM tasks
WHERE status = 'completed'
AND completed_at > NOW() - INTERVAL '24 hours'
ORDER BY completed_at DESC;

-- Insert some initial system metrics
INSERT INTO system_metrics (metric_name, metric_value, metric_type, labels) VALUES
('system_startup', 1, 'counter', '{"component": "database"}'),
('schema_version', 1.0, 'gauge', '{"version": "1.0.0"}');

-- Grant permissions (adjust as needed for your security requirements)
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO ai_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO ai_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO ai_user;

-- Print setup completion
DO $$
BEGIN
    RAISE NOTICE 'AI Content Creator database schema initialized successfully!';
    RAISE NOTICE 'Database: %', current_database();
    RAISE NOTICE 'Schema version: 1.0.0';
    RAISE NOTICE 'Tables created: tasks, files, processing_stats, user_sessions, system_metrics';
END $$;