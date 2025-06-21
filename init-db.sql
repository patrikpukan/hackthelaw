-- Initialize Legal RAG Database Schema
-- This script creates all tables based on SQLAlchemy models

-- Create pgvector extension if available
CREATE EXTENSION IF NOT EXISTS vector;

-- Set database defaults
ALTER DATABASE legalrag SET timezone TO 'UTC';

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create documents table (main table for legal documents)
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename VARCHAR(255) NOT NULL,
    file_type VARCHAR(10) NOT NULL,
    upload_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    file_size BIGINT NOT NULL,
    extracted_text TEXT,
    document_hash VARCHAR(64) UNIQUE,
    user_id UUID,
    processing_status VARCHAR(20) DEFAULT 'pending',

    -- Versioning fields
    document_family_id UUID,
    version_number INTEGER DEFAULT 1,
    is_latest_version BOOLEAN DEFAULT TRUE,
    parent_document_id UUID REFERENCES documents(id),
    version_description TEXT,
    similarity_to_parent FLOAT,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create embeddings table (for vector representations)
CREATE TABLE IF NOT EXISTS embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_id VARCHAR(50) NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_type VARCHAR(20) DEFAULT 'paragraph',
    start_char INTEGER,
    end_char INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create clause_versions table (for tracking legal clauses)
CREATE TABLE IF NOT EXISTS clause_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    clause_id UUID NOT NULL,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    clause_text TEXT NOT NULL,
    clause_type VARCHAR(50),
    version_number INTEGER DEFAULT 1,
    change_type VARCHAR(20),
    change_description TEXT,
    confidence_score FLOAT,
    previous_version_id UUID REFERENCES clause_versions(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create clause_changes table (for tracking changes between clause versions)
CREATE TABLE IF NOT EXISTS clause_changes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    from_version_id UUID REFERENCES clause_versions(id),
    to_version_id UUID REFERENCES clause_versions(id),
    change_summary TEXT,
    semantic_similarity FLOAT,
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create detected_conflicts table (for storing identified conflicts)
CREATE TABLE IF NOT EXISTS detected_conflicts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    clause_1_id UUID REFERENCES clause_versions(id),
    clause_2_id UUID REFERENCES clause_versions(id),
    conflict_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) DEFAULT 'medium',
    description TEXT,
    suggested_resolution TEXT,
    status VARCHAR(20) DEFAULT 'open',
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    resolved_at TIMESTAMP WITH TIME ZONE
);

-- Create chat_sessions table (for user chat interactions)
CREATE TABLE IF NOT EXISTS chat_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID,
    session_name VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create chat_messages table (for individual messages)
CREATE TABLE IF NOT EXISTS chat_messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    message_type VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    context_documents TEXT,
    message_metadata TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create document_changes table (for tracking document changes)
CREATE TABLE IF NOT EXISTS document_changes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    from_document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    to_document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    change_type VARCHAR(20) NOT NULL,
    section_title VARCHAR(255),
    old_text TEXT,
    new_text TEXT,
    start_position INTEGER,
    end_position INTEGER,
    similarity_score FLOAT,
    change_summary TEXT,
    confidence_score FLOAT DEFAULT 1.0,
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create document_versions table (for enhanced version tracking)
CREATE TABLE IF NOT EXISTS document_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    version_tag VARCHAR(50),
    version_notes TEXT,
    author VARCHAR(255),
    approval_status VARCHAR(20) DEFAULT 'draft',
    approved_by VARCHAR(255),
    approved_at TIMESTAMP WITH TIME ZONE,
    is_published BOOLEAN DEFAULT FALSE,
    change_summary TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create search_history table (for tracking searches)
CREATE TABLE IF NOT EXISTS search_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID,
    query TEXT NOT NULL,
    search_type VARCHAR(20) DEFAULT 'all',
    target_versions TEXT,
    results_count INTEGER DEFAULT 0,
    execution_time FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create processing_tasks table (for background tasks)
CREATE TABLE IF NOT EXISTS processing_tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_type VARCHAR(50) NOT NULL,
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    status VARCHAR(20) DEFAULT 'pending',
    progress FLOAT DEFAULT 0.0,
    error_message TEXT,
    result_data TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Create indexes for better performance
-- Documents table indexes
CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename);
CREATE INDEX IF NOT EXISTS idx_documents_file_type ON documents(file_type);
CREATE INDEX IF NOT EXISTS idx_documents_upload_date ON documents(upload_date);
CREATE INDEX IF NOT EXISTS idx_documents_processing_status ON documents(processing_status);
CREATE INDEX IF NOT EXISTS idx_documents_document_family_id ON documents(document_family_id);
CREATE INDEX IF NOT EXISTS idx_documents_version_number ON documents(version_number);
CREATE INDEX IF NOT EXISTS idx_documents_is_latest_version ON documents(is_latest_version);
CREATE INDEX IF NOT EXISTS idx_documents_parent_document_id ON documents(parent_document_id);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at);
CREATE INDEX IF NOT EXISTS idx_documents_updated_at ON documents(updated_at);
CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id);

-- Embeddings table indexes
CREATE INDEX IF NOT EXISTS idx_embeddings_document_id ON embeddings(document_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_id ON embeddings(chunk_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_type ON embeddings(chunk_type);
CREATE INDEX IF NOT EXISTS idx_embeddings_created_at ON embeddings(created_at);

-- Clause versions indexes
CREATE INDEX IF NOT EXISTS idx_clause_versions_clause_id ON clause_versions(clause_id);
CREATE INDEX IF NOT EXISTS idx_clause_versions_document_id ON clause_versions(document_id);
CREATE INDEX IF NOT EXISTS idx_clause_versions_clause_type ON clause_versions(clause_type);
CREATE INDEX IF NOT EXISTS idx_clause_versions_version_number ON clause_versions(version_number);
CREATE INDEX IF NOT EXISTS idx_clause_versions_created_at ON clause_versions(created_at);

-- Chat sessions indexes
CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_created_at ON chat_sessions(created_at);

-- Chat messages indexes
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_message_type ON chat_messages(message_type);
CREATE INDEX IF NOT EXISTS idx_chat_messages_created_at ON chat_messages(created_at);

-- Processing tasks indexes
CREATE INDEX IF NOT EXISTS idx_processing_tasks_document_id ON processing_tasks(document_id);
CREATE INDEX IF NOT EXISTS idx_processing_tasks_task_type ON processing_tasks(task_type);
CREATE INDEX IF NOT EXISTS idx_processing_tasks_status ON processing_tasks(status);
CREATE INDEX IF NOT EXISTS idx_processing_tasks_created_at ON processing_tasks(created_at);

-- Search history indexes
CREATE INDEX IF NOT EXISTS idx_search_history_user_id ON search_history(user_id);
CREATE INDEX IF NOT EXISTS idx_search_history_search_type ON search_history(search_type);
CREATE INDEX IF NOT EXISTS idx_search_history_created_at ON search_history(created_at);

-- Detected conflicts indexes
CREATE INDEX IF NOT EXISTS idx_detected_conflicts_clause_1_id ON detected_conflicts(clause_1_id);
CREATE INDEX IF NOT EXISTS idx_detected_conflicts_clause_2_id ON detected_conflicts(clause_2_id);
CREATE INDEX IF NOT EXISTS idx_detected_conflicts_conflict_type ON detected_conflicts(conflict_type);
CREATE INDEX IF NOT EXISTS idx_detected_conflicts_severity ON detected_conflicts(severity);
CREATE INDEX IF NOT EXISTS idx_detected_conflicts_status ON detected_conflicts(status);
CREATE INDEX IF NOT EXISTS idx_detected_conflicts_detected_at ON detected_conflicts(detected_at);

-- Document changes indexes
CREATE INDEX IF NOT EXISTS idx_document_changes_from_document_id ON document_changes(from_document_id);
CREATE INDEX IF NOT EXISTS idx_document_changes_to_document_id ON document_changes(to_document_id);
CREATE INDEX IF NOT EXISTS idx_document_changes_change_type ON document_changes(change_type);
CREATE INDEX IF NOT EXISTS idx_document_changes_detected_at ON document_changes(detected_at);

-- Document versions indexes
CREATE INDEX IF NOT EXISTS idx_document_versions_document_id ON document_versions(document_id);
CREATE INDEX IF NOT EXISTS idx_document_versions_approval_status ON document_versions(approval_status);
CREATE INDEX IF NOT EXISTS idx_document_versions_is_published ON document_versions(is_published);
CREATE INDEX IF NOT EXISTS idx_document_versions_created_at ON document_versions(created_at);

-- Add constraints for data integrity using DO blocks to handle IF NOT EXISTS
DO $$
BEGIN
    -- Documents table constraints
    IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints
                   WHERE constraint_name = 'chk_version_number_positive' AND table_name = 'documents') THEN
        ALTER TABLE documents ADD CONSTRAINT chk_version_number_positive
        CHECK (version_number > 0);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints
                   WHERE constraint_name = 'chk_similarity_to_parent_range' AND table_name = 'documents') THEN
        ALTER TABLE documents ADD CONSTRAINT chk_similarity_to_parent_range
        CHECK (similarity_to_parent IS NULL OR (similarity_to_parent >= 0 AND similarity_to_parent <= 1));
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints
                   WHERE constraint_name = 'chk_processing_status_valid' AND table_name = 'documents') THEN
        ALTER TABLE documents ADD CONSTRAINT chk_processing_status_valid
        CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed', 'cancelled'));
    END IF;

    -- Processing tasks constraints
    IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints
                   WHERE constraint_name = 'chk_progress_range' AND table_name = 'processing_tasks') THEN
        ALTER TABLE processing_tasks ADD CONSTRAINT chk_progress_range
        CHECK (progress >= 0 AND progress <= 1);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints
                   WHERE constraint_name = 'chk_status_valid' AND table_name = 'processing_tasks') THEN
        ALTER TABLE processing_tasks ADD CONSTRAINT chk_status_valid
        CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled'));
    END IF;

    -- Chat messages constraints
    IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints
                   WHERE constraint_name = 'chk_message_type_valid' AND table_name = 'chat_messages') THEN
        ALTER TABLE chat_messages ADD CONSTRAINT chk_message_type_valid
        CHECK (message_type IN ('user', 'assistant', 'system'));
    END IF;

    -- Detected conflicts constraints
    IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints
                   WHERE constraint_name = 'chk_severity_valid' AND table_name = 'detected_conflicts') THEN
        ALTER TABLE detected_conflicts ADD CONSTRAINT chk_severity_valid
        CHECK (severity IN ('low', 'medium', 'high', 'critical'));
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints
                   WHERE constraint_name = 'chk_status_valid_conflicts' AND table_name = 'detected_conflicts') THEN
        ALTER TABLE detected_conflicts ADD CONSTRAINT chk_status_valid_conflicts
        CHECK (status IN ('open', 'reviewing', 'resolved', 'dismissed'));
    END IF;

    -- Document versions constraints
    IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints
                   WHERE constraint_name = 'chk_approval_status_valid' AND table_name = 'document_versions') THEN
        ALTER TABLE document_versions ADD CONSTRAINT chk_approval_status_valid
        CHECK (approval_status IN ('draft', 'pending', 'approved', 'rejected'));
    END IF;

    RAISE NOTICE 'All constraints have been processed';
END $$;

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for automatic timestamp updates
DROP TRIGGER IF EXISTS trigger_update_documents_updated_at ON documents;
CREATE TRIGGER trigger_update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS trigger_update_chat_sessions_updated_at ON chat_sessions;
CREATE TRIGGER trigger_update_chat_sessions_updated_at
    BEFORE UPDATE ON chat_sessions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create function to automatically assign document_family_id
CREATE OR REPLACE FUNCTION assign_document_family_id()
RETURNS TRIGGER AS $$
BEGIN
    -- If no document_family_id is provided, use the document's own ID
    IF NEW.document_family_id IS NULL THEN
        NEW.document_family_id = NEW.id;
    END IF;

    -- Ensure timestamps are set
    IF NEW.created_at IS NULL THEN
        NEW.created_at = NOW();
    END IF;

    IF NEW.updated_at IS NULL THEN
        NEW.updated_at = NOW();
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to automatically assign document_family_id
DROP TRIGGER IF EXISTS trigger_assign_document_family_id ON documents;
CREATE TRIGGER trigger_assign_document_family_id
    BEFORE INSERT ON documents
    FOR EACH ROW
    EXECUTE FUNCTION assign_document_family_id();

-- Create function to manage latest version flags
CREATE OR REPLACE FUNCTION manage_latest_version()
RETURNS TRIGGER AS $$
BEGIN
    -- If this is marked as latest version, unmark others in the same family
    IF NEW.is_latest_version = TRUE AND NEW.document_family_id IS NOT NULL THEN
        UPDATE documents
        SET is_latest_version = FALSE
        WHERE document_family_id = NEW.document_family_id
          AND id != NEW.id
          AND is_latest_version = TRUE;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for version management
DROP TRIGGER IF EXISTS trigger_manage_latest_version ON documents;
CREATE TRIGGER trigger_manage_latest_version
    AFTER INSERT OR UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION manage_latest_version();

-- Add helpful comments to tables
COMMENT ON TABLE documents IS 'Legal documents with versioning and change tracking support';
COMMENT ON TABLE embeddings IS 'Vector embeddings metadata for document chunks';
COMMENT ON TABLE clause_versions IS 'Versioned legal clauses extracted from documents';
COMMENT ON TABLE clause_changes IS 'Changes between clause versions';
COMMENT ON TABLE detected_conflicts IS 'Automatically detected conflicts between clauses';
COMMENT ON TABLE chat_sessions IS 'User chat sessions for document analysis';
COMMENT ON TABLE chat_messages IS 'Individual messages within chat sessions';
COMMENT ON TABLE document_changes IS 'Tracked changes between document versions';
COMMENT ON TABLE document_versions IS 'Enhanced version metadata for documents';
COMMENT ON TABLE search_history IS 'History of user searches across document versions';
COMMENT ON TABLE processing_tasks IS 'Background processing tasks for documents';

-- Add column comments for key fields
COMMENT ON COLUMN documents.document_family_id IS 'Groups related document versions together';
COMMENT ON COLUMN documents.version_number IS 'Version number within document family';
COMMENT ON COLUMN documents.is_latest_version IS 'Whether this is the latest version in the family';
COMMENT ON COLUMN documents.parent_document_id IS 'Reference to the parent document version';
COMMENT ON COLUMN documents.version_description IS 'User description of changes in this version';
COMMENT ON COLUMN documents.similarity_to_parent IS 'Similarity score compared to parent version';

-- Create a view for latest documents only
CREATE OR REPLACE VIEW latest_documents AS
SELECT * FROM documents
WHERE is_latest_version = TRUE;

-- Create a view for document families with version counts
CREATE OR REPLACE VIEW document_families AS
SELECT
    document_family_id,
    COUNT(*) as version_count,
    MIN(created_at) as first_version_date,
    MAX(created_at) as latest_version_date,
    MAX(version_number) as latest_version_number
FROM documents
WHERE document_family_id IS NOT NULL
GROUP BY document_family_id;

-- Insert initial data or setup if needed
-- (This section can be used for default data)

-- Log successful initialization
DO $$
DECLARE
    table_count INTEGER;
    index_count INTEGER;
    trigger_count INTEGER;
BEGIN
    -- Count created objects
    SELECT COUNT(*) INTO table_count
    FROM information_schema.tables
    WHERE table_schema = 'public'
    AND table_type = 'BASE TABLE';

    SELECT COUNT(*) INTO index_count
    FROM pg_indexes
    WHERE schemaname = 'public';

    SELECT COUNT(*) INTO trigger_count
    FROM information_schema.triggers
    WHERE trigger_schema = 'public';

    RAISE NOTICE '';
    RAISE NOTICE '=== LEGAL RAG DATABASE INITIALIZATION COMPLETED ===';
    RAISE NOTICE 'Database: legalrag';
    RAISE NOTICE 'Tables created: %', table_count;
    RAISE NOTICE 'Indexes created: %', index_count;
    RAISE NOTICE 'Triggers created: %', trigger_count;
    RAISE NOTICE '';
    RAISE NOTICE 'Features enabled:';
    RAISE NOTICE '✅ Document versioning and families';
    RAISE NOTICE '✅ Legal clause tracking';
    RAISE NOTICE '✅ Conflict detection';
    RAISE NOTICE '✅ Chat sessions';
    RAISE NOTICE '✅ Background processing';
    RAISE NOTICE '✅ Search history';
    RAISE NOTICE '✅ Automatic timestamp management';
    RAISE NOTICE '✅ Data integrity constraints';
    RAISE NOTICE '';
    RAISE NOTICE 'Database is ready for Legal RAG application!';
    RAISE NOTICE '=================================================';
END $$;