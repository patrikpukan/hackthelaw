# Legal RAG Agent with Change History

A sophisticated legal document processing system that uses Retrieval-Augmented Generation (RAG) to provide intelligent document analysis with comprehensive change history tracking.

## Features

- **Multi-format Document Processing**: Support for PDF, DOCX, TXT, and RTF files
- **Intelligent Text Chunking**: Legal clause-aware text segmentation
- **Change History Tracking**: Semantic analysis of clause changes over time
- **Conflict Detection**: Automated identification of conflicting clauses
- **Interactive Chat Interface**: RAG-powered document Q&A
- **Vector Search**: Semantic similarity search across document corpus
- **Background Processing**: Async document processing with Celery workers

## Architecture

- **FastAPI**: Modern async web framework for API endpoints
- **PostgreSQL**: Primary database with pgvector extension for embeddings
- **Redis**: Message broker and caching layer
- **Weaviate**: Vector database for semantic search
- **Celery**: Distributed task queue for background processing
- **Docker**: Containerized deployment

## Quick Start

### Prerequisites

- Docker and Docker Compose
- At least 4GB RAM available for containers

### 1. Clone and Setup

```bash
git clone <repository-url>
cd hackathon
```

### 2. Environment Configuration

```bash
cp .env.example .env
# Edit .env file with your preferred settings
```

### 3. Start Services

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api
```

### 4. Verify Installation

```bash
# Check API health
curl http://localhost:8000/health

# Access API documentation
open http://localhost:8000/docs

# Monitor Celery tasks
open http://localhost:5555  # Flower UI
```

## API Endpoints

### Document Management

- `POST /api/v1/documents/upload` - Upload documents for processing
- `GET /api/v1/documents/` - List uploaded documents
- `GET /api/v1/documents/{id}` - Get document details
- `DELETE /api/v1/documents/{id}` - Delete document
- `GET /api/v1/documents/{id}/status` - Check processing status

### Chat Interface

- `POST /api/v1/chat/query` - Ask questions about documents
- `GET /api/v1/chat/sessions` - List chat sessions
- `GET /api/v1/chat/sessions/{id}/messages` - Get chat history

### Change History

- `GET /api/v1/history/clauses/{id}` - Get clause change history
- `GET /api/v1/history/documents/{id}/changes` - Get document changes
- `GET /api/v1/history/recent` - Get recent changes across all documents

### Conflict Detection

- `GET /api/v1/conflicts/` - List detected conflicts
- `GET /api/v1/conflicts/{id}` - Get conflict details
- `PATCH /api/v1/conflicts/{id}/status` - Update conflict status

## Usage Examples

### Upload a Document

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@contract.pdf"
```

### Ask Questions

```bash
curl -X POST "http://localhost:8000/api/v1/chat/query" \
     -H "Content-Type: application/json" \
     -d '{"message": "What are the payment terms in the uploaded contracts?"}'
```

### Check Processing Status

```bash
curl "http://localhost:8000/api/v1/documents/{document_id}/status"
```

## Development

### Local Development Setup

```bash
# Install dependencies
cd app
pip install -r requirements.txt

# Run database migrations
python -m alembic upgrade head

# Start development server
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/
```

### Code Formatting

```bash
# Format code
black app/

# Check linting
flake8 app/
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DEBUG` | Enable debug mode | `false` |
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `REDIS_URL` | Redis connection string | Required |
| `WEAVIATE_URL` | Weaviate endpoint | Required |
| `MAX_FILE_SIZE` | Maximum upload file size (bytes) | `52428800` |
| `CHUNK_SIZE` | Text chunking size (tokens) | `500` |
| `EMBEDDING_MODEL` | Sentence transformer model | `all-MiniLM-L6-v2` |

### Chunking Strategies

- `hybrid` (default): Legal clause detection + sentence chunking
- `legal`: Legal clause-based chunking only
- `sentence`: Sentence-based chunking with overlap
- `paragraph`: Simple paragraph-based chunking

## Monitoring

### Health Checks

- API: `http://localhost:8000/health`
- Database: Check via API or direct connection
- Redis: Check via Flower UI
- Weaviate: `http://localhost:8080/v1/meta`

### Logs

```bash
# View API logs
docker-compose logs -f api

# View worker logs
docker-compose logs -f worker

# View all logs
docker-compose logs -f
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Increase Docker memory allocation to 4GB+
2. **Slow Processing**: Check if Weaviate is properly started
3. **Database Connection**: Verify PostgreSQL is accessible

### Database Reset

```bash
# Reset database (DANGER: loses all data)
docker-compose down -v
docker-compose up -d postgres
# Wait for postgres to start, then:
docker-compose up -d
```

## Roadmap

### Phase 1 (Current)
- ✅ Basic document upload and processing
- ✅ Text extraction and chunking
- ✅ Database schema and API structure
- ✅ Docker deployment setup

### Phase 2 (Next)
- 🔄 Vector embeddings and similarity search
- 🔄 Basic RAG chat functionality
- ⏳ Change history detection
- ⏳ Conflict identification

### Phase 3 (Future)
- ⏳ Advanced clause matching
- ⏳ ML-based conflict resolution
- ⏳ Web frontend interface
- ⏳ Advanced analytics and reporting

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support, please open an issue in the repository or contact the development team. 