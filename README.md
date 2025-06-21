# Legal Memory System - Hack the Law 2025

A sophisticated legal document processing system designed for the **Hack the Law 2025** hackathon challenge. This prototype demonstrates how AI can behave like a junior lawyer with perfect memory, tracking legal positions across multiple documents and reasoning about relationships that span contracts, emails, policies, and negotiations.

## Challenge Background

In legal work, risk and meaning rarely live in just one document. They emerge across a web of contracts, emails, policies, side letters, and internal notes, often written by different people at different times. Lawyers naturally connect these dots—they remember what was agreed in one place and apply it somewhere else, notice contradictions between documents, and track how legal positions evolve over time.

Most AI tools treat each document as a closed box and lose sight of the bigger picture. This prototype addresses that limitation by creating a **legal memory system** that can recall relevant past agreements, spot conflicts across matters, and determine whether clauses are consistent with previous negotiations.

## Our Solution

This system goes beyond simple summarization or search to provide intelligent legal memory capabilities:

## Features

### Core Legal Memory Capabilities

- **Cross-Document Position Tracking**: Monitor how legal positions evolve across drafts, contracts, and negotiations
- **Intelligent Conflict Detection**: Automatically flag contradictions, duplications, and subtle inconsistencies across related documents
- **Contextual Content Linking**: Cluster related clauses that discuss the same obligations using different language across contracts, policies, and memos
- **Legal Recall Intelligence**: Answer questions like "Have we agreed to something like this before?" and "What was our fallback position last time?"
- **Timeline-Aware Analysis**: Track how liability caps, termination rights, and other key terms change through negotiation cycles

### Technical Implementation

- **Multi-format Document Processing**: Support for PDF, DOCX, TXT, and RTF files
- **Intelligent Text Chunking**: Legal clause-aware text segmentation
- **Change History Tracking**: Semantic analysis of clause changes over time
- **Interactive Chat Interface**: RAG-powered document Q&A that remembers context across conversations
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

### Ask Legal Memory Questions

```bash
curl -X POST "http://localhost:8000/api/v1/chat/query" \
     -H "Content-Type: application/json" \
     -d '{"message": "Have we previously agreed to a 30-day termination clause with this client?"}'

curl -X POST "http://localhost:8000/api/v1/chat/query" \
     -H "Content-Type: application/json" \
     -d '{"message": "What liability caps have we negotiated across all contracts this year?"}'

curl -X POST "http://localhost:8000/api/v1/chat/query" \
     -H "Content-Type: application/json" \
     -d '{"message": "Are there any conflicting confidentiality terms between the NDA and service agreement?"}'
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

| Variable          | Description                      | Default            |
| ----------------- | -------------------------------- | ------------------ |
| `DEBUG`           | Enable debug mode                | `false`            |
| `DATABASE_URL`    | PostgreSQL connection string     | Required           |
| `REDIS_URL`       | Redis connection string          | Required           |
| `WEAVIATE_URL`    | Weaviate endpoint                | Required           |
| `MAX_FILE_SIZE`   | Maximum upload file size (bytes) | `52428800`         |
| `CHUNK_SIZE`      | Text chunking size (tokens)      | `500`              |
| `EMBEDDING_MODEL` | Sentence transformer model       | `all-MiniLM-L6-v2` |

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

### Phase 1 (Hackathon Demo)

- ✅ Multi-document ingestion and processing
- ✅ Legal clause-aware chunking and vectorization
- ✅ Cross-document semantic search
- ✅ Basic conflict detection framework
- ✅ RESTful API with Docker deployment

### Phase 2 (Post-Hackathon Enhancement)

- 🔄 Advanced legal memory queries and reasoning
- 🔄 Timeline-based position tracking
- 🔄 Sophisticated conflict resolution suggestions
- ⏳ Enhanced clause similarity matching
- ⏳ Interactive legal memory dashboard

### Phase 3 (Production Roadmap)

- ⏳ Machine learning-enhanced pattern recognition
- ⏳ Integration with document management systems
- ⏳ Advanced analytics and negotiation insights
- ⏳ Multi-tenant enterprise deployment
- ⏳ Real-time collaboration features

## Hack the Law 2025 - Innovation Focus

This prototype demonstrates how AI can transcend traditional document processing limitations by:

1. **Thinking Like a Lawyer**: Understanding that legal meaning spans multiple documents and evolves over time
2. **Perfect Memory**: Never forgetting previous agreements, positions, or negotiation outcomes
3. **Pattern Recognition**: Identifying subtle relationships and conflicts that human reviewers might miss
4. **Contextual Intelligence**: Providing answers that consider the full legal context, not just isolated document fragments
5. **Proactive Insights**: Surfacing relevant historical information before it's explicitly requested

The system serves as a **digital legal memory** that can be consulted and reasoned with, moving beyond simple search to provide intelligent, context-aware legal assistance.

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
