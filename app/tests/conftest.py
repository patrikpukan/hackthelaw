import pytest
import asyncio
from typing import AsyncGenerator, Generator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from httpx import AsyncClient
import tempfile
import shutil
from pathlib import Path

from app.api.main import create_app
from app.db.models import Base
from app.db.connection import get_db
from app.utils.config import settings


# Test database URL
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_engine():
    """Create test database engine."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a database session for testing."""
    async_session = sessionmaker(
        bind=test_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
        await session.rollback()


@pytest.fixture
async def test_app(db_session):
    """Create test FastAPI application."""
    app = create_app()
    
    # Override database dependency
    async def override_get_db():
        yield db_session
    
    app.dependency_overrides[get_db] = override_get_db
    
    yield app
    
    # Clear overrides
    app.dependency_overrides.clear()


@pytest.fixture
async def client(test_app) -> AsyncGenerator[AsyncClient, None]:
    """Create test HTTP client."""
    async with AsyncClient(app=test_app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def temp_upload_dir():
    """Create temporary upload directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_pdf_content():
    """Sample PDF content for testing."""
    return b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"


@pytest.fixture
def sample_text_content():
    """Sample text content for testing."""
    return """EMPLOYMENT AGREEMENT

This Employment Agreement is entered into between Company ABC and John Doe.

SECTION 1. TERM
The term of employment shall be for a period of two (2) years.

SECTION 2. COMPENSATION
Employee shall receive a salary of $100,000 per year.

SECTION 3. TERMINATION
Either party may terminate this agreement with thirty (30) days notice.
"""


@pytest.fixture
def sample_contract_with_changes():
    """Sample contract with version changes for testing."""
    return {
        "version_1": """EMPLOYMENT AGREEMENT

SECTION 1. TERM
The term of employment shall be for a period of one (1) year.

SECTION 2. COMPENSATION
Employee shall receive a salary of $80,000 per year.

SECTION 3. TERMINATION
Either party may terminate this agreement with sixty (60) days notice.
""",
        "version_2": """EMPLOYMENT AGREEMENT

SECTION 1. TERM
The term of employment shall be for a period of two (2) years.

SECTION 2. COMPENSATION
Employee shall receive a salary of $100,000 per year.

SECTION 3. TERMINATION
Either party may terminate this agreement with thirty (30) days notice.
"""
    }


@pytest.fixture(autouse=True)
def reset_settings():
    """Reset settings after each test."""
    yield
    # Reset any modified settings if needed
    pass 