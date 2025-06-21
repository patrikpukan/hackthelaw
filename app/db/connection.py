from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
import asyncio

from app.utils.config import settings
from app.db.models import Base

# Create async engine
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    future=True,
    pool_pre_ping=True,
    pool_recycle=300,
)

# Create async session maker
async_session = async_sessionmaker(
    engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)


async def get_db() -> AsyncSession:
    """Dependency for getting database session."""
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """Initialize database tables."""
    async with engine.begin() as conn:
        # Check if pgvector extension is available (for future vector storage)
        try:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        except Exception as e:
            print(f"Note: pgvector extension not available: {e}")
        
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    """Close database connection."""
    await engine.dispose()


# Test database connection
async def test_connection():
    """Test database connection."""
    try:
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT 1"))
            return result.scalar() == 1
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False


if __name__ == "__main__":
    # Test connection
    async def main():
        connected = await test_connection()
        if connected:
            print("✅ Database connection successful")
            await init_db()
            print("✅ Database tables created")
        else:
            print("❌ Database connection failed")
    
    asyncio.run(main()) 