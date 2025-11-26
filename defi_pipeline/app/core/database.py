"""
Database connection management and session handling.
Provides SQLAlchemy engine, session management, and TimescaleDB utilities.
"""

from typing import Generator
from contextlib import contextmanager
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from app.config import database_settings
from app.models.database import Base, create_hypertables, create_continuous_aggregates
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Global database engine
_engine = None
_SessionLocal = None


def get_engine():
    """Get or create the database engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            database_settings.url,
            poolclass=QueuePool,
            pool_size=database_settings.pool_size,
            max_overflow=database_settings.max_overflow,
            pool_timeout=database_settings.pool_timeout,
            pool_recycle=database_settings.pool_recycle,
            echo=database_settings.url.startswith("sqlite") or False,  # Enable for SQLite debugging
            future=True,  # Use SQLAlchemy 2.0 style
        )

        # Add event listeners for connection management
        @event.listens_for(_engine, "connect")
        def set_timezone(dbapi_connection, connection_record):
            """Set timezone to UTC for all connections."""
            if dbapi_connection.driver == "psycopg2":
                cursor = dbapi_connection.cursor()
                cursor.execute("SET TIME ZONE 'UTC'")
                cursor.close()

        @event.listens_for(_engine, "checkout")
        def ping_connection(dbapi_connection, connection_record, connection_proxy):
            """Ping connections to ensure they're alive."""
            if dbapi_connection.driver == "psycopg2":
                cursor = dbapi_connection.cursor()
                cursor.execute("SELECT 1")
                cursor.close()

        logger.info("Database engine created")

    return _engine


def get_session_local():
    """Get or create the session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=get_engine(),
            expire_on_commit=False
        )
        logger.info("Database session factory created")

    return _SessionLocal


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI to get database sessions.
    Provides a database session that is automatically closed after use.
    """
    db = get_session_local()()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_session():
    """
    Context manager for database sessions.
    Useful for background tasks and utilities.
    """
    db = get_session_local()()
    try:
        yield db
    finally:
        db.close()


def init_database() -> None:
    """
    Initialize the database schema.
    Creates all tables and TimescaleDB hypertables.
    """
    try:
        logger.info("Initializing database schema...")

        # Create all tables
        Base.metadata.create_all(bind=get_engine())

        # Create TimescaleDB hypertables
        if database_settings.is_timescaledb_enabled:
            with get_engine().connect() as conn:
                create_hypertables(conn)
                create_continuous_aggregates(conn)

        logger.info("Database schema initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def reset_database() -> None:
    """
    Reset the database by dropping and recreating all tables.
    WARNING: This will delete all data!
    """
    try:
        logger.warning("Resetting database - all data will be lost!")

        # Drop all tables
        Base.metadata.drop_all(bind=get_engine())

        # Reinitialize
        init_database()

        logger.info("Database reset complete")

    except Exception as e:
        logger.error(f"Failed to reset database: {e}")
        raise


def check_database_health() -> bool:
    """
    Check if the database is healthy and accessible.

    Returns:
        bool: True if healthy, False otherwise
    """
    try:
        with get_db_session() as db:
            db.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


def get_database_stats() -> dict:
    """
    Get database connection and performance statistics.

    Returns:
        dict: Database statistics
    """
    engine = get_engine()
    pool = engine.pool

    return {
        "pool_size": pool.size(),
        "checked_in": pool.checkedin(),
        "checked_out": pool.checkedout(),
        "invalid": pool.invalid(),
        "overflow": pool.overflow(),
        "total_connections": pool.size() + pool.overflow(),
        "available_connections": pool.checkedin(),
        "url": database_settings.url.replace(
            database_settings.url.split('@')[0].split('//')[1].split(':')[1],
            "***"  # Hide password
        ) if '@' in database_settings.url else database_settings.url,
    }
