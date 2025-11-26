#!/usr/bin/env python3
"""
Database initialization script.
Creates database schema, TimescaleDB hypertables, and initial data.
"""

import sys
import os
from pathlib import Path

# Add the app directory to Python path
app_dir = Path(__file__).parent.parent / "app"
sys.path.insert(0, str(app_dir))

from app.core.database import init_database, check_database_health
from app.config import settings
from app.utils.logger import setup_logging

# Set up logging
setup_logging(level="INFO")

def main():
    """Initialize the database."""
    print("Initializing DeFi Data Pipeline database...")

    try:
        # Check database connection
        print("Checking database connection...")
        if not check_database_health():
            print("❌ Database connection failed. Please check your DATABASE_URL configuration.")
            sys.exit(1)

        print("✅ Database connection successful")

        # Initialize schema
        print("Creating database schema and TimescaleDB hypertables...")
        init_database()

        print("✅ Database initialization completed successfully!")
        print("\nNext steps:")
        print("1. Start the FastAPI server: uvicorn app.main:app --reload")
        print("2. Start Celery worker: celery -A app.tasks.scheduled worker --loglevel=info")
        print("3. Start Celery beat: celery -A app.tasks.scheduled beat --loglevel=info")
        print("4. Visit http://localhost:8000/docs for API documentation")

    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
