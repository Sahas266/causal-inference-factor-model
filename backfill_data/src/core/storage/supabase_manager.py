"""Supabase client singleton for database operations"""

import os
from typing import Optional
import logging
from supabase import create_client, Client
from threading import Lock

logger = logging.getLogger('backfill_system.storage')


class SupabaseManager:
    """
    Singleton manager for Supabase client.
    Ensures single shared connection pool across all threads.
    """
    
    _instance: Optional['SupabaseManager'] = None
    _lock: Lock = Lock()
    _client: Optional[Client] = None
    
    def __new__(cls):
        """Implement singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize Supabase client on first instantiation"""
        if self._client is None:
            self._initialize_client()
    
    def _initialize_client(self) -> None:
        """
        Initialize the Supabase client with credentials from environment.
        
        Raises:
            ValueError: If required environment variables are not set
        """
        url = os.getenv('SUPABASE_URL')
        key = os.getenv('SUPABASE_KEY')
        
        if not url:
            raise ValueError(
                "SUPABASE_URL environment variable not set. "
                "Please set it to your Supabase project URL."
            )
        
        if not key:
            raise ValueError(
                "SUPABASE_KEY environment variable not set. "
                "Please set it to your Supabase service role key."
            )
        
        try:
            self._client = create_client(url, key)
            logger.info(f"Supabase client initialized successfully: {url}")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            raise
    
    @property
    def client(self) -> Client:
        """
        Get the Supabase client instance.
        
        Returns:
            Supabase client
            
        Raises:
            RuntimeError: If client is not initialized
        """
        if self._client is None:
            raise RuntimeError("Supabase client not initialized")
        return self._client
    
    def get_client(self) -> Client:
        """
        Get the Supabase client instance (alias for client property).
        
        Returns:
            Supabase client
        """
        return self.client
    
    def test_connection(self) -> bool:
        """
        Test the database connection.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # Try a simple query to test connection
            result = self.client.table('backfill_progress').select('id').limit(1).execute()
            logger.info("Supabase connection test successful")
            return True
        except Exception as e:
            logger.error(f"Supabase connection test failed: {e}")
            return False
    
    def execute_sql(self, sql: str) -> dict:
        """
        Execute raw SQL query.
        
        Args:
            sql: SQL query string
            
        Returns:
            Query result
            
        Raises:
            Exception: If query fails
        """
        try:
            result = self.client.rpc('exec_sql', {'sql': sql}).execute()
            return result.data
        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            raise
    
    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset the singleton instance (mainly for testing).
        """
        with cls._lock:
            cls._instance = None
            cls._client = None
            logger.info("SupabaseManager instance reset")

