"""Database writer for batch upsert operations"""

from typing import List, Dict
import logging
from .supabase_manager import SupabaseManager

logger = logging.getLogger('backfill_system.storage')


class DatabaseWriter:
    """
    Handles batch database write operations with upsert logic.
    Memory-efficient streaming writes with configurable batch sizes.
    """
    
    def __init__(self, supabase_manager: SupabaseManager):
        """
        Initialize database writer.
        
        Args:
            supabase_manager: SupabaseManager instance
        """
        self.supabase_manager = supabase_manager
        self.client = supabase_manager.get_client()
    
    def upsert_batch(
        self,
        table: str,
        data: List[Dict],
        primary_keys: List[str],
        batch_size: int = 1000
    ) -> int:
        """
        Upsert data in batches with conflict resolution.
        
        Args:
            table: Target table name
            data: List of records to upsert
            primary_keys: List of columns that form the primary key
            batch_size: Number of records per batch
            
        Returns:
            Total number of records upserted
            
        Raises:
            Exception: If upsert operation fails
        """
        if not data:
            logger.debug(f"No data to upsert into {table}")
            return 0
        
        total_upserted = 0
        
        # Process in batches
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            
            try:
                # Upsert with conflict resolution, retry on transient errors
                import time as _time
                for attempt in range(3):
                    try:
                        result = self.client.table(table).upsert(
                            batch,
                            on_conflict=','.join(primary_keys)
                        ).execute()
                        break
                    except Exception as retry_err:
                        err_str = str(retry_err).lower()
                        if attempt < 2 and ('disconnect' in err_str or 'timeout' in err_str or 'connection' in err_str):
                            logger.warning(
                                f"Batch {i//batch_size + 1} attempt {attempt+1} failed "
                                f"({retry_err}), retrying in 3s..."
                            )
                            _time.sleep(3)
                            continue
                        raise

                total_upserted += len(batch)
                logger.debug(
                    f"Upserted {len(batch)} records to {table} "
                    f"(total: {total_upserted}/{len(data)})"
                )

            except Exception as e:
                logger.error(
                    f"Failed to upsert batch {i//batch_size + 1} to {table}: {e}",
                    exc_info=True
                )
                raise
        
        logger.info(f"Successfully upserted {total_upserted} records to {table}")
        return total_upserted
    
    def insert_batch(
        self,
        table: str,
        data: List[Dict],
        batch_size: int = 1000
    ) -> int:
        """
        Insert data in batches (fails on conflict).
        
        Args:
            table: Target table name
            data: List of records to insert
            batch_size: Number of records per batch
            
        Returns:
            Total number of records inserted
        """
        if not data:
            logger.debug(f"No data to insert into {table}")
            return 0
        
        total_inserted = 0
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            
            try:
                result = self.client.table(table).insert(batch).execute()
                total_inserted += len(batch)
                logger.debug(
                    f"Inserted {len(batch)} records to {table} "
                    f"(total: {total_inserted}/{len(data)})"
                )
                
            except Exception as e:
                logger.error(
                    f"Failed to insert batch {i//batch_size + 1} to {table}: {e}",
                    exc_info=True
                )
                raise
        
        logger.info(f"Successfully inserted {total_inserted} records to {table}")
        return total_inserted
    
    def update_record(
        self,
        table: str,
        match_criteria: Dict,
        updates: Dict
    ) -> bool:
        """
        Update a single record.
        
        Args:
            table: Target table name
            match_criteria: Dictionary of column:value pairs to match
            updates: Dictionary of column:value pairs to update
            
        Returns:
            True if update succeeded
        """
        try:
            query = self.client.table(table)
            
            # Apply match criteria
            for column, value in match_criteria.items():
                query = query.eq(column, value)
            
            result = query.update(updates).execute()
            logger.debug(f"Updated record in {table}: {match_criteria}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update record in {table}: {e}")
            raise
    
    def delete_records(
        self,
        table: str,
        match_criteria: Dict
    ) -> int:
        """
        Delete records matching criteria.
        
        Args:
            table: Target table name
            match_criteria: Dictionary of column:value pairs to match
            
        Returns:
            Number of records deleted
        """
        try:
            query = self.client.table(table)
            
            # Apply match criteria
            for column, value in match_criteria.items():
                query = query.eq(column, value)
            
            result = query.delete().execute()
            count = len(result.data) if result.data else 0
            logger.info(f"Deleted {count} records from {table}")
            return count
            
        except Exception as e:
            logger.error(f"Failed to delete records from {table}: {e}")
            raise
    
    def query_records(
        self,
        table: str,
        columns: str = '*',
        filters: Dict = None,
        limit: int = None
    ) -> List[Dict]:
        """
        Query records from table.
        
        Args:
            table: Target table name
            columns: Columns to select (default '*')
            filters: Dictionary of column:value pairs to filter
            limit: Maximum number of records to return
            
        Returns:
            List of matching records
        """
        try:
            query = self.client.table(table).select(columns)
            
            # Apply filters
            if filters:
                for column, value in filters.items():
                    query = query.eq(column, value)
            
            # Apply limit
            if limit:
                query = query.limit(limit)
            
            result = query.execute()
            return result.data if result.data else []
            
        except Exception as e:
            logger.error(f"Failed to query records from {table}: {e}")
            raise

