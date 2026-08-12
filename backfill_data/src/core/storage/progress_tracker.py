"""Progress tracking for backfill operations"""

from typing import Optional, Dict
from datetime import datetime
import logging
from .supabase_manager import SupabaseManager

logger = logging.getLogger('backfill_system.storage')


def _to_json_safe(value):
    """
    Recursively convert values to JSON-serializable forms.

    Supabase upserts fail if config payloads contain datetime objects.
    """
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(v) for v in value]
    return value


class ProgressTracker:
    """
    Track backfill progress with checkpoint support.
    Enables resumption from last successful point.
    """
    
    def __init__(self, supabase_manager: SupabaseManager):
        """
        Initialize progress tracker.
        
        Args:
            supabase_manager: SupabaseManager instance
        """
        self.supabase_manager = supabase_manager
        self.client = supabase_manager.get_client()
        self.table = 'backfill_progress'
    
    def get_progress(self, endpoint_id: str) -> Optional[Dict]:
        """
        Get progress record for an endpoint.
        
        Args:
            endpoint_id: Unique endpoint identifier (includes provider)
            
        Returns:
            Progress record or None if not found
        """
        try:
            result = self.client.table(self.table).select('*').eq(
                'endpoint_id', endpoint_id
            ).execute()
            
            if result.data and len(result.data) > 0:
                return result.data[0]
            return None
            
        except Exception as e:
            logger.error(f"Failed to get progress for {endpoint_id}: {e}")
            return None
    
    def initialize_progress(
        self,
        endpoint_id: str,
        provider: str,
        endpoint_type: str,
        table_name: str,
        config: Dict
    ) -> Dict:
        """
        Initialize a new progress record.
        
        Args:
            endpoint_id: Unique endpoint identifier
            provider: Provider name
            endpoint_type: Type of endpoint
            table_name: Target table name
            config: Endpoint configuration
            
        Returns:
            Created progress record
        """
        try:
            progress_data = {
                'endpoint_id': endpoint_id,
                'provider': provider,
                'endpoint_type': endpoint_type,
                'table_name': table_name,
                'status': 'pending',
                'total_records_fetched': 0,
                'total_api_calls': 0,
                'started_at': datetime.utcnow().isoformat(),
                'config': _to_json_safe(config),
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat()
            }
            
            result = self.client.table(self.table).upsert(
                progress_data,
                on_conflict='endpoint_id,provider'
            ).execute()
            
            logger.info(f"Initialized progress tracking for {endpoint_id}")
            return result.data[0] if result.data else progress_data
            
        except Exception as e:
            logger.error(f"Failed to initialize progress for {endpoint_id}: {e}")
            raise
    
    def update_progress(
        self,
        endpoint_id: str,
        last_time: Optional[datetime] = None,
        records_count: int = 0,
        api_calls: int = 1
    ) -> None:
        """
        Update progress for an endpoint (checkpoint).
        
        Args:
            endpoint_id: Unique endpoint identifier
            last_time: Last successfully processed timestamp
            records_count: Number of records processed in this update
            api_calls: Number of API calls made in this update
        """
        try:
            # Get current progress
            current = self.get_progress(endpoint_id)
            if not current:
                logger.warning(f"No progress record found for {endpoint_id}")
                return
            
            # Prepare updates
            updates = {
                'status': 'running',
                'total_records_fetched': current.get('total_records_fetched', 0) + records_count,
                'total_api_calls': current.get('total_api_calls', 0) + api_calls,
                'updated_at': datetime.utcnow().isoformat()
            }
            
            if last_time:
                updates['last_successful_time'] = last_time.isoformat()
            
            # Update record
            self.client.table(self.table).update(updates).eq(
                'endpoint_id', endpoint_id
            ).execute()
            
            logger.debug(
                f"Updated progress for {endpoint_id}: "
                f"+{records_count} records, +{api_calls} API calls"
            )
            
        except Exception as e:
            logger.error(f"Failed to update progress for {endpoint_id}: {e}")
            # Don't raise - progress tracking failure shouldn't stop backfill

    def reopen_progress(self, endpoint_id: str, config: Dict) -> None:
        """Reopen a completed endpoint for a later requested range.

        Preserve its checkpoint and counters so the orchestrator resumes
        incrementally instead of replaying the full history.
        """
        updates = {
            'status': 'pending',
            'completed_at': None,
            'error_message': None,
            'config': _to_json_safe(config),
            'updated_at': datetime.utcnow().isoformat(),
        }
        self.client.table(self.table).update(updates).eq(
            'endpoint_id', endpoint_id
        ).execute()
        logger.info(f"Reopened progress tracking for {endpoint_id}")
    
    def mark_completed(self, endpoint_id: str) -> None:
        """
        Mark endpoint backfill as completed.
        
        Args:
            endpoint_id: Unique endpoint identifier
        """
        try:
            updates = {
                'status': 'completed',
                'completed_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat()
            }
            
            self.client.table(self.table).update(updates).eq(
                'endpoint_id', endpoint_id
            ).execute()
            
            logger.info(f"Marked {endpoint_id} as completed")
            
        except Exception as e:
            logger.error(f"Failed to mark {endpoint_id} as completed: {e}")
            raise
    
    def mark_failed(self, endpoint_id: str, error_message: str) -> None:
        """
        Mark endpoint backfill as failed.
        
        Args:
            endpoint_id: Unique endpoint identifier
            error_message: Error description
        """
        try:
            current = self.get_progress(endpoint_id)
            retry_count = current.get('retry_count', 0) if current else 0
            
            updates = {
                'status': 'failed',
                'error_message': error_message,
                'retry_count': retry_count + 1,
                'updated_at': datetime.utcnow().isoformat()
            }
            
            self.client.table(self.table).update(updates).eq(
                'endpoint_id', endpoint_id
            ).execute()
            
            logger.error(f"Marked {endpoint_id} as failed: {error_message}")
            
        except Exception as e:
            logger.error(f"Failed to mark {endpoint_id} as failed: {e}")
    
    def get_failed_endpoints(self) -> list:
        """
        Get all failed endpoint records.
        
        Returns:
            List of failed progress records
        """
        try:
            result = self.client.table(self.table).select('*').eq(
                'status', 'failed'
            ).execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            logger.error(f"Failed to get failed endpoints: {e}")
            return []
    
    def reset_progress(self, endpoint_id: str) -> None:
        """
        Reset progress for an endpoint (for retries).
        
        Args:
            endpoint_id: Unique endpoint identifier
        """
        try:
            updates = {
                'status': 'pending',
                'last_successful_time': None,
                'error_message': None,
                'updated_at': datetime.utcnow().isoformat()
            }
            
            self.client.table(self.table).update(updates).eq(
                'endpoint_id', endpoint_id
            ).execute()
            
            logger.info(f"Reset progress for {endpoint_id}")
            
        except Exception as e:
            logger.error(f"Failed to reset progress for {endpoint_id}: {e}")
            raise

