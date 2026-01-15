"""Structured logging configuration for the backfill system"""

import logging
import sys
from typing import Optional
from datetime import datetime
import json


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields from LogRecord
        if hasattr(record, 'provider'):
            log_data['provider'] = record.provider
        if hasattr(record, 'endpoint_id'):
            log_data['endpoint_id'] = record.endpoint_id
        if hasattr(record, 'records_count'):
            log_data['records_count'] = record.records_count
        
        return json.dumps(log_data)


class ProviderLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds provider context to log messages"""
    
    def process(self, msg, kwargs):
        # Add provider context to extra fields
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        kwargs['extra'].update(self.extra)
        return msg, kwargs


def setup_logger(
    name: str = 'backfill_system',
    level: str = 'INFO',
    json_format: bool = True,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Setup and configure logger for the application.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Use JSON formatting for structured logs
        log_file: Optional file path for log output
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    if json_format:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
    
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        if json_format:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            )
        logger.addHandler(file_handler)
    
    return logger


def get_provider_logger(provider_name: str, base_logger: Optional[logging.Logger] = None) -> ProviderLoggerAdapter:
    """
    Get a logger adapter with provider context.
    
    Args:
        provider_name: Name of the provider
        base_logger: Base logger to wrap (uses default if None)
        
    Returns:
        Logger adapter with provider context
    """
    if base_logger is None:
        base_logger = logging.getLogger('backfill_system')
    
    return ProviderLoggerAdapter(base_logger, {'provider': provider_name})


# Default logger instance
logger = setup_logger()

