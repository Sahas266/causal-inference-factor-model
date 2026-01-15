#!/usr/bin/env python3
"""
Multi-Provider Data Backfill System - CLI Interface

Usage:
    python backfill.py --config config/endpoints/btc_metrics.json
    python backfill.py --config config/endpoints/ --max-workers 5
    python backfill.py --validate-only
    python backfill.py --list-providers
    python backfill.py --resume-failed
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.orchestrator import BackfillOrchestrator
from src.core.utils.logger import setup_logger
from src.core.utils.config_loader import ConfigLoader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Multi-Provider Data Backfill System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backfill a single endpoint
  python backfill.py --config config/endpoints/btc_metrics.json
  
  # Backfill all endpoints in a directory
  python backfill.py --config config/endpoints/
  
  # Validate endpoints without backfilling
  python backfill.py --validate-only
  
  # List available providers
  python backfill.py --list-providers
  
  # Resume failed backfills
  python backfill.py --resume-failed
  
  # Show provider statistics
  python backfill.py --stats
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to endpoint config file or directory'
    )
    
    parser.add_argument(
        '--config-dir',
        type=str,
        default='config',
        help='Path to config directory (default: config/)'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=10,
        help='Maximum number of concurrent workers (default: 10)'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate endpoints without backfilling'
    )
    
    parser.add_argument(
        '--list-providers',
        action='store_true',
        help='List available data providers'
    )
    
    parser.add_argument(
        '--resume-failed',
        action='store_true',
        help='Resume failed backfill operations'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show provider statistics'
    )
    
    parser.add_argument(
        '--provider',
        type=str,
        help='Filter by specific provider'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='Log file path'
    )
    
    parser.add_argument(
        '--json-logs',
        action='store_true',
        help='Use JSON formatted logs'
    )
    
    return parser.parse_args()


def load_endpoint_configs(config_path: str, config_loader: ConfigLoader) -> List[dict]:
    """
    Load endpoint configurations from file or directory.
    
    Args:
        config_path: Path to config file or directory
        config_loader: ConfigLoader instance
        
    Returns:
        List of endpoint configurations
    """
    path = Path(config_path)
    
    if path.is_file():
        # Single config file
        return [config_loader.load_endpoint_config(str(path))]
    elif path.is_dir():
        # Directory of config files
        configs = []
        for config_file in path.glob('*.json'):
            try:
                config = config_loader.load_endpoint_config(str(config_file))
                configs.append(config)
            except Exception as e:
                logger.error(f"Failed to load {config_file}: {e}")
        return configs
    else:
        raise FileNotFoundError(f"Config path not found: {config_path}")


def print_results(results: dict):
    """
    Print backfill results in a formatted way.
    
    Args:
        results: Results dictionary from orchestrator
    """
    print("\n" + "="*80)
    print("BACKFILL RESULTS")
    print("="*80)
    
    print(f"\nTotal Endpoints: {results['total_endpoints']}")
    print(f"Completed: {results['completed']}")
    print(f"Failed: {results['failed']}")
    print(f"Skipped: {results['skipped']}")
    
    if results['provider_stats']:
        print("\nPer-Provider Statistics:")
        print("-" * 80)
        print(f"{'Provider':<20} {'Completed':<12} {'Failed':<12} {'Records':<15}")
        print("-" * 80)
        
        for provider, stats in results['provider_stats'].items():
            print(
                f"{provider:<20} "
                f"{stats['completed']:<12} "
                f"{stats['failed']:<12} "
                f"{stats['records']:<15,}"
            )
    
    print("="*80 + "\n")


def main():
    """Main CLI entry point"""
    args = parse_args()
    
    # Setup logging
    global logger
    logger = setup_logger(
        level=args.log_level,
        json_format=args.json_logs,
        log_file=args.log_file
    )
    
    try:
        # Initialize orchestrator
        logger.info("Initializing backfill orchestrator...")
        orchestrator = BackfillOrchestrator(config_dir=args.config_dir)
        
        # Handle different commands
        if args.list_providers:
            # List available providers
            providers = orchestrator.list_providers()
            print("\nAvailable Providers:")
            print("-" * 40)
            for provider in providers:
                print(f"  • {provider}")
            print()
            return 0
        
        if args.stats:
            # Show provider statistics
            print("\nProvider Statistics:")
            print("=" * 80)
            stats = orchestrator.get_provider_stats()
            print(json.dumps(stats, indent=2, default=str))
            print()
            return 0
        
        if args.resume_failed:
            # Resume failed backfills
            logger.info("Resuming failed backfills...")
            failed_endpoints = orchestrator.progress_tracker.get_failed_endpoints()
            
            if not failed_endpoints:
                print("No failed backfills to resume.")
                return 0
            
            print(f"\nFound {len(failed_endpoints)} failed backfills")
            
            for endpoint in failed_endpoints:
                logger.info(f"Resetting progress for {endpoint['endpoint_id']}")
                orchestrator.progress_tracker.reset_progress(endpoint['endpoint_id'])
            
            print("Failed backfills reset. Re-run without --resume-failed to execute.")
            return 0
        
        # Load endpoint configurations
        if not args.config:
            logger.error("No config specified. Use --config or --list-providers")
            return 1
        
        logger.info(f"Loading endpoint configurations from: {args.config}")
        config_loader = ConfigLoader(args.config_dir)
        endpoint_configs = load_endpoint_configs(args.config, config_loader)
        
        if not endpoint_configs:
            logger.error("No endpoint configurations loaded")
            return 1
        
        logger.info(f"Loaded {len(endpoint_configs)} endpoint configuration(s)")
        
        # Run backfill
        logger.info("Starting backfill operation...")
        start_time = datetime.now()
        
        results = orchestrator.run_backfill(
            endpoint_configs=endpoint_configs,
            max_workers=args.max_workers,
            validate_only=args.validate_only
        )
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Print results
        print_results(results)
        print(f"Duration: {duration}")
        
        # Exit code based on results
        if results['failed'] > 0:
            logger.warning(f"{results['failed']} endpoint(s) failed")
            return 1
        else:
            logger.info("All endpoints completed successfully")
            return 0
        
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())

