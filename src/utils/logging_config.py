"""Logging configuration for all modules."""

import logging
import logging.config
from typing import Optional


def setup_logging(config: Optional[dict] = None) -> None:
    """Configure logging from config dict or defaults."""
    if config is None:
        config = {
            'level': 'INFO',
            'format': '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            'date_format': '%Y-%m-%d %H:%M:%S'
        }

    log_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': config.get('format'),
                'datefmt': config.get('date_format')
            }
        },
        'handlers': {
            'default': {
                'level': config.get('level'),
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            }
        },
        'loggers': {
            '': {
                'handlers': ['default'],
                'level': config.get('level'),
                'propagate': True
            }
        }
    }

    logging.config.dictConfig(log_config)
    logging.getLogger(__name__).info(f"Logging configured at level {config.get('level')}")
