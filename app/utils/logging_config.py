import os
import logging
import sys
from logging.handlers import RotatingFileHandler

def configure_logging(log_level=None, log_file=None):
    """Configure logging for the application
    
    Args:
        log_level: The logging level (default: INFO)
        log_file: Path to the log file (default: logs/app.log)
    """
    if log_level is None:
        log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
    
    if log_file is None:
        log_dir = os.environ.get('LOG_DIR', 'logs')
        log_file = os.path.join(log_dir, 'app.log')
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:  
        root_logger.removeHandler(handler)
    
    # Create formatters
    verbose_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # Create file handler
    try:
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=10485760,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(verbose_formatter)
        root_logger.addHandler(file_handler)
    except (IOError, PermissionError) as e:
        logging.warning(f"Could not create log file at {log_file}: {str(e)}")
    
    # Set level for specific loggers
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    logging.info(f"Logging configured with level {log_level}")
    return root_logger