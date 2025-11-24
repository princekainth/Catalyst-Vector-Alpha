# logging_config.py - Enhanced logging for CVA
import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

LOG_DIR = "logs"

def setup_logging():
    """Setup enhanced logging with per-agent files and rotation."""
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Main catalyst logger
    main_handler = RotatingFileHandler(
        f"{LOG_DIR}/catalyst.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    main_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    ))
    
    catalyst_logger = logging.getLogger("CatalystLogger")
    catalyst_logger.setLevel(logging.INFO)
    catalyst_logger.addHandler(main_handler)
    
    # Agent-specific loggers
    agents = ["Planner", "Worker", "Observer", "Security", "Notifier"]
    for agent in agents:
        handler = RotatingFileHandler(
            f"{LOG_DIR}/agent_{agent.lower()}.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s]: %(message)s'
        ))
        logger = logging.getLogger(f"Agent_{agent}")
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
    
    # Error-only log
    error_handler = RotatingFileHandler(
        f"{LOG_DIR}/errors.log",
        maxBytes=5*1024*1024,
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    ))
    catalyst_logger.addHandler(error_handler)
    
    return catalyst_logger

def get_agent_logger(agent_name: str) -> logging.Logger:
    """Get logger for specific agent."""
    for key in ["Planner", "Worker", "Observer", "Security", "Notifier"]:
        if key.lower() in agent_name.lower():
            return logging.getLogger(f"Agent_{key}")
    return logging.getLogger("CatalystLogger")
