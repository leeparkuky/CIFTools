# custom_logger.py
import logging

# Configure Logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Create a logger instance
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
