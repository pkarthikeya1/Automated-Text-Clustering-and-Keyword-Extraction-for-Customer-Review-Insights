import os
import sys
from pathlib import Path
import logging

log_dir = "logging"
file_name = "logging.log"
log_filepath = os.path.join(log_dir, file_name)

os.makedirs(log_dir, exist_ok=True)

LOGGING_STR = "[%(asctime)s, %(levelname)s, %(modulename)s, %(message)s ]"

logging.basicConfig(
    level= logging.INFO,
    format= LOGGING_STR,
    handlers= [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_filepath)
    ]
)
logger=logging.getLogger("New_Delhi_Reviews")