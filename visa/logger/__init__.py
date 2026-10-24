import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

try:
    from from_root import from_root
    log_dir_path = os.path.join(from_root(), "logs")
except Exception:
    log_dir_path = os.path.join(os.getcwd(), "logs")

logs_path = os.path.join(log_dir_path, LOG_FILE)
os.makedirs(log_dir_path, exist_ok=True)

logging.basicConfig(
    filename=logs_path,
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)