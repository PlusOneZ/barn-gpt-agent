import logging
import sys
from datetime import datetime
from .my_logging import logger

time = datetime.now().isoformat()
logging.basicConfig(filename=f'logs/{time}.log', encoding='utf-8', level=logging.DEBUG)

