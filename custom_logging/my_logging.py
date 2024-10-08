import logging
import sys

logger = logging.getLogger("default")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
formatter = logging.Formatter(FORMAT)
ch.setFormatter(formatter)
logger.addHandler(ch)

logger.debug("I am sent to standard out.")