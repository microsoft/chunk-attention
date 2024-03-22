import logging
import sys

logger = logging.getLogger("ChunkAttn")
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "[%(asctime)s] [%(name)s] [%(levelname)s] [%(process)d:%(thread)d] %(message)s",
)
handler.setFormatter(formatter)
logger.addHandler(handler)