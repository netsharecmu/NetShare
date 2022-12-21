import logging
import os

logger: logging.Logger = logging.getLogger("netshare")

if os.environ.get("NETSHARE_DEBUG", "").lower() == "true":
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)
