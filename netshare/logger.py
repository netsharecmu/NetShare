import logging
import os
import sys

logger: logging.Logger = logging.getLogger("netshare")

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

if os.environ.get("NETSHARE_DEBUG", "").lower() == "true":
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)
