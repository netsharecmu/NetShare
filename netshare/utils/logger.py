import inspect
import io
import logging
import sys

logger: logging.Logger = logging.getLogger("netshare")

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class TqdmToLogger(io.StringIO):
    """
    Util to output tqdm progress bar to the logger.
    """

    def __init__(self, description: str) -> None:
        super().__init__()
        self.description = description

    def write(self, buf: str) -> int:
        if buf.strip():
            logger.debug(f"{self.description}: {buf.strip()}")
        return len(buf)
