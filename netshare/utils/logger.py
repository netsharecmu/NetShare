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
logger.setLevel(logging.DEBUG)


class TqdmToLogger(io.StringIO):
    """
    Util to output tqdm progress bar to the logger.
    """

    def write(self, buf: str) -> int:
        if buf.strip():
            last_function_name = None
            for stack in inspect.stack()[1:]:
                if "site-packages/tqdm" not in stack.filename:
                    last_function_name = stack.function
                    break
            logger.debug(f"{last_function_name}: {buf.strip()}")
        return len(buf)
