import logging
import warnings

from tqdm import tqdm


class TqdmStream:
    def write(self, msg):
        tqdm.write(msg, end='')


def setup(file=None, console=True, capture_warnings=True):
    console_handler = logging.StreamHandler()
    console_handler.setStream(TqdmStream())

    handlers = []

    if console:
        handlers += [console_handler]

    if file is not None:
        handlers += [logging.FileHandler(file)]

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d [%(levelname)-8s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=handlers,
    )

    if capture_warnings:
        logging.captureWarnings(True)
        warnings.filterwarnings('default')


class Logger:
    # This class essentially emulates logger.getLogger(pfx) without leaking
    # every logger if the prefix changes. Useful if that prefix changes a lot,
    # e.g. when logging stages and epochs.

    base: logging.Logger

    def __init__(self, pfx=''):
        self.pfx = pfx

    def new(self, pfx, sep=':', indent=0):
        if self.pfx:
            pfx = f"{self.pfx}{sep}{pfx}"

        if indent:
            pfx = ' ' * indent + pfx

        return Logger(pfx)

    def debug(self, msg, *args, **kwargs):
        logging.debug(f"{self.pfx}: {msg}" if self.pfx else msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        logging.info(f"{self.pfx}: {msg}" if self.pfx else msg, *args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        logging.warn(f"{self.pfx}: {msg}" if self.pfx else msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        logging.error(f"{self.pfx}: {msg}" if self.pfx else msg, *args, **kwargs)
