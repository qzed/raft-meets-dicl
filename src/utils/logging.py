import io
import logging
import re
import sys
import warnings

from tqdm import tqdm


class TqdmStream:
    def write(self, msg):
        tqdm.write(msg, end='')


class TqdmLogWrapper(io.StringIO):
    def __init__(self, logger, level=logging.INFO):
        super().__init__()

        self.logger = logger
        self.level = level
        self.buf = ''

        self.re_ansi_esc = re.compile(r'''(?:\x1B[@-Z\\-_])''', re.VERBOSE)

    def write(self, buf):
        self.buf += self.re_ansi_esc.sub('', buf).strip('\r\n\t ')

    def flush(self):
        if self.buf:
            self.logger.log(self.level, self.buf)
            self.buf = ''


def setup(file=None, console=True, capture_warnings=True, tqdm_to_log=not sys.stderr.isatty()):
    console_handler = logging.StreamHandler()

    # If we output tqdm progress to stderr, add handler for logging. In the
    # other case we expect tqdm to be redirected to the logger.
    if not tqdm_to_log:
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


def progress(data, *args, to_log=not sys.stderr.isatty(), update_pct_log=5, logger=None, **kwargs):
    if not to_log:
        return tqdm(data, *args, **kwargs)

    else:
        # limit number of updates to not flood log
        miniters = int(len(data) / 100 * update_pct_log)
        mininterval = 15        # update at most once in 15s
        maxinterval = 900       # update at least once in 15m

        # use logger to write tqdm output
        tqdm_out = TqdmLogWrapper(logger if logger is not None else Logger())

        return tqdm(data, *args, **kwargs, miniters=miniters, mininterval=mininterval,
                    maxinterval=maxinterval, file=tqdm_out)


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
        logging.warning(f"{self.pfx}: {msg}" if self.pfx else msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        logging.error(f"{self.pfx}: {msg}" if self.pfx else msg, *args, **kwargs)

    def log(self, level: int, msg, *args, **kwargs):
        logging.log(level, f"{self.pfx}: {msg}" if self.pfx else msg, *args, **kwargs)
