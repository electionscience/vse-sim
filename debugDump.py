import logging

TRACE = 5
logging.addLevelName(TRACE, "TRACE")

logger = logging.getLogger("vse_sim")
logger.setLevel(TRACE)


def trace(*args):
    """Log low-level diagnostic values at the TRACE level."""
    logger.log(TRACE, " ".join(str(arg) for arg in args))


debug = trace


def setDebug(state):
    """Backward-compatible switch for trace diagnostics."""
    logger.setLevel(TRACE if state else logging.CRITICAL + 1)
