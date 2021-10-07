import logging


def get_console_logger(name, level=logging.WARNING):
    return logging.getLogger("pyrouge")


def get_global_console_logger(level=logging.WARNING):
    return logging.getLogger("pyrouge")
