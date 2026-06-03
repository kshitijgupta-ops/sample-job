import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)-8s %(message)s")


def get_logger(name=__name__):
    return logging.getLogger(name)
