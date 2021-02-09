# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import atexit
import builtins
import decimal
import functools
import logging
import os
import sys

from .comm import is_master_process as is_master_proc

def _suppress_print():
    """
    Suppresses printing from the current process.
    """
    def print_pass(*objects, sep=" ", end="\n", file=sys.stdout, flush=False):
        pass

    builtins.print = print_pass


def setup_logging(save_path, mode='a'):
    """
    Sets up the logging for multiple processes. Only enable the logging for the
    master process, and suppress logging for the non-master processes.
    """
    if is_master_proc():
        # Enable logging for the master process.
        logging.root.handlers = []
    else:
        # Suppress logging for non-master processes.
        _suppress_print()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    print_plain_formatter = logging.Formatter(
        "[%(asctime)s]: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )
    fh_plain_formatter = logging.Formatter("%(message)s")

    if is_master_proc():
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(print_plain_formatter)
        logger.addHandler(ch)

    if save_path is not None and is_master_proc():
        fh = logging.FileHandler(save_path, mode=mode)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fh_plain_formatter)
        logger.addHandler(fh)


def get_logger(name):
    """
    Retrieve the logger with the specified name or, if name is None, return a
    logger which is the root logger of the hierarchy.
    Args:
        name (string): name of the logger.
    """
    return logging.getLogger(name)



