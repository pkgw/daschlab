# Copyright 2024 the President and Fellows of Harvard College
# Licensed under the MIT License

"""
The main ``daschlab`` package.
"""

import pathlib
from typing import Iterable

__all__ = [
    "Session",
    "open_session",
]


class Session:
    _root: pathlib.Path
    _internal: bool = False

    def __init__(self, root: str, _internal: bool = False):
        self._root = pathlib.Path(root)
        self._internal = _internal

    def path(self, *pieces: Iterable[str]) -> pathlib.Path:
        return self._root.joinpath(*pieces)


def open_session(root: str = ".", _internal: bool = False) -> Session:
    """
    Open or create a new daschlab analysis session.
    """
    return Session(root, _internal=_internal)
