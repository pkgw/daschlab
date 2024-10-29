# Copyright 2024 the President and Fellows of Harvard College
# Licensed under the MIT License

"""
Centralized code for communicating with the DASCH data APIs.

This should move into astroquery at some point.
"""

import json
import os
import subprocess
import sys
from typing import Optional

import requests


__all__ = ["ApiClient"]


# TODO: genericize!!
_BASE_URL = "https://api.dev.starglass.cfa.harvard.edu/public/dasch/dr7"


class ApiClient:
    base_url: str
    local_program: Optional[str]
    debug: bool

    def __init__(self):
        self.base_url = _BASE_URL
        self.local_program = None
        self.debug = False

        # You can set this environment variable to the path to the "oneshot"
        # version of the dasch-science-lambda tool to test APIs using a locally
        # modified codebase. This is only possible if you have the authorization
        # to directly access the underlying DASCH AWS resources, though.
        program = os.environ.get("DASCHLAB_API_PROGRAM")
        if program:
            self.base_url = ""
            self.local_program = program
            print(
                f"NOTE: using local program `{program}` for API calls", file=sys.stderr
            )

    def invoke(self, endpoint: str, payload: dict) -> object:
        """
        Directly invoke a DASCH data API endpoint.

        Due to the way that DASCH's APIs are implemented on the AWS Lambda
        system, the responses from all APIs must be in the form of JSON. This
        has the unfortunate consequence that it becomes quite inconvenient to
        try to process any API output in streaming fashion. Instead, we're more
        or less forced to buffer everything up in memory. The maximum response
        size from a buffered AWS Lambda is 6 MB, which isn't too bad, but it's
        still unfortunate.
        """

        if self.debug:
            print(f"daschlab API: {endpoint} : {payload!r}", file=sys.stderr)

        if self.local_program is not None:
            return self._invoke_program(endpoint, payload)
        else:
            return self._invoke_web(endpoint, payload)

    def _invoke_web(self, endpoint: str, payload: dict) -> object:
        url = f"{self.base_url}/{endpoint}"

        with requests.post(url, json=payload) as resp:
            return resp.json()

    def _invoke_program(self, endpoint: str, payload: dict) -> object:
        pltext = json.dumps(payload)
        argv = [self.local_program, endpoint, pltext]
        result = subprocess.check_output(argv, shell=False, text=True)
        return json.loads(result)