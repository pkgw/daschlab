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

# If you use the AWS development API, `api.dev.starglass.cfa.harvard.edu`,
# remember that it can only be accessed from a limited range of IP addresses.
# You'll need to be at the CfA, or logged onto its VPN.

_URL_ROOT = "https://api.starglass.cfa.harvard.edu"


class ApiClient:
    base_url: str
    local_program: Optional[str] = None
    api_key: Optional[str] = None
    debug: bool = False

    def __init__(self):
        api_key = os.environ.get("DASCHLAB_API_KEY")
        if api_key:
            self.api_key = api_key

        self.base_url = os.environ.get("DASCHLAB_API_BASE_URL")

        if not self.base_url:
            definition = "full" if api_key else "public"
            self.base_url = f"{_URL_ROOT}/{definition}"

        self.local_program = None
        self.debug = False

        # You can set this environment variable to the path to the "oneshot"
        # version of the dasch-science-lambda tool to test APIs using a locally
        # modified codebase. This is only possible if you have the authorization
        # to directly access the underlying DASCH AWS resources, though.
        program = os.environ.get("DASCHLAB_API_PROGRAM")
        if program:
            self.local_program = program
            print(
                f"NOTE: using local program `{program}` for main API calls",
                file=sys.stderr,
            )

    def invoke(
        self, endpoint: str, payload: Optional[dict], method: str = "post"
    ) -> object:
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
            print(f"daschlab API: {method} {endpoint} : {payload!r}", file=sys.stderr)

        if self.local_program is not None and endpoint.startswith("/dasch/dr7/"):
            return self._invoke_program(endpoint, payload)
        else:
            return self._invoke_web(endpoint, payload, method)

    def _invoke_web(
        self, endpoint: str, payload: Optional[dict], method: str
    ) -> object:
        url = f"{self.base_url}{endpoint}"
        headers = {
            "User-Agent": "daschlab",
            "Accept": "application/json",
        }

        if self.api_key:
            headers["x-api-key"] = self.api_key

        try:
            with requests.request(
                method, url, json=payload, headers=headers, allow_redirects=False
            ) as resp:
                return resp.json()
        except Exception as e:
            # TODO: this might be a timeout, which we should handle better
            terse_payload = json.dumps(
                payload, ensure_ascii=False, indent=None, separators=(",", ":")
            )
            raise Exception(
                f"error invoking web API {method} {url} with payload {terse_payload}"
            ) from e

    def _invoke_program(self, endpoint: str, payload: dict) -> object:
        assert endpoint.startswith("/dasch/dr7/")
        short_endpoint = endpoint[11:]

        pltext = json.dumps(payload)
        argv = [self.local_program, short_endpoint, pltext]
        result = subprocess.check_output(argv, shell=False, text=True)
        return json.loads(result)
