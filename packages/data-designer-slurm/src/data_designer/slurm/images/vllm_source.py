# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve versioned public vLLM images before the immutable OCI import path."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

_VLLM_REPOSITORY = "vllm/vllm-openai"
_VERSION = r"v[0-9]+\.[0-9]+\.[0-9]+"
_VERSIONED_TAG = re.compile(rf"^(?:docker\.io/)?{_VLLM_REPOSITORY}:(?P<tag>{_VERSION})$")
_PINNED_VERSION = re.compile(rf"^(?:docker\.io/)?{_VLLM_REPOSITORY}:(?P<tag>{_VERSION})@sha256:[0-9a-f]{{64}}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_TOKEN_URL = "https://auth.docker.io/token"
_MANIFEST_URL = f"https://registry-1.docker.io/v2/{_VLLM_REPOSITORY}/manifests/"
_MANIFEST_ACCEPT = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)
_MAX_TOKEN_RESPONSE_BYTES = 65536
_REQUEST_TIMEOUT_SECONDS = 10


class VllmSourceResolutionError(Exception):
    """A versioned vLLM source could not be pinned to an OCI digest."""


class VllmTagNotFoundError(VllmSourceResolutionError):
    """The requested official vLLM release tag does not exist."""


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(
        self,
        request: Request,
        response: object,
        code: int,
        message: str,
        headers: object,
        new_url: str,
    ) -> None:
        return None


def is_versioned_vllm_tag(source: str) -> bool:
    """Return whether a source is an unpinned official vLLM version tag."""
    return _VERSIONED_TAG.fullmatch(source) is not None


def pinned_vllm_version(source: str) -> str | None:
    """Extract the release version recorded alongside an immutable OCI digest."""
    match = _PINNED_VERSION.fullmatch(source)
    return None if match is None else match.group("tag").removeprefix("v")


def resolve_versioned_vllm_source(source: str) -> str:
    """Resolve one official vLLM version tag and return a digest-pinned source.

    Only fixed Docker Hub hosts and the official vLLM repository are contacted.
    Bearer credentials never enter persisted plans, output, or exception text.
    """
    match = _VERSIONED_TAG.fullmatch(source)
    if match is None:
        raise ValueError("source is not an official versioned vLLM image")
    tag = match.group("tag")
    try:
        query = urlencode({"service": "registry.docker.io", "scope": f"repository:{_VLLM_REPOSITORY}:pull"})
        token_body, _ = _request(Request(f"{_TOKEN_URL}?{query}", method="GET"))
        token_payload = json.loads(token_body)
        token = token_payload.get("token") if isinstance(token_payload, dict) else None
        if not isinstance(token, str) or not token or len(token) > 16384:
            raise VllmSourceResolutionError("Docker Hub did not return a valid vLLM pull token")
        try:
            _, headers = _request(
                Request(
                    f"{_MANIFEST_URL}{tag}",
                    headers={"Authorization": f"Bearer {token}", "Accept": _MANIFEST_ACCEPT},
                    method="HEAD",
                )
            )
        except HTTPError as error:
            if error.code == 404:
                raise VllmTagNotFoundError("the requested vLLM image version was not found") from None
            raise
        digest = headers.get("Docker-Content-Digest", "")
        if _SHA256.fullmatch(digest) is None:
            raise VllmSourceResolutionError("Docker Hub did not return a valid vLLM manifest digest")
    except HTTPError:
        raise VllmSourceResolutionError("the vLLM image version could not be resolved") from None
    except (URLError, TimeoutError, OSError, HTTPException, UnicodeError, json.JSONDecodeError):
        raise VllmSourceResolutionError("the vLLM image version could not be resolved") from None
    return f"{_VLLM_REPOSITORY}:{tag}@{digest}"


def _request(request: Request) -> tuple[bytes, Mapping[str, str]]:
    opener = build_opener(_RejectRedirects)
    with opener.open(request, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
        body = response.read(_MAX_TOKEN_RESPONSE_BYTES + 1)
        if len(body) > _MAX_TOKEN_RESPONSE_BYTES:
            raise VllmSourceResolutionError("Docker Hub returned an oversized response")
        return body, response.headers
