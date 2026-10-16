# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from http.client import IncompleteRead
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

import data_designer.slurm.images.vllm_source as vllm_source


def test_resolves_official_version_to_immutable_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[Request] = []

    def request(resource: Request) -> tuple[bytes, dict[str, str]]:
        requests.append(resource)
        if len(requests) == 1:
            return b'{"token":"private-token"}', {}
        return b"", {"Docker-Content-Digest": f"sha256:{'a' * 64}"}

    monkeypatch.setattr(vllm_source, "_request", request)

    pinned = vllm_source.resolve_versioned_vllm_source("vllm/vllm-openai:v0.22.0")

    assert pinned == f"vllm/vllm-openai:v0.22.0@sha256:{'a' * 64}"
    assert requests[0].full_url.startswith("https://auth.docker.io/token?")
    assert requests[1].full_url == "https://registry-1.docker.io/v2/vllm/vllm-openai/manifests/v0.22.0"
    assert requests[1].get_method() == "HEAD"
    assert requests[1].get_header("Authorization") == "Bearer private-token"
    assert vllm_source.pinned_vllm_version(pinned) == "0.22.0"


@pytest.mark.parametrize(
    "source",
    (
        "vllm/vllm-openai:latest",
        "other/vllm-openai:v0.22.0",
        "vllm/vllm-openai:v0.22.0-rc1",
        "https://vllm/vllm-openai:v0.22.0",
    ),
)
def test_rejects_non_release_sources(source: str) -> None:
    assert not vllm_source.is_versioned_vllm_tag(source)
    with pytest.raises(ValueError, match="official versioned vLLM"):
        vllm_source.resolve_versioned_vllm_source(source)


@pytest.mark.parametrize(
    "failure",
    (
        URLError("private-token"),
        HTTPError("url", 500, "private-token", {}, None),
        IncompleteRead(b"private-token", 100),
    ),
)
def test_resolution_failure_is_sanitized(monkeypatch: pytest.MonkeyPatch, failure: Exception) -> None:
    def request(_resource: Request) -> tuple[bytes, dict[str, str]]:
        raise failure

    monkeypatch.setattr(vllm_source, "_request", request)
    with pytest.raises(vllm_source.VllmSourceResolutionError) as caught:
        vllm_source.resolve_versioned_vllm_source("vllm/vllm-openai:v0.22.0")
    assert "private-token" not in str(caught.value)


def test_resolution_rejects_missing_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(((b'{"token":"token"}', {}), (b"", {"Docker-Content-Digest": "not-a-digest"})))
    monkeypatch.setattr(vllm_source, "_request", lambda _resource: next(responses))

    with pytest.raises(vllm_source.VllmSourceResolutionError, match="manifest digest"):
        vllm_source.resolve_versioned_vllm_source("vllm/vllm-openai:v0.22.0")


def test_missing_version_has_distinct_sanitized_error(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(((b'{"token":"token"}', {}), HTTPError("url", 404, "private-token", {}, None)))

    def request(_resource: Request) -> tuple[bytes, dict[str, str]]:
        result = next(responses)
        if isinstance(result, HTTPError):
            raise result
        return result

    monkeypatch.setattr(vllm_source, "_request", request)
    with pytest.raises(vllm_source.VllmTagNotFoundError, match="not found") as caught:
        vllm_source.resolve_versioned_vllm_source("vllm/vllm-openai:v0.22.0")
    assert "private-token" not in str(caught.value)


def test_token_endpoint_not_found_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def request(_resource: Request) -> tuple[bytes, dict[str, str]]:
        raise HTTPError("url", 404, "private-token", {}, None)

    monkeypatch.setattr(vllm_source, "_request", request)
    with pytest.raises(vllm_source.VllmSourceResolutionError, match="could not be resolved") as caught:
        vllm_source.resolve_versioned_vllm_source("vllm/vllm-openai:v0.22.0")
    assert not isinstance(caught.value, vllm_source.VllmTagNotFoundError)
