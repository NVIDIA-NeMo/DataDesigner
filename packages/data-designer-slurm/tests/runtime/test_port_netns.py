# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux TCP namespaces")
def test_pre_listener_probe_cannot_consume_selected_service_port() -> None:
    if any(shutil.which(tool) is None for tool in ("unshare", "ip", "ss", "curl")):
        pytest.skip("requires unshare, iproute2, and curl")
    available = subprocess.run(
        ("unshare", "--user", "--map-root-user", "--net", "true"), capture_output=True, text=True
    )
    if available.returncode:
        pytest.skip("private user/network namespaces are unavailable")
    program = r"""
import errno
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from data_designer.slurm.runtime.network import get_service_ports
from data_designer.slurm.runtime.probes import HttpReadinessProber

assert str(Path("/proc/self/ns/net").readlink()) != os.environ["PARENT_NETNS"]
subprocess.run(("ip", "link", "set", "lo", "up"), check=True)
settings = Path("/proc/sys") / "net" / "ipv4"
settings.joinpath("ip_local_reserved_ports").write_text("\n")
settings.joinpath("ip_local_port_range").write_text("10000 10000\n")
settings.joinpath("ip_unprivileged_port_start").write_text("1024\n")

with socket.socket() as preflight:
    preflight.bind(("0.0.0.0", 10000))
probe = subprocess.run(
    ("curl", "--noproxy", "*", "--fail", "--silent", "--show-error", "--max-time", "1",
     "http://127.0.0.1:10000/health"), capture_output=True, text=True,
)
assert probe.returncode != 0
state = subprocess.run(("ss", "-tan", "( sport = :10000 )"), capture_output=True, text=True, check=True)
assert "TIME-WAIT" in state.stdout, (probe.stderr, state.stdout)
with socket.socket() as failed:
    failed.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        failed.bind(("127.0.0.1", 10000))
    except OSError as error:
        assert error.errno == errno.EADDRINUSE
    else:
        raise AssertionError("expected the published self-connection failure")

port = get_service_ports()[0]
assert port != 10000
for _ in range(3):
    assert not HttpReadinessProber().is_ready("127.0.0.1", port, "/health", timeout_seconds=0.1)
with socket.socket() as control:
    control.bind(("127.0.0.1", port))

with TemporaryDirectory() as directory:
    marker = Path(directory) / "endpoint.listening"
    command = (sys.executable, "-m", "data_designer.slurm.runtime.proxy", "--listen-port", str(port),
               "--backend", f"http://127.0.0.1:{port + 1}", "--listening-file", str(marker))
    proxy = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 10
        while not marker.exists():
            assert proxy.poll() is None, proxy.communicate()[1]
            assert time.monotonic() < deadline, "listener ownership timed out"
            time.sleep(0.01)
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            pass
        contender = subprocess.run(command[:-1] + (str(marker.with_name("contender")),),
                                   capture_output=True, text=True, timeout=10)
        assert contender.returncode != 0
        assert "Address already in use" in contender.stderr, contender.stderr
        assert not marker.with_name("contender").exists()
    finally:
        proxy.terminate()
        proxy.communicate(timeout=10)
    assert not marker.exists()
with socket.socket() as released:
    released.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    released.bind(("127.0.0.1", port))
"""
    completed = subprocess.run(
        ("unshare", "--user", "--map-root-user", "--net", sys.executable, "-c", program),
        env={**os.environ, "PARENT_NETNS": str(Path("/proc/self/ns/net").readlink())},
        capture_output=True,
        text=True,
        timeout=35,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
