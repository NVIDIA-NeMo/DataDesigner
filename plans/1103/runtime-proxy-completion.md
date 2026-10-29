# Plan: Complete the Slurm inference proxy lifecycle

## Scope and delivered behavior

Issue #1103 carries forward the outstanding lifecycle, telemetry, and overload
acceptance requirements from #966 after the core proxy implementation in #967.

- The coordinated allocation runtime starts a CPU-only health monitor after
  startup readiness. It probes each logical endpoint's aggregate health, writes
  revisioned backend counts and probe evidence, and fails the attempt when a
  required service loses health while its process remains alive. Finalization
  preserves that evidence. See `runtime/health.py:28` and `runtime/entrypoint.py:156`
  under `packages/data-designer-slurm/src/data_designer/slurm/`.
- Health probes use their own bounded pool, independent of inference admission,
  so saturated inference connections cannot starve health checks or distort
  least-connections selection. See `runtime/proxy.py`.
- Cleanup stops monitoring, sends `TERM` to each exact endpoint task step with
  `scancel`, and allows 12 seconds for their 10-second drain window and step exit.
  The task publishes its job and step IDs before executing the proxy; cleanup
  validates that the identity belongs to the current allocation. Launchers and
  required serving steps stay available until endpoint termination. Remaining steps then stop with bounded
  TERM/KILL escalation. See `runtime/cleanup.sh:48`.
- Request events contain per-attempt origins, connection/reuse timing, response
  header and total durations, outcomes, and completion state. Counters include
  expired/stale/discarded connections and terminal/streaming timeouts. Payloads,
  request paths, and authorization headers are excluded. Eviction accounting
  uses narrow aiohttp 3.x connector hooks because no public eviction signal exists.

## Automated acceptance

The final complete Slurm suite passed with 2,123 tests and one platform skip. The
focused health, proxy, and request-admission suite passed all 54 cases.

Coverage includes real-process cleanup on completion, cancellation, and failure;
forced termination after a deadline; health loss without process exit and
retention of failure evidence; health checks under connection saturation; expired,
stale, and discarded connections; streaming timeout without replay; and draining
an active stream with production runner settings.

An eight-replica loopback test drives the actual HTTP adapter and request-admission
controller. It verifies the three-attempt ceiling and end-to-end 429/cooldown
handling separately from the two-replica GPU acceptance below.

## Live overload acceptance — 2026-10-28

The repeatable harness is `scripts/validate_slurm_proxy_overload.py`. The complete
sanitized report and immutable artifact identities are in
[`live-overload-acceptance.json`](live-overload-acceptance.json).

The run installed four wheels built from commit
`f499303aaf7867f2ee50b21544406ebaa6fa4084` into a new Python 3.11 environment.
The third-party dependency inventory was pinned separately; `uv pip check`
verified compatibility of all 97 installed packages. The report records both
that inventory's digest and the source revision's `uv.lock` digest.
Two GPU-backed vLLM 0.22.0 processes served `openai/gpt-oss-20b` in one Slurm
allocation. The production queue middleware, proxy entrypoint, HTTP adapter, and
request-admission controller were used without synthesized overload responses.
The allocation completed with exit status zero and released its serving steps.

For this bounded test, each replica used `max-num-seqs=1`, the supported queue
threshold override `DD_VLLM_MAX_WAITING_REQUESTS=0`, and `Retry-After: 3`.
Eight long inference requests per overloaded replica created real waiting queues.
The harness required positive vLLM waiting-request metrics before testing the
proxy. The production default threshold remains 128.

| Case | Observed result |
| --- | --- |
| Healthy generation | 32 requested and 32 persisted records, all answers nonempty; 22.325 seconds, 86.003 records/minute |
| Healthy proxy traffic | 33 HTTP 200 responses including startup generation; zero failures, retries, or timeouts; 2 connections opened and 31 reused |
| Partial overload | Queue depth 7 on one replica; one local 429 retry, then success on the spare; admission remains 8 with no cooldown |
| Total overload | Queue depths 7 and 7; two upstream attempts, one local retry, final 429 and `Retry-After: 3` reach the real client |
| Admission and recovery | Admission decreases from 8 to 4; initial cooldown is 3.000 seconds; recovery succeeds after a 3.083-second request with 2.581 seconds of cooldown remaining at its start |
| Proxy resources | Healthy case: 0.26 CPU seconds and 38.340 MiB peak RSS; partial/total cases: 0.18/0.17 CPU seconds and approximately 37.65 MiB peak RSS |

Overload-case proxy metric snapshots precede the recovery request. Background load
requests go directly to the serving backends and are excluded from proxy counts.
The throughput number describes this small serial validation workload, not a
capacity benchmark. Raw scheduler, serving, proxy, and generated-data artifacts
remain private; the checked-in report excludes infrastructure names and paths,
credentials, prompts, and completions.

## Reproduction and acceptance boundaries

Launch two idle vLLM replicas using the package's
`data_designer.slurm.runtime.backpressure.QueueDepthBackpressureMiddleware` with
the queue settings above and `--max-num-seqs 1`. Within that allocation, invoke
the harness with two `--backend` origins, their served `--model` name, the exact
`--source-commit`, the four-wheel `--wheels` directory, and a private `--output`
path. The script exits nonzero on any failed assertion and writes a sanitized
summary only after every scenario passes.

The live run proves actual queue admission and overload propagation through
installed artifacts, including persisted Data Designer output. It does not run
the full managed `slurm execute` release matrix. Managed health/finalization and
shutdown paths have the automated process/state coverage described above.
The broader security, multi-node, retry, and final release-artifact matrix remains
owned by #870. The recorded live run applies to its exact source and wheel
identities. Subsequent review changes affect Python attempt finalization and shell
endpoint signal handling. The proxy, health monitor, backpressure middleware, and
overload acceptance harness remain byte-for-byte identical to the GPU-tested
source. Finalization changes have regression coverage; shell signal handling is
validated separately below.

## Review follow-up

Greptile identified a shutdown race: the monitor could persist a health failure
after the client's final process check, but success finalization would overwrite
it. Finalization now checks the monitor's saved evidence after cleanup has stopped
it and refuses to publish a winner after a recorded health failure, including
failure evidence retained in `STOPPED` readiness.

Independent review also found that failure cleanup attempted an invalid
`STOPPED` → `FAILED` readiness transition after unsuccessful winner finalization.
Failure cleanup now retains terminal readiness while marking the attempt failed.
Regression cases exercise both the late health failure and recovery after winner
publication fails, and the bundle test explicitly verifies that the health module
is staged.

Ten review passes covered success finalization, terminal-state recovery, health
protocol/state validation, shell monitoring and drain ordering, connector
accounting, streaming cancellation and telemetry, overload/admission behavior,
packaging/layering/secret scope, live-harness provenance, and the final diff and
validation record. Those passes preceded the additional Slurm signal-handling
finding described below.

## Slurm signal-handling review

Review identified that the cleanup PIDs belong to `srun` launchers. Slurm forwards
`TERM` sent to those launchers as `KILL` to tasks, bypassing the proxy drain
handler. Endpoint cleanup now uses the endpoint task's recorded numeric
`job.step` identity to send `TERM` directly with `scancel`. Launchers remain alive
during the drain window and are forcibly stopped only after that window expires.
Cleanup never sends a task-directed signal to a different allocation or to a
whole job. Stale step identity files are removed before launch.

The process regressions now separate the launcher from its endpoint task and
model Slurm's launcher `TERM` → task `KILL` behavior. They verify successful
draining before backend shutdown on completion, failure, and cancellation, plus
numeric/current-allocation identity checks and forced cleanup after the deadline.

### Live Slurm drain acceptance — 2026-10-29

The [sanitized report](live-drain-acceptance.json) records a separate CPU-only
Slurm allocation exercising the production Bash step-launch, endpoint-start, and
cleanup helpers, real `srun` and `scancel`, and the installed production proxy.
A synthetic HTTP backend streamed an initial chunk, then delayed its final chunk
for two seconds while cleanup began. The response completed on normal completion,
failure cleanup, and controller cancellation (statuses 0, 1, and 130). Each proxy
logged a completed response, released its listener marker, and finished its Slurm
task with exit status 0 before cleanup stopped the backend. The allocation
completed with exit status 0 and released all steps.

The report records hashes of every shell source used and the installed proxy
source; they match this PR's source. This validates the scheduler signal path
separately from GPU overload acceptance. It does not run the full managed
`slurm execute` state/bootstrap flow, and the synthetic backend is not vLLM.
