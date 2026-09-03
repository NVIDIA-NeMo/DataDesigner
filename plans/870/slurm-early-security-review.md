# Slurm early security and provenance review

This document records the dependency-ready portion of #870. It reviews the public Slurm implementation through
the one-node runtime merge and the hardening changes developed with this review. It is not final release acceptance:
the sealed commit, complete wheel set, runtime checksum, real-cluster scenarios, sanitized profile rerun, and all
dependent implementation and documentation must still be frozen and validated together.

## Threat model boundaries

| Boundary | Threats reviewed | Existing or added controls | Evidence |
| --- | --- | --- | --- |
| Slurm submission | Shell injection, option confusion, inherited secrets, unbounded caller-visible diagnostics | Commands use argument vectors without a shell; batch options reject controls; script paths reject option-like names; the launcher forwards only an explicit environment; diagnostic text is normalized, redacted, and limited to 512 characters | `tests/launcher/test_client.py`, `tests/launcher/test_renderer.py`, `tests/launcher/test_runner.py` |
| Batch entrypoint | Directive splitting, shell expansion, plan or runtime substitution, unsafe task identity | Directive names and values are validated; shell values are escaped; the script fixes `PATH`, checks both SHA-256 identities before extraction, validates the array-task ID, and creates a private attempt-local runtime directory | `tests/launcher/test_renderer.py`, `tests/slurm_test_fakes/test_rendered_scripts.py` |
| Runtime commands and environment | Shell fragments, ambient environment leakage, persisted plaintext credentials | Runtime steps are immutable argument vectors; `Popen` uses `shell=False`; only package-owned scheduler variables and explicitly resolved bindings are forwarded; secret-shaped values require environment references and are not persisted | `tests/contracts/test_config_records.py`, `tests/runtime/test_steps.py`, `tests/runtime/test_supervisor.py` |
| Host and container paths | Parent traversal, ambiguous paths, mount escape, read-only mount writes | Persisted paths are normalized absolute POSIX paths below `/`; container translation selects the most-specific resolved mount and separately enforces write access; state and logs use descriptor-bound, no-follow operations | `tests/runtime/test_paths.py`, `tests/runtime/test_preflight.py`, `tests/state/test_store.py` |
| Images and runtime archives | Credential-bearing image references, archive traversal, replacement races, altered runtime source | OCI sources reject credentials and ambiguous schemes; image/state publication is restrictive and atomic; the runtime archive is package-built with fixed member names and metadata, content-addressed, and verified before extraction | `tests/images/test_lifecycle.py`, `tests/images/test_registry_store.py`, `tests/runtime/test_bundle.py` |
| Logs and public evidence | Secret or site-specific data copied into public artifacts | Runtime logs are private `0600` files below private execution directories. The public-artifact audit reports only a display path and rule name, scans explicit log paths without echoing matches, and rejects high-confidence credentials and environment-specific infrastructure values | `tests/runtime/test_supervisor.py`, `tests/test_public_artifacts.py` |
| Cleanup and signals | Orphaned process groups, repeated cleanup, partial publication | Runtime children start in owned sessions; cleanup is idempotent, terminates in reverse order, escalates after a bounded grace period, and surfaces normalized failure state | `tests/runtime/test_supervisor.py`, `tests/runtime/test_controller.py` |

## Findings resolved in this slice

- The Slurm wheel declared Apache-2.0 but did not carry the license text. The package now includes a canonical copy of
  the repository license, verifies the copies are byte-identical, and fails the wheel audit unless the expanded wheel
  contains Apache License 2.0 text.
- Slurm command failures bounded and control-normalized scheduler stderr but did not explicitly redact recognizable
  credentials. A shared redaction helper now covers secret-shaped assignments and options, authorization headers, URL
  user information, and high-confidence provider token formats before diagnostic truncation.
- Public-artifact checks were limited to individual golden tests. The new scanner covers deployable source, package
  metadata, public fixtures, examples/documentation, release scripts, explicit logs, ZIP/wheel members, and tar members.
  Archive traversal, links, excessive member counts, excessive expanded content, missing packaged-source SPDX headers,
  and missing wheel license text fail closed.

## Provenance and dependency review

- All reviewed package Python and shell resources carry NVIDIA Apache-2.0 SPDX headers. Package history identifies the
  resources as repository contributions; no copied or adapted third-party source was identified in this scope.
- Direct runtime dependencies remain `data-designer`, `packaging`, `pydantic`, and `pyyaml`. The dependency inventory
  reports only `click` and `typer` as transitive imports guaranteed by the exact-version `data-designer` dependency;
  there are no unresolved imported modules.
- The scanner deliberately permits only generic test representations that its high-confidence rules do not classify:
  `example.test` hosts, loopback addresses, and `/workspace` paths. Python test modules contain deliberate credential
  sentinels and are not default publication inputs; maintained golden and fixture artifacts are scanned.

## Commands for this review slice

```bash
python scripts/audit_slurm_public_artifacts.py
python scripts/audit_slurm_public_artifacts.py path/to/wheel.whl path/to/sanitized.log
make test-slurm
make test-slurm-wheel-install
make check-slurm
make check-dependency-licenses
```

## Remaining final acceptance

After all #850 implementation and documentation dependencies are merged, #870 must select one public commit and build
the complete wheel set once. The source, wheels, runtime archive, sanitized scenario evidence, and sanitized profile
rerun must all be tied to those exact digests. Any code or wheel change after sealing invalidates the affected evidence.
Environment-specific profiles and raw scheduler or allocation logs remain outside public artifacts and issue comments.
