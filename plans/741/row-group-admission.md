# Plan: Public Row-Group Admission Controls

Fixes #741.

## Problem

Large async runs expose public controls for batch size, scheduler task leases, and
model request caps, but the row-group admission horizon is still hidden in the
async scheduler. Users cannot choose whether the scheduler admits a fixed number
of row groups or ramps the active row-group target adaptively, even though that
choice affects checkpoint cadence, active state size, and endpoint occupancy.

## Goals

1. Expose row-group admission as a supported `RunConfig` policy.
2. Preserve the existing fixed default behavior unless users opt into a wider or
   adaptive policy.
3. Thread the public policy through the dataset-builder boundary into
   `AsyncTaskScheduler`.
4. Keep scheduler diagnostics/capacity plans aligned with the effective public
   settings.
5. Validate fixed and adaptive policies with local mock-provider experiments.

## Non-Goals

- Do not redesign task admission or request admission.
- Do not make adaptive row-group admission AIMD; it remains additive ramp-up
  beneath a hard cap.
- Do not add new model/provider request-concurrency knobs.

## Design

Add `RowGroupAdmissionConfig` and `RowGroupAdmissionMode` to the config package.
`RunConfig.row_group_admission` defaults to a fixed horizon of three active row
groups, matching the current scheduler default while making the policy visible.

`DatasetBuilder._prepare_async_run()` translates the public config into the
existing scheduler constructor arguments:

- `max_concurrent_row_groups`
- `adaptive_row_group_admission`
- `adaptive_row_group_initial_target`
- `max_admitted_rows`

The scheduler keeps ownership of the actual admission loop and capacity-plan
diagnostics. It records whether the row-group horizon and active-row budget came
from `run_config` or internal derivation so capacity reports can distinguish
public configuration from internal defaults.
The public config bounds `max_concurrent_row_groups`. The historical default
fixed horizon preserves row-group-count-only behavior, while widened fixed
horizons and adaptive mode derive an active-row guard when `max_admitted_rows`
is omitted so public row-group tuning cannot multiply active buffers silently.
Adaptive mode rejects row groups larger than the effective active-row guard
instead of admitting an oversized first group.

## Validation

- Config tests cover default exposure, dict/object construction, exports, and
  invalid adaptive-only fields.
- Builder tests verify public row-group admission settings are passed to the
  scheduler.
- Scheduler tests verify capacity diagnostics report the public source, fixed
  cap, adaptive target, and explicit max-admitted-row guard.
- Local mock-provider experiments compare fixed and adaptive horizons across
  fan-out, dependency-chain, and wide-row-group workloads.
