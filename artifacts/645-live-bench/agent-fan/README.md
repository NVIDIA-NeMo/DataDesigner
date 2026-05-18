# Agent Fan Live Benchmark

Output: `/Users/etramel/src/DataDesigner/artifacts/645-live-bench/agent-fan`

## fan_out_mp2_short
- DAG shape: fan-out
- Wall time: 6.556s
- Rows / columns: 2 / 6
- max_parallel_requests: 2; observed max request in-flight: 2; waiters: 4
- Scheduler events: `{'dependency_ready': 7, 'ready_enqueued': 7, 'selected': 7, 'task_lease_acquired': 7, 'worker_spawned': 7, 'queue_drained': 2, 'task_completed': 7, 'task_lease_released': 7}`
- Request events: `{'request_resource_registered': 1, 'request_effective_cap_changed': 1, 'request_queue_formed': 3, 'request_wait_started': 6, 'request_wait_completed': 6, 'request_lease_acquired': 6, 'request_queue_drained': 3, 'model_request_started': 6, 'request_lease_released': 6, 'model_request_completed': 6}`
- Timeline: `/Users/etramel/src/DataDesigner/artifacts/645-live-bench/agent-fan/fan_out_mp2_short/timeline.jsonl`

## fan_in_mp2_mixed
- DAG shape: fan-in
- Wall time: 10.622s
- Rows / columns: 2 / 7
- max_parallel_requests: 2; observed max request in-flight: 2; waiters: 4
- Scheduler events: `{'dependency_ready': 9, 'ready_enqueued': 9, 'selected': 9, 'task_lease_acquired': 9, 'worker_spawned': 9, 'queue_drained': 4, 'task_completed': 9, 'task_lease_released': 9}`
- Request events: `{'request_resource_registered': 1, 'request_effective_cap_changed': 1, 'request_queue_formed': 5, 'request_wait_started': 8, 'request_wait_completed': 8, 'request_lease_acquired': 8, 'request_queue_drained': 5, 'model_request_started': 8, 'request_lease_released': 8, 'model_request_completed': 8}`
- Timeline: `/Users/etramel/src/DataDesigner/artifacts/645-live-bench/agent-fan/fan_in_mp2_mixed/timeline.jsonl`
