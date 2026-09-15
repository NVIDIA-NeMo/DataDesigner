# Slurm documentation coverage map

This map records the behavioral migration from the legacy Big Iron guides to the public Fern Slurm guide. It tracks user-visible concepts rather than source wording or environment-specific examples.

| Legacy area | Behavior retained in Fern | Fern destination | Disposition |
| --- | --- | --- | --- |
| Installation | Python environment, optional Slurm extra, CLI discovery | `overview.mdx`, `getting-started.mdx` | Retained and updated for the same-version optional wheel |
| Environment variables | Profile-file override, secret references, runtime environment boundaries | `profiles.mdx`, `dependencies-and-tools.mdx` | Relocated into the owning workflows |
| Cluster prerequisites | Submit commands, shared storage, Enroot, scheduler and GPU facts | `overview.mdx` | Expanded with workstation, login-host, and compute-node personas |
| Quick start | Install, profile, image import, builder, run config, dry-run, submit, status, cancel, output | `getting-started.mdx` | Replaced with a clean 13-step public flow |
| Profile configuration | Account, partitions, GPU discovery, GPU request mode, mounts | `profiles.mdx` | Expanded to a named multi-cluster catalog |
| Image management | OCI import, existing SQSH inspection, list, inspect, replace, remove | `images.mdx` | Expanded with digest and role verification rules |
| Client and serving images | Separate image roles and compatibility requirements | `images.mdx` | Retained and made explicit |
| Run configuration | Builder input, invocation, client, deployment, submission, output | `run-configuration.mdx` | Replaced with the shipped strict public schema |
| Server options | vLLM timeouts, readiness, backpressure, environment, safe arguments | `run-configuration.mdx` | Retained through supported public fields |
| Multi-node serving | Nodes, tensor parallelism, nodes per replica | `run-configuration.mdx` | Retained and expressed as deployment topology |
| Multiple models | Per-alias deployment, concurrency, and image selection | `run-configuration.mdx` | Retained and expanded to separate serving images |
| Job arrays | Shard count, concurrency throttle, persisted shard identity | `run-configuration.mdx` | Retained with public array constraints |
| Execute and observe | Dry-run, submit, JSON results, scheduler reconciliation | `operations.mdx` | Expanded with durable state behavior |
| Cancellation | Managed job ownership and idempotent cancellation | `operations.mdx` | Retained with run-ID ownership checks |
| Logs and results | Durable logs, attempt records, winners, default output | `operations.mdx` | Expanded with the persistent storage boundary |
| Runtime scratch | Runtime, dependency, cache, and Enroot placement and cleanup | `overview.mdx`, `operations.mdx`, `troubleshooting.mdx` | Intentional change to allocation-local scratch with stable container mount |
| Retry | Sparse task selection, resume policy, confirmation, dry-run | `retry-and-collection.mdx` | Expanded with immutable attempt history |
| Merge and collection | Winner-driven collection to an explicit destination | `retry-and-collection.mdx` | Retained as the public `merge` command |
| Dependencies | Pure-wheel overlay, immutable locks, client-image compatibility | `dependencies-and-tools.mdx` | Expanded with strict accepted sources |
| Plugins | Client image or dependency-overlay installation | `dependencies-and-tools.mdx` | Relocated and linked to public plugin contracts |
| Private packages and secrets | Built client images, immutable wheel or lock inputs, and external secret references | `dependencies-and-tools.mdx` | Replaced with the shipped public-index resolver and typed environment references |
| MCP providers | Remote and local stdio providers in client allocations | `dependencies-and-tools.mdx` | Added to match the shipped public surface |
| Code sandbox | Auxiliary isolated execution service | `dependencies-and-tools.mdx` | Intentionally unsupported in Slurm v1; alternatives documented |
| Benchmarks | Case expansion, child runs, status, point-in-time analysis | `benchmarks.mdx` | Retained and expanded with durable benchmark state |
| FAQ and recovery | Installation, profile, image, scheduler, readiness, retry, output, scratch | `troubleshooting.mdx` | Reorganized by observed failure |
| Command summary | Commands, common options, JSON output, exit codes | `cli-reference.mdx` | Replaced with a concise shipped-CLI reference |

## Acceptance boundary

The Fern guide and checked examples are validated in repository CI and against locally built wheels. Formal sealed-artifact acceptance, including the complete real-cluster scenario matrix and sanitized evidence bundle, remains issue #870 and must run after this documentation and release-integration slice merges and any required fixes land.
