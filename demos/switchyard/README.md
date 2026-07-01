# Data Designer + Switchyard experiments

This demo measures two claims:

1. Data Designer task hints can preserve quality while reducing estimated inference cost.
2. A model mixture can change output diversity behind one Data Designer model alias.

The quality experiment runs 20 exact-answer tasks through weak-only, strong-only, and task-hinted profiles. The hinted route sends easy rows to the weak model and hard rows to the strong model without an extra classifier call. The diversity experiment compares weak-only generation with a seeded 35% strong / 65% weak route.

## Setup

Install Data Designer from the repository root:

```bash
make install-dev
```

Install [Switchyard](https://github.com/NVIDIA-NeMo/Switchyard) in a separate environment. Set these variables without storing endpoint URLs or keys in this repository:

```bash
export NVIDIA_INFERENCE_API_KEY="<API key>"
export NVIDIA_INFERENCE_BASE_URL="<OpenAI-compatible base URL>"
export SWITCHYARD_BASE_URL="<local Switchyard OpenAI base URL>"
```

The optional variables `SWITCHYARD_WEAK_MODEL`, `SWITCHYARD_STRONG_MODEL`, and `SWITCHYARD_CLASSIFIER_MODEL` override the default model IDs. Set `SWITCHYARD_PYTHON` when the Switchyard environment does not provide `python3` on `PATH`.

## Run

Start the proxy:

```bash
make -C demos/switchyard serve
```

In another terminal, run the experiments:

```bash
make -C demos/switchyard run
```

Raw datasets are written under `.scratch/switchyard-experiments`. The sanitized aggregate result is written to `demos/switchyard/results.json`.

Compile the report:

```bash
make -C demos/switchyard report
```

The PDF is written to `output/pdf/switchyard-data-designer-experiments.pdf`.

## Notes

- The server builds five routes directly to avoid upstream catalog discovery and format probes.
- Data Designer health checks are skipped during measured runs after the script verifies Switchyard's health endpoint.
- The `[ROUTE_HINT=...]` prompt marker is a prototype for a future per-call Data Designer metadata contract.
- Cost estimates use Switchyard's model price table. The server also exposes `dd-smart` for optional LLM-classifier experiments.
- Restart the server before a measured run to reset the seeded random-routing sequence.
