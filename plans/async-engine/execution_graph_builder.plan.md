---
name: Execution Graph Builder
overview: Build a memory-efficient execution graph that models cell-level dependencies for async dataset generation, supporting different generator execution traits (start, cell-by-cell, row-streamable, barrier) with trait inference from generator properties.
todos:
  - id: node-types
    content: Create node_id.py with CellNodeId, BarrierNodeId, and NodeId type alias
    status: pending
  - id: traits-enum
    content: Create traits.py with ExecutionTraits Flag enum
    status: pending
  - id: column-descriptor
    content: Create column_descriptor.py with ColumnDescriptor dataclass
    status: pending
  - id: base-class-property
    content: Add is_row_streamable property to ColumnGenerator base class and override in subclasses
    status: pending
  - id: execution-graph
    content: Create graph.py with ExecutionGraph class (hybrid representation, node iteration, dependency queries)
    status: pending
  - id: graph-builder
    content: Create builder.py with GraphBuilder factory that infers traits from generator properties
    status: pending
  - id: completion-tracker
    content: Create completion.py with CompletionTracker (simple) and ThreadSafeCompletionTracker variants
    status: pending
  - id: init-exports
    content: Create __init__.py with public API exports
    status: pending
  - id: tests
    content: Add comprehensive tests for graph building, node iteration, and dependency resolution
    status: pending
isProject: false
---

# Execution Graph Builder

## Overview

Build a new graph-based execution framework that models cell-level dependencies for async dataset generation. The graph uses a **hybrid representation** where column structure is stored explicitly while cell-level nodes are computed on-demand to handle millions of records efficiently.

## Architecture

```mermaid
graph TD
    subgraph config [Configuration Layer]
        DDC[DataDesignerConfig]
        CC[ColumnConfigs]
        GEN[ColumnGenerators]
    end

    subgraph graphLayer [Graph Builder]
        GB[GraphBuilder]
        GT[GeneratorTraits]
        CD[ColumnDescriptor]
        EG[ExecutionGraph]
    end

    subgraph execution [Execution Engine Interface]
        NI[iter_ready_nodes]
        EC[ExecutionContext]
        CT[CompletionTracker]
    end

    DDC --> CC
    CC --> GB
    GEN --> GT
    GB --> CD
    CD --> EG
    EG --> NI
    EG --> EC
    NI --> CT
```



## Generator Execution Traits

Traits are inferred from generator properties (no hardcoded class-name matching):


| Trait            | Source Property                             | Description                                 |
| ---------------- | ------------------------------------------- | ------------------------------------------- |
| `START`          | `can_generate_from_scratch = True`          | Can initiate workflows without input        |
| `CELL_BY_CELL`   | `get_generation_strategy() == CELL_BY_CELL` | Processes individual cells independently    |
| `ROW_STREAMABLE` | `is_row_streamable = True`                  | Full column generator that can emit per-row |
| `BARRIER`        | `is_row_streamable = False` on full-column  | Requires all inputs before any output       |


### New Generator Property

Add `is_row_streamable` property to `ColumnGenerator` base class in [packages/data-designer-engine/src/data_designer/engine/column_generators/generators/base.py](packages/data-designer-engine/src/data_designer/engine/column_generators/generators/base.py):

```python
@property
def is_row_streamable(self) -> bool:
    """Whether this generator can emit results as rows complete."""
    return self.get_generation_strategy() == GenerationStrategy.CELL_BY_CELL
```

Override in specific generators:

- `ExpressionColumnGenerator`: `True` (processes rows independently)
- `ValidationColumnGenerator`: `False` (needs all rows for batch processing)
- `SamplerColumnGenerator`: `True` (generates per-row data)

## Dependency Resolution

### Cell-by-Cell / Row-Streamable Columns

Row-local dependencies:

- Cell `(row=5, col="question")` depends on `(row=5, col="context")`

### Barrier Columns

All-or-nothing dependencies:

- `BarrierNodeId("validation")` depends on ALL cells of dependency columns
- Output cells `(row=*, col="validation")` depend on `BarrierNodeId("validation")`

```mermaid
graph LR
    subgraph inputColumn [Input Column]
        C0[Cell 0]
        C1[Cell 1]
        CN[Cell N]
    end

    subgraph barrier [Barrier Column]
        B[BarrierNode]
        O0[Output 0]
        O1[Output 1]
        ON[Output N]
    end

    C0 --> B
    C1 --> B
    CN --> B
    B --> O0
    B --> O1
    B --> ON
```



## File Structure

```
packages/data-designer-engine/src/data_designer/engine/execution_graph/
  __init__.py           # Public exports
  node_id.py            # CellNodeId, BarrierNodeId, NodeId type alias
  traits.py             # ExecutionTraits Flag enum
  column_descriptor.py  # ColumnDescriptor dataclass
  graph.py              # ExecutionGraph class (hybrid representation)
  builder.py            # GraphBuilder factory (infers traits from generators)
  completion.py         # CompletionTracker (memory-efficient tracking)
```

## Key Components

### 1. Node Identification (`node_id.py`)

```python
@dataclass(frozen=True, slots=True)
class CellNodeId:
    row: int
    column: str

@dataclass(frozen=True, slots=True)
class BarrierNodeId:
    column: str

NodeId = CellNodeId | BarrierNodeId
```

### 2. Execution Traits (`traits.py`)

```python
class ExecutionTraits(Flag):
    NONE = 0
    START = auto()           # Can generate from scratch
    CELL_BY_CELL = auto()    # Processes individual cells
    ROW_STREAMABLE = auto()  # Can emit as rows complete
    BARRIER = auto()         # Requires all inputs first
```

### 3. Column Descriptor (`column_descriptor.py`)

```python
@dataclass(slots=True)
class ColumnDescriptor:
    name: str
    config: ConfigBase
    generator_cls: type[ColumnGenerator]
    traits: ExecutionTraits
    dependencies: list[str]     # from required_columns
    side_effects: list[str]     # additional columns produced
```

### 4. Execution Graph (`graph.py`)

Key methods for execution engines:

- `iter_start_nodes()` - Nodes that can begin immediately
- `iter_ready_nodes(completed)` - Nodes with satisfied dependencies
- `get_dependencies(node)` - Dependencies for a node
- `get_generator_and_config(node)` - Generator class and config for execution

### 5. Graph Builder (`builder.py`)

```python
class GraphBuilder:
    def build(self, config: DataDesignerConfig, num_records: int) -> ExecutionGraph:
        # 1. Build descriptors from column configs
        # 2. Infer traits from generator properties (not class names!)
        # 3. Handle multi-column configs (mark additional columns as side effects)
        # 4. Validate dependencies exist
        # 5. Return ExecutionGraph with descriptors in topo order
```

Trait inference (plugin-compatible):

```python
def _infer_traits(self, gen_cls: type[ColumnGenerator]) -> ExecutionTraits:
    traits = ExecutionTraits.NONE

    # Check can_generate_from_scratch property (works for any generator)
    if getattr(gen_cls, 'can_generate_from_scratch', False):
        traits |= ExecutionTraits.START

    # Check generation strategy
    strategy = gen_cls.get_generation_strategy()
    if strategy == GenerationStrategy.CELL_BY_CELL:
        traits |= ExecutionTraits.CELL_BY_CELL | ExecutionTraits.ROW_STREAMABLE
    else:  # FULL_COLUMN
        # Check is_row_streamable property
        if getattr(gen_cls, 'is_row_streamable', False):
            traits |= ExecutionTraits.ROW_STREAMABLE
        else:
            traits |= ExecutionTraits.BARRIER

    return traits
```

### 6. Completion Tracker (`completion.py`)

Memory-efficient tracking for large datasets with two variants:

**Base Protocol:**

```python
class CompletionTrackerProtocol(Protocol):
    """Protocol for completion tracking - enables duck typing."""

    def mark_complete(self, node: NodeId) -> None: ...
    def is_complete(self, node: NodeId) -> bool: ...
    def is_column_complete(self, column: str) -> bool: ...
    def __contains__(self, node: NodeId) -> bool: ...
```

**Simple Tracker (asyncio / single-threaded):**

```python
class CompletionTracker:
    """O(C) memory instead of O(C x R). No locks - for single-threaded use."""

    def mark_complete(self, node: NodeId) -> None: ...
    def is_complete(self, node: NodeId) -> bool: ...
    def is_column_complete(self, column: str) -> bool: ...
```

**Thread-Safe Tracker (thread pool / multi-threaded):**

```python
class ThreadSafeCompletionTracker:
    """Thread-safe variant with internal locking for concurrent access."""

    def mark_complete(self, node: NodeId) -> None:
        with self._lock:
            # ... same logic as CompletionTracker

    def is_complete(self, node: NodeId) -> bool:
        with self._lock:
            # ... same logic as CompletionTracker
```

Both trackers use the same memory-efficient strategy:

- Track fully completed columns as a set of names: O(C)
- Only store partial completion progress for in-progress columns
- Automatically compact when columns fully complete

**Thread-Safety Notes:**

- `ExecutionGraph` is **immutable after construction** - no locks needed
- Only the `CompletionTracker` holds mutable state
- Choose tracker variant based on your execution model:
  - `CompletionTracker`: asyncio, single-threaded executors
  - `ThreadSafeCompletionTracker`: thread pools, concurrent.futures

## Memory Efficiency


| Aspect            | Storage                                        |
| ----------------- | ---------------------------------------------- |
| Column metadata   | O(C) where C = number of columns               |
| Cell nodes        | Virtual - O(1) per node, computed on iteration |
| Topological order | O(C), cached once                              |
| NodeId objects    | Frozen dataclasses with `slots=True`           |
| Dependency edges  | Computed mathematically, not stored            |


A graph with 1M records and 10 columns uses ~same memory as 10 records.

## Example Usage

```python
from data_designer.engine.execution_graph import GraphBuilder, ExecutionGraph
from data_designer.engine.execution_graph.completion import (
    CompletionTracker,
    ThreadSafeCompletionTracker,
)

# Build graph from config (immutable after construction)
builder = GraphBuilder(column_generator_registry)
graph: ExecutionGraph = builder.build(config, num_records=1_000_000)

# Option 1: Single-threaded / asyncio execution
tracker = CompletionTracker(graph.num_records)
for node in graph.iter_ready_nodes(tracker):
    gen_cls, config = graph.get_generator_and_config(node)
    # Execute node...
    tracker.mark_complete(node)

# Option 2: Multi-threaded execution (e.g., with ThreadPoolExecutor)
tracker = ThreadSafeCompletionTracker(graph.num_records)
# Safe to call mark_complete() from multiple threads concurrently
```

## Multi-Column Config Handling

Multi-column configs (samplers, seed datasets) produce multiple output columns. The graph handles this by:

1. Using the first column as the primary descriptor name
2. Marking additional columns as `side_effects`
3. Building a reverse mapping `_side_effect_to_parent` for dependency resolution
4. Allowing downstream columns to depend on any produced column

## Validation

The graph builder validates:

1. All dependency columns exist (or are side effects of existing columns)
2. No circular dependencies (enforced by topological order)
3. At least one column has `START` trait (can generate from scratch)

