# Code Sketches

Structural sketches of the main components. Not runnable — shows how the pieces
fit together. See [async-generators-and-task-queue.md](async-generators-and-task-queue.md)
for the full design.

## ExecutionGraph

```python
@dataclass
class ExecutionGraph:
    _upstream: dict[str, set[str]]             # column → upstream columns
    _downstream: dict[str, set[str]]           # column → downstream columns
    _strategies: dict[str, GenerationStrategy]  # column → cell-by-cell or full-column
    _side_effect_map: dict[str, str]            # e.g. "summary__trace" → "summary"

    def upstream(self, column: str) -> set[str]: ...
    def downstream(self, column: str) -> set[str]: ...
    def strategy(self, column: str) -> GenerationStrategy: ...

    def cell_dependencies(
        self,
        column: str,
        row_group: int,
        row_index: int | None,
        row_group_size: int,
    ) -> list[tuple[str, int, int | None]]:
        """Derive cell-level deps on demand from column-level DAG + strategy.

        cell-by-cell upstream, row 2:  [(upstream, rg, 2)]
        full-column upstream:          [(upstream, rg, 0), (upstream, rg, 1), ...]
        from-scratch (no upstream):    []
        """
        deps: list[tuple[str, int, int | None]] = []
        for up_col in self.upstream(column):
            up_strategy = self.strategy(up_col)
            if up_strategy == GenerationStrategy.CELL_BY_CELL:
                if row_index is not None:
                    deps.append((up_col, row_group, row_index))
                else:
                    for ri in range(row_group_size):
                        deps.append((up_col, row_group, ri))
            else:
                deps.append((up_col, row_group, None))
        return deps

    def topological_order(self) -> list[str]: ...
    def critical_path(self) -> list[str]: ...
    def task_count(self, num_records: int, buffer_size: int) -> dict[str, int]: ...
    def to_mermaid(self) -> str: ...
```

### Building the graph

```python
def build_execution_graph(
    column_configs: list[ColumnConfigT],
    strategies: dict[str, GenerationStrategy],
) -> ExecutionGraph:
    graph = ExecutionGraph()
    for config in column_configs:
        name = config.name
        graph._strategies[name] = strategies[name]

        required = set(config.required_columns)
        resolved = set()
        for req in required:
            # "summary__trace" → dependency on the "summary" generator
            if req in graph._side_effect_map:
                resolved.add(graph._side_effect_map[req])
            else:
                resolved.add(req)
        graph._upstream[name] = resolved

        for dep in resolved:
            graph._downstream.setdefault(dep, set()).add(name)

        # Register side-effect outputs (e.g., __trace, __reasoning_content)
        # so downstream references resolve correctly
        ...

    # Validate: acyclic, all required columns resolve to known producers
    ...
    return graph
```

## CompletionTracker

Pure state store — does not resolve dependencies or determine readiness.

```python
class CompletionTracker:
    def __init__(self) -> None:
        self._completed: dict[int, dict[str, set[int]]] = {}  # rg → col → {row indices}
        self._dropped: dict[int, set[int]] = {}                # rg → {row indices}

    def mark_complete(self, column: str, row_group: int, row_index: int) -> None:
        self._completed.setdefault(row_group, {}).setdefault(column, set()).add(row_index)

    def mark_batch_complete(self, column: str, row_group: int, row_group_size: int) -> None:
        self._completed.setdefault(row_group, {})[column] = set(range(row_group_size))

    def is_complete(self, column: str, row_group: int, row_index: int) -> bool:
        return row_index in self._completed.get(row_group, {}).get(column, set())

    def all_complete(self, cells: list[tuple[str, int, int | None]]) -> bool:
        """Used by the scheduler: tracker.all_complete(graph.cell_dependencies(...))"""
        for col, rg, ri in cells:
            if ri is None:
                if col not in self._completed.get(rg, {}):
                    return False
            elif not self.is_complete(col, rg, ri):
                return False
        return True

    def drop_row(self, row_group: int, row_index: int) -> None:
        self._dropped.setdefault(row_group, set()).add(row_index)

    def is_dropped(self, row_group: int, row_index: int) -> bool:
        return row_index in self._dropped.get(row_group, set())

    def is_row_group_complete(
        self, row_group: int, row_group_size: int, all_columns: list[str],
    ) -> bool:
        dropped = self._dropped.get(row_group, set())
        completed = self._completed.get(row_group, {})
        for ri in range(row_group_size):
            if ri in dropped:
                continue
            for col in all_columns:
                if ri not in completed.get(col, set()):
                    return False
        return True
```

## Task model

```python
@dataclass(frozen=True)
class Task:
    column: str
    row_group: int
    row_index: int | None  # None for batch/full-column tasks
    task_type: Literal["from_scratch", "cell", "batch", "pre_batch_processor", "post_batch_processor"]

@dataclass
class TaskResult:
    task: Task
    status: Literal["success", "error"]
    output: Any = None
    error: Exception | None = None
    retryable: bool = False

@dataclass
class TaskTrace:
    task: Task
    dispatched_at: float = 0.0
    slot_acquired_at: float = 0.0
    completed_at: float = 0.0
    status: str = ""
    error: str | None = None
```

## AsyncTaskScheduler

```python
class AsyncTaskScheduler:
    def __init__(
        self,
        generators: dict[str, ColumnGenerator],  # column name → generator (multi-column: same instance)
        graph: ExecutionGraph,
        tracker: CompletionTracker,
        row_groups: list[tuple[int, int]],  # (rg_id, rg_size)
        *,
        max_concurrent_row_groups: int = 3,
        max_submitted_tasks: int = 256,
        max_execution_slots: int = 128,
        salvage_max_rounds: int = 2,
        trace: bool = False,
    ) -> None:
        self._generators = generators
        self._graph = graph
        self._tracker = tracker
        self._rg_semaphore = asyncio.Semaphore(max_concurrent_row_groups)
        self._submission_semaphore = asyncio.Semaphore(max_submitted_tasks)
        self._execution_semaphore = asyncio.Semaphore(max_execution_slots)
        self._dispatched: set[Task] = set()
        self._wake_event = asyncio.Event()
        self.traces: list[TaskTrace] = []

        # Multi-column dedup: group output columns by generator identity
        instance_map: dict[int, list[str]] = {}
        for col, gen in generators.items():
            instance_map.setdefault(id(gen), []).append(col)
        self._instance_to_columns = instance_map
        ...

    async def run(self) -> None:
        # Admit row groups behind semaphore, dispatch seeds
        for rg_id, rg_size in self._row_groups:
            await self._rg_semaphore.acquire()
            await self._dispatch_seeds(rg_id, rg_size)

        # Main loop: find ready tasks, dispatch, repeat
        while not self._all_complete():
            self._wake_event.clear()
            ready = self._get_ready_tasks()
            for task in ready:
                await self._submission_semaphore.acquire()
                asyncio.create_task(self._execute_task(task))
            if not ready:
                await self._wake_event.wait()

        # Salvage rounds for deferred retryable failures
        for _ in range(self._salvage_max_rounds):
            if not self._deferred:
                break
            await self._run_salvage_round()

    def _get_ready_tasks(self) -> list[Task]:
        """The core readiness check — combines graph + tracker + scheduler policy."""
        ready: list[Task] = []
        seen_instances: set[int] = set()
        for rg_id, rg_size in self._active_row_groups():
            for col in self._graph.topological_order():
                gen = self._generators[col]
                if id(gen) in seen_instances:
                    continue  # multi-column: already dispatched via sibling
                strategy = self._graph.strategy(col)
                if strategy == GenerationStrategy.CELL_BY_CELL:
                    for ri in range(rg_size):
                        task = Task(col, rg_id, ri, "cell")
                        if task in self._dispatched:
                            continue
                        if self._tracker.is_dropped(rg_id, ri):
                            continue
                        deps = self._graph.cell_dependencies(col, rg_id, ri, rg_size)
                        if self._tracker.all_complete(deps):
                            ready.append(task)
                else:
                    task = Task(col, rg_id, None, "batch")
                    if task in self._dispatched:
                        continue
                    deps = self._graph.cell_dependencies(col, rg_id, None, rg_size)
                    if self._tracker.all_complete(deps):
                        ready.append(task)
                        seen_instances.add(id(gen))
        return ready

    async def _execute_task(self, task: Task) -> None:
        self._dispatched.add(task)
        try:
            generator = self._generators[task.column]

            # 1. acquire execution slot → prepare request
            await self._execution_semaphore.acquire()
            # ... prepare request ...
            self._execution_semaphore.release()

            # 2. await throttle permit (LLM tasks only; skipped without PR #344)
            # ...

            # 3. reacquire execution slot → execute + writeback
            await self._execution_semaphore.acquire()
            if task.task_type == "from_scratch":
                result_df = await generator.agenerate_from_scratch(...)
                # write all rows to buffer
            elif task.task_type == "cell":
                row_data = ...  # read from buffer manager
                result = await generator.agenerate(row_data)
                if not self._tracker.is_dropped(task.row_group, task.row_index):
                    ...  # buffer_manager.update_cell(...)
            else:
                batch_df = ...  # read from buffer manager
                result_df = await generator.agenerate(batch_df.copy())
                # merge result columns back
            self._execution_semaphore.release()

            # Mark all output columns complete (handles multi-column generators)
            output_cols = self._instance_to_columns[id(generator)]
            for col in output_cols:
                if task.row_index is None:
                    self._tracker.mark_batch_complete(col, task.row_group, ...)
                else:
                    self._tracker.mark_complete(col, task.row_group, task.row_index)
        except Exception as exc:
            self._handle_task_failure(task, exc)
        finally:
            self._submission_semaphore.release()
            self._wake_event.set()
            # check if row group is complete → post-batch → checkpoint
```

## Generator base class changes

Additions to the existing `ColumnGenerator`:

```python
class ColumnGenerator(ConfigurableTask[TaskConfigT], ABC):

    @property
    def is_stateful(self) -> bool:
        """Override to True for generators that maintain state across calls."""
        return False

    # --- Symmetric bridging ---

    async def agenerate(self, data: dict) -> dict:
        """Default: wrap sync generate in a thread."""
        return await asyncio.to_thread(self.generate, data)

    def generate(self, data: dict) -> dict:
        """Default (for async-first subclasses): safe sync bridge."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.agenerate(data))
        else:
            loop = ...  # builder's background event loop
            future = asyncio.run_coroutine_threadsafe(self.agenerate(data), loop)
            return future.result(timeout=300)


class FromScratchColumnGenerator(ColumnGenerator[TaskConfigT], ABC):

    async def agenerate_from_scratch(self, num_records: int) -> pd.DataFrame:
        return await asyncio.to_thread(self.generate_from_scratch, num_records)

    async def agenerate(self, data: pd.DataFrame) -> pd.DataFrame:
        return await asyncio.to_thread(self.generate, data.copy())


class ColumnGeneratorFullColumn(ColumnGenerator[TaskConfigT], ABC):

    async def agenerate(self, data: pd.DataFrame) -> pd.DataFrame:
        return await asyncio.to_thread(self.generate, data.copy())
```

## Builder integration

Additions to `ColumnWiseDatasetBuilder`:

```python
class ColumnWiseDatasetBuilder:

    def _build_async(self, generators: list[ColumnGenerator], num_records: int, buffer_size: int) -> None:
        # 1. Build graph
        strategies = {g.task_config.name: g.get_generation_strategy() for g in generators}
        graph = build_execution_graph(self._column_configs, strategies)

        # 2. Partition into row groups
        row_groups = []
        remaining = num_records
        rg_id = 0
        while remaining > 0:
            size = min(buffer_size, remaining)
            row_groups.append((rg_id, size))
            remaining -= size
            rg_id += 1

        # 3. Create tracker and scheduler
        tracker = CompletionTracker()
        # Multi-column generators: multiple column keys → same instance
        gen_map = {g.task_config.name: g for g in generators}
        scheduler = AsyncTaskScheduler(
            generators=gen_map,  # scheduler deduplicates by identity
            graph=graph,
            tracker=tracker,
            row_groups=row_groups,
        )

        # 4. Run on background event loop
        loop = self._ensure_async_engine_loop()
        future = asyncio.run_coroutine_threadsafe(scheduler.run(), loop)
        future.result()
```

## How the three layers interact

On each task completion:

```
1. Task completes
   → tracker.mark_complete(col, rg, row)  # all output columns for multi-column generators

2. Scheduler wakes up (_wake_event.set())
   → for each candidate task in active row groups (deduped by generator identity):
       deps = graph.cell_dependencies(col, rg, row, rg_size)
       if tracker.all_complete(deps) and task not dispatched:
           dispatch it (acquire submission slot → execute behind execution semaphore)

3. If tracker.is_row_group_complete(rg, rg_size, all_columns):
   → run post-batch processors
   → checkpoint to parquet
   → release row group semaphore slot
```

The graph owns **what depends on what** (static).
The tracker owns **what's done** (runtime state).
The scheduler owns **what to do next** (runtime policy).
