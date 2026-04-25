## ADDED Requirements

### Requirement: Read-only loader for `data/public/input/`

`DABenchPublicDataset` SHALL load tasks from a configurable input root (default `data/public/input/`) and SHALL NOT write to that directory or any descendant during normal operation.

#### Scenario: Loader does not modify the input tree
- **WHEN** `DABenchPublicDataset` iterates tasks and accesses task assets
- **THEN** the modification times and contents of files under the input root are unchanged

### Requirement: Strict task.json schema validation

Each `task_<n>/task.json` SHALL be parsed and validated against the documented schema (required fields, types). Missing required fields or wrong types MUST raise a clear error before any task is yielded.

#### Scenario: Missing required field raises clearly
- **WHEN** a `task.json` is missing the `id` field and the dataset is instantiated for that input root
- **THEN** the error path raises with a message naming the missing field and the offending file path

#### Scenario: Wrong type raises clearly
- **WHEN** a `task.json` has `difficulty` as an integer instead of a string
- **THEN** the error message names the field and the offending file path

### Requirement: Natural-number ordering of tasks

`iter_tasks()` SHALL return tasks ordered by the integer suffix of `task_<n>` (i.e. `task_1`, `task_2`, …, `task_10`, `task_11`), not lexicographic order.

#### Scenario: Natural ordering across mixed widths
- **WHEN** the input root contains `task_1`, `task_2`, `task_10`, `task_11`, `task_3`
- **THEN** `[t.id for t in dataset.iter_tasks()]` equals the natural-order list `["task_1","task_2","task_3","task_10","task_11"]`

### Requirement: `iter_tasks` accepts no filter kwargs

`iter_tasks()` SHALL be parameterless. Filter arguments (`task_ids=`, `difficulty=`, `difficulties=`) MUST NOT be accepted; callers that need filtering SHALL apply it after the iterator.

#### Scenario: Calling iter_tasks with kwargs raises TypeError
- **WHEN** `dataset.iter_tasks(difficulty="easy")` is called
- **THEN** Python raises `TypeError` ("unexpected keyword argument 'difficulty'")

### Requirement: Difficulty counts via `difficulty_breakdown`

The dataset SHALL expose a method that returns a mapping from difficulty label to task count.

#### Scenario: Breakdown sums to total task count
- **WHEN** `dataset.difficulty_breakdown()` is called against an input root with 5 tasks
- **THEN** `sum(dataset.difficulty_breakdown().values()) == 5`

### Requirement: Idempotent loading

Calling `iter_tasks()` repeatedly SHALL produce the same sequence of `PublicTask` objects (identity not required; equality of `task.id`, `task.difficulty`, and `task.context_dir` is required).

#### Scenario: Two iterations produce same id sequence
- **WHEN** `iter_tasks()` is called twice
- **THEN** `[t.id for t in run_1] == [t.id for t in run_2]`
