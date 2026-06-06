"""只读文件系统工具集。

给 Agent 暴露有限的目录/文件读取能力，所有访问都限定在
`task.context_dir` 内（`resolve_context_path` 负责越界检查）。
这些工具都是幂等且无副作用的——Agent 可以反复调用而不影响外部状态。

本包原为单文件 `filesystem.py`（972 行），按职责拆分为：
- `path`          路径解析与目录树遍历
- `constants`     共享常量与扩展名集合
- `csv_preview`   CSV 预览与列级 profiling
- `json_preview`  JSON 预览与流式 schema 抽取
- `doc_preview`   文档（markdown/txt）预览
- `pdf_preview`   PDF 预览与元信息提取
- `inspect`       inspect_context_files 及格式分流
"""

from agents.tools.filesystem import constants as constants
from agents.tools.filesystem.csv_preview import (
    classify_dtype as classify_dtype,
)
from agents.tools.filesystem.csv_preview import (
    read_csv_preview as read_csv_preview,
)
from agents.tools.filesystem.doc_preview import (
    read_doc_preview as read_doc_preview,
)
from agents.tools.filesystem.inspect import (
    inspect_context_files as inspect_context_files,
)
from agents.tools.filesystem.json_preview import (
    read_json_preview as read_json_preview,
)
from agents.tools.filesystem.path import (
    list_context_tree as list_context_tree,
)
from agents.tools.filesystem.path import (
    resolve_context_path as resolve_context_path,
)
