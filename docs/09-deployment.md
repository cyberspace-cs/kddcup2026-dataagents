# 9. 部署方案（Deployment）

> **目标**：把 `src/dabench` 源码打进 Docker 镜像，满足规则 §3（≤10 GB、可直接 `docker run`、不硬编码 URL/Key）。
> 并提供本地脚本：构建、评测、对拍、打包、提交。

## 9.1 Dockerfile 设计

### 9.1.1 分层策略与大小目标

| 层 | 内容 | 预估大小 |
|---|---|---|
| Base | `python:3.11-slim` | ~45 MB |
| System deps | `build-essential`（编译 wheel）+ tesseract + poppler-utils | ~500 MB（build stage） / ~150 MB（runtime 保留） |
| Python deps | pandas, polars, duckdb, pyarrow, pdfplumber, pymupdf, pillow, paddleocr, openai, pyyaml, pydantic, numpy | ~1.5-2 GB |
| Source | `src/dabench/`、`configs/eval.yaml` | <10 MB |
| **镜像总计** | | **~2-2.5 GB未压缩** → **~800 MB 压缩后** |

远低于 10 GB 上限，留出 embedding 模型 / 额外数据的余量。

### 9.1.2 多阶段 Dockerfile

`docker/Dockerfile`：

```dockerfile
# syntax=docker/dockerfile:1.6

# ─── Stage 1: builder ─────────────────────────────────────────────
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 系统构建依赖（仅 builder 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# 装 uv（快）
RUN pip install --no-cache-dir uv

WORKDIR /build
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-dev --no-hashes -o requirements.txt
RUN uv pip install --system --no-cache -r requirements.txt

# ─── Stage 2: runtime ─────────────────────────────────────────────
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# 运行时系统依赖（OCR、PDF）
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-chi-sim \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /root/.cache

# 从 builder 拷已装好的 site-packages（避免重装）
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 应用代码
WORKDIR /app
COPY src/dabench ./dabench
COPY configs/eval.yaml ./configs/eval.yaml

# 评测挂载点（仅声明，实际由 docker run 挂）
# /input (ro), /output (rw), /logs (rw)

# 降级到非 root（规则允许 root，但安全实践）
# RUN useradd -m -u 1000 dabench && chown -R dabench:dabench /app
# USER dabench
# ↑ 先保留注释；规则说允许 root，且 starter kit 用 root，先不改

ENTRYPOINT ["python", "-m", "dabench.main"]
```

### 9.1.3 `.dockerignore`

```
# 源码以外的一切不进镜像
.git/
.venv/
data/
artifacts/
tests/
docs/
scripts/
vendor/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
*.egg-info/
*.tar.gz
```

### 9.1.4 `pyproject.toml` 关键字段

```toml
[project]
name = "dabench"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "openai>=2.28.0",
    "typer>=0.15.0",
    "rich>=13.9.0",
    "pyyaml>=6.0.0",
    "pydantic>=2.10.0",
    "pandas>=2.2.0",
    "polars>=1.10.0",
    "duckdb>=1.1.0",
    "pyarrow>=18.0.0",
    "numpy>=2.0.0",
    "pdfplumber>=0.11.0",
    "pymupdf>=1.25.0",
    "pillow>=11.0.0",
    "jsonschema>=4.23.0",
    # OCR fallback（兜底路径）
    "pytesseract>=0.3.13",
    # Embedding for RAG（体积考量：用 bge-small-zh 或 text2vec-base-chinese，约 100MB）
    "sentence-transformers>=3.3.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project.scripts]
dabench = "dabench.cli:main"
```

**依赖选型理由**：
- `paddleocr` 体积大（~500MB），改用 `pytesseract`（已装 tesseract binary）
- `sentence-transformers` 约 400MB；若镜像紧张，可延后到 v2
- `polars` + `duckdb` 二者其一已足；保留 polars 处理流式读 CSV，duckdb 处理内存内 join

## 9.2 本地评测脚本

### 9.2.1 `scripts/eval_local.sh`

模拟组委会评测环境的**功能等价**跑法（网络部分有妥协——真断网无法调开发 LLM；改用白名单代理或本地 mock）：

```bash
#!/usr/bin/env bash
set -euo pipefail

# ─── 配置 ─────────────────────────────────────────────
IMAGE_TAG="${IMAGE_TAG:-dabench:dev}"
INPUT_DIR="${INPUT_DIR:-$PWD/data/public/input}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/artifacts/eval_output}"
LOGS_DIR="${LOGS_DIR:-$PWD/artifacts/eval_logs}"
MODEL_API_URL="${MODEL_API_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
MODEL_API_KEY="${MODEL_API_KEY:-$DASHSCOPE_API_KEY}"
MODEL_NAME="${MODEL_NAME:-qwen-max-latest}"

# 本机规格：16GB RAM → 给容器 10GB；24 核 → 给 12
CPUS="${CPUS:-12}"
MEMORY="${MEMORY:-10g}"

# ─── 预检 ─────────────────────────────────────────────
[[ -d "$INPUT_DIR" ]]    || { echo "ERROR: input dir missing: $INPUT_DIR"; exit 2; }
[[ -n "$MODEL_API_KEY" ]] || { echo "ERROR: MODEL_API_KEY empty"; exit 2; }
command -v docker >/dev/null || { echo "ERROR: docker not found"; exit 2; }

mkdir -p "$OUTPUT_DIR" "$LOGS_DIR"

# ─── 构建 ─────────────────────────────────────────────
echo "=== Building image $IMAGE_TAG ==="
docker build -t "$IMAGE_TAG" -f docker/Dockerfile .

# 检查镜像大小
SIZE_MB=$(docker image inspect "$IMAGE_TAG" --format='{{.Size}}' | awk '{printf "%.0f", $1/1024/1024}')
echo "Image size: ${SIZE_MB} MB"
if (( SIZE_MB > 8000 )); then
    echo "WARNING: image > 8 GB, check 10 GB limit"
fi

# ─── 运行 ─────────────────────────────────────────────
echo "=== Running container ==="
# 注意：这里 --network=bridge 以允许调开发 LLM；真评测是 --network=none
docker run --rm \
    --cpus="$CPUS" \
    --memory="$MEMORY" \
    --memory-swap="$MEMORY" \
    -e MODEL_API_URL="$MODEL_API_URL" \
    -e MODEL_API_KEY="$MODEL_API_KEY" \
    -e MODEL_NAME="$MODEL_NAME" \
    -v "$INPUT_DIR:/input:ro" \
    -v "$OUTPUT_DIR:/output:rw" \
    -v "$LOGS_DIR:/logs:rw" \
    "$IMAGE_TAG"

echo "=== Run complete ==="

# ─── 评分 ─────────────────────────────────────────────
GOLD_DIR="$(dirname "$INPUT_DIR")/output"
if [[ -d "$GOLD_DIR" ]]; then
    echo "=== Scoring ==="
    python scripts/score_predictions.py \
        --pred "$OUTPUT_DIR" \
        --gold "$GOLD_DIR" \
        --lambda 0.1 \
        --output "$LOGS_DIR/scores.json" \
        --verbose
fi
```

### 9.2.2 `scripts/eval_local_offline.sh`（压测用）

真正模拟 `--network=none`，配合本地 mock vLLM 服务：

```bash
#!/usr/bin/env bash
set -euo pipefail
# 前置：本机开着 mock vLLM 在 host.docker.internal:8000

docker run --rm \
    --network=none \
    --cpus=12 --memory=10g \
    -e MODEL_API_URL="http://host.docker.internal:8000/v1" \
    -e MODEL_API_KEY="sk-mock" \
    -e MODEL_NAME="qwen3.5-35b-a3b" \
    --add-host=host.docker.internal:host-gateway \
    -v "$PWD/data/public/input:/input:ro" \
    -v "$PWD/artifacts/eval_output_offline:/output:rw" \
    -v "$PWD/artifacts/eval_logs_offline:/logs:rw" \
    dabench:dev
```

**注意**：`--network=none` 下 `--add-host=host.docker.internal:host-gateway` 无效；真正封网测试需先用 `--network=bridge` + iptables 白名单，或切换到有宿主网络的 mock server 放 `localhost` 然后用 `host` network mode（Linux 独占）。Windows/Docker Desktop 下较麻烦，作为**可选验证**即可，非硬性。

## 9.3 提交前检查脚本

### 9.3.1 `scripts/pre_submit_check.sh`

每次提交前强制跑，任何一步失败阻塞 submit：

```bash
#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:?Usage: pre_submit_check.sh <version_number>}"
PREV_SCORES="artifacts/submissions/v$((VERSION-1))_scores.json"
NEW_SCORES="artifacts/submissions/v${VERSION}_scores.json"

echo "=== Step 1: ruff check ==="
uv run ruff check src/ tests/

echo "=== Step 2: pytest ==="
uv run pytest tests/ -q --tb=short

echo "=== Step 3: build image ==="
docker build -t "dabench:v${VERSION}" -f docker/Dockerfile .

echo "=== Step 4: image size check ==="
IMG_SIZE_GB=$(docker image inspect "dabench:v${VERSION}" \
    --format='{{.Size}}' | awk '{printf "%.2f", $1/1024/1024/1024}')
echo "Image size: ${IMG_SIZE_GB} GB"
if (( $(echo "$IMG_SIZE_GB > 9.0" | bc -l) )); then
    echo "FAIL: image > 9 GB, too close to 10 GB limit"
    exit 1
fi

echo "=== Step 5: run on demo ==="
IMAGE_TAG="dabench:v${VERSION}" bash scripts/eval_local.sh

echo "=== Step 6: score ==="
python scripts/score_predictions.py \
    --pred artifacts/eval_output \
    --gold data/public/output \
    --lambda 0.1 \
    --output "$NEW_SCORES" \
    ${PREV_SCORES:+--fail-on-regression "$PREV_SCORES"}

echo "=== Step 7: tar.gz size check ==="
mkdir -p artifacts/submissions
docker save "dabench:v${VERSION}" | gzip > "artifacts/submissions/dabench_v${VERSION}.tar.gz"
TGZ_SIZE_GB=$(du -b "artifacts/submissions/dabench_v${VERSION}.tar.gz" | awk '{printf "%.2f", $1/1024/1024/1024}')
echo "Archive size: ${TGZ_SIZE_GB} GB"
if (( $(echo "$TGZ_SIZE_GB > 10.0" | bc -l) )); then
    echo "FAIL: tar.gz > 10 GB limit"
    exit 1
fi

echo "=== All checks passed ==="
echo "Next steps:"
echo "  1. Rename archive: mv artifacts/submissions/dabench_v${VERSION}.tar.gz artifacts/submissions/<team_id>_v${VERSION}.tar.gz"
echo "  2. Upload to Google Drive"
echo "  3. bash scripts/submit.sh <team_id> ${VERSION} '<gdrive_link>'"
```

### 9.3.2 `scripts/submit.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

TEAM_ID="${1:?team_id required (e.g., team0042)}"
VERSION="${2:?version required (e.g., 1)}"
GDRIVE_LINK="${3:?gdrive link required}"

ARCHIVE="artifacts/submissions/${TEAM_ID}_v${VERSION}.tar.gz"
[[ -f "$ARCHIVE" ]] || { echo "FAIL: $ARCHIVE not found"; exit 2; }

SUBJECT="[KDDCup2026 Data Agents] Submission - ${TEAM_ID} - v${VERSION}"
BODY_FILE="artifacts/submissions/${TEAM_ID}_v${VERSION}_email.txt"

cat > "$BODY_FILE" <<EOF
Team ID: ${TEAM_ID}
Version: v${VERSION}
Submission date: $(date -u +%Y-%m-%dT%H:%M:%SZ)
Google Drive link: ${GDRIVE_LINK}

Archive: ${TEAM_ID}_v${VERSION}.tar.gz (size: $(du -h "$ARCHIVE" | awk '{print $1}'))
SHA256: $(sha256sum "$ARCHIVE" | awk '{print $1}')

Change summary:
$(cat CHANGELOG.md | head -30)

—
Submitted by ready2devallnight@gmail.com
EOF

echo "=== Email draft ready ==="
echo "To: kddcup@hkust-gz.edu.cn"
echo "Subject: $SUBJECT"
echo ""
echo "Body saved to: $BODY_FILE"
echo ""
echo "Now manually:"
echo "  1. Open your mail client"
echo "  2. Paste subject & body"
echo "  3. Send"
echo "  4. Record in docs/submission_log.md"
```

## 9.4 构建缓存与加速

### 9.4.1 BuildKit 缓存

```bash
export DOCKER_BUILDKIT=1
docker build \
    --cache-from dabench:dev \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    -t dabench:dev \
    -f docker/Dockerfile .
```

### 9.4.2 国内镜像源加速

`docker/pip.conf`（build 阶段 COPY 到容器）：

```ini
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/
trusted-host = mirrors.aliyun.com
timeout = 120
```

Dockerfile 中：
```dockerfile
COPY docker/pip.conf /etc/pip.conf
RUN pip install --no-cache-dir ...
```

构建速度从 8-15 分钟降到 3-5 分钟。

## 9.5 镜像瘦身技巧（若接近 10 GB）

按影响力从大到小排序：

1. **用 `python:3.11-slim-bookworm` 而非 `python:3.11`**（省 700 MB+）——已选
2. **多阶段构建，只拷 site-packages**（省 build tools 约 400 MB）——已选
3. **`apt-get clean && rm -rf /var/lib/apt/lists/*`** 每个 RUN 末尾——已选
4. **删除 `__pycache__/`**：`find /usr/local/lib -name __pycache__ -exec rm -rf {} +`
5. **考虑去掉 paddleocr**（500 MB）——已弃用
6. **sentence-transformers 模型用 INT8 量化版**（400→100 MB）
7. **pandas 有 100+ MB，若仅用 polars 可去掉**
8. **极限优化：pandas 只装 core，去 `pandas[all]`**

## 9.6 启动时自检（main.py 首步）

```python
def main() -> int:
    # 环境变量硬性要求检查
    missing = [k for k in ("MODEL_API_URL", "MODEL_API_KEY", "MODEL_NAME") if not os.environ.get(k)]
    if missing:
        sys.stderr.write(f"FATAL: missing env vars: {missing}\n")
        return 78   # EX_CONFIG

    # 挂载点检查
    for p in (Path("/input"), Path("/output"), Path("/logs")):
        if not p.is_dir():
            sys.stderr.write(f"FATAL: {p} not mounted\n")
            return 78

    # 写权限
    try:
        (Path("/output") / ".writable_probe").touch()
        (Path("/output") / ".writable_probe").unlink()
    except OSError as e:
        sys.stderr.write(f"FATAL: /output not writable: {e}\n")
        return 73   # EX_CANTCREAT

    # LLM 连通性（仅首次）
    try:
        llm = LLMClient.from_env_or_config(...)
        llm.complete([ChatMessage(role="user", content="hi")], max_tokens=4)
    except Exception as e:
        sys.stderr.write(f"FATAL: LLM probe failed: {e}\n")
        return 69   # EX_UNAVAILABLE

    # … 正式工作 …
```

## 9.7 镜像命名、版本、归档

| 阶段 | 镜像 tag | tar.gz 文件名 |
|---|---|---|
| 开发迭代 | `dabench:dev` | — |
| 提交 v1 | `team<NNNN>:v1` | `team<NNNN>_v1.tar.gz` |
| 提交 vN | `team<NNNN>:v<N>` | `team<NNNN>_v<N>.tar.gz` |
| 紧急回滚 | 保留上一版本 tag 不动 | — |

**归档政策**：所有提交过的 tar.gz 永久保留在 `artifacts/submissions/`（.gitignore）

## 9.8 Windows WSL2 + Docker Desktop 特别说明

### 9.8.1 `.wslconfig` 建议

`C:\Users\16321\.wslconfig`：
```ini
[wsl2]
memory=12GB
processors=20
swap=8GB
localhostForwarding=true
```
之后 PowerShell 跑 `wsl --shutdown` 重启生效。

### 9.8.2 构建性能

- 源码放在 **WSL 文件系统**（`/home/<user>/projects/kdd/`）而非 `/mnt/e/KDD比赛/`，构建快 5-10x（NTFS 跨界 I/O 慢）
- 但源码编辑仍可在 Windows IDE 里通过 `\\wsl$\Ubuntu\home\...` 路径
- 若嫌麻烦，直接在 E:\KDD比赛\ 也能跑，只是慢

## 9.9 Git 与镜像不变性

**提交流程必须保证**：
```
git commit → git tag v<N> → docker build → docker save → 文件名含 v<N>
```
tag 必须推到 remote，保证任何时候可回溯到该镜像对应的源码。
若 pre_submit_check.sh 失败，**不 tag，不打镜像**。
