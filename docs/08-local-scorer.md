# 8. 本地评分器（Local Scorer）

> **目的**：按规则 §6 的语义复现官方评分逻辑，用于每次提交前的**对拍验证**。
> 不公开发布；仅内部使用。若组委会开放 reference scorer，直接对比我方实现是否一致。
> 位置：`scripts/score_predictions.py`（独立脚本，不进容器镜像）。

## 8.1 规则原文映射（§6）

| 规则条款 | 关键点 | 实现要求 |
|---|---|---|
| §6.2 | 列忽略列名；按列值**无序向量**匹配；支持重复列（同签名出现次数必须一致） | 列向量排序后哈希，形成"签名" |
| §6.3 | `Recall = Matched / Gold`；`Score = Recall − λ·(Extra / Predicted)`；下界 0 | λ 待定，先用 0.1 假设值；CLI 可调 |
| §6.4 | 鼓励覆盖、控制冗余、保持鲁棒 | 惩罚项对 Recall=0 的预测不额外加重（下界 0 保证） |
| §6.5 | 数值 round 2 位；时间 aware → UTC Z，naive → ISO；字符串 trim `\r\n`，大小写敏感 | 归一化函数复用 OutputWriter.normalize_cell，但以 gold 的处理为权威 |
| §6.6 | prediction.csv 缺文件 → 0；部分列不匹配 → 降分；冗余列 → 惩罚 | 缺文件单独处理；分数 clamp 到 [0, 1] |
| §6.7 | 每任务独立打分；总分 = 算术平均；同分按提交时间 | 本地只算分，不管排名 |

## 8.2 CLI 规格

```bash
python scripts/score_predictions.py \
  --pred artifacts/eval_output \
  --gold data/public/output \
  --lambda 0.1 \
  --output artifacts/scores.json \
  [--verbose] \
  [--fail-on-regression previous_scores.json]
```

| 参数 | 含义 |
|---|---|
| `--pred` | 预测目录，下含 `task_<id>/prediction.csv` |
| `--gold` | 标答目录，下含 `task_<id>/gold.csv` |
| `--lambda` | λ 值，默认 0.1 |
| `--output` | 结果 JSON |
| `--verbose` | 打印每任务每列的匹配/失配详情 |
| `--fail-on-regression` | 比对旧结果，任务 0→1 回退则非零退出 |

## 8.3 输出 JSON 格式

```json
{
  "meta": {
    "pred_dir": "artifacts/eval_output",
    "gold_dir": "data/public/output",
    "lambda": 0.1,
    "ts": "2026-04-24T06:12:15Z",
    "n_tasks_gold": 30,
    "n_tasks_pred": 28
  },
  "per_task": {
    "task_1": {
      "gold_columns": 3,
      "predicted_columns": 4,
      "matched_columns": 3,
      "extra_columns": 1,
      "recall": 1.0,
      "penalty": 0.025,
      "score": 0.975,
      "status": "scored",
      "mismatched_columns": [],
      "extra_column_fingerprints": ["sha256:ab12..."]
    },
    "task_2": {
      "status": "missing_prediction",
      "score": 0.0
    },
    "task_3": {
      "gold_columns": 2,
      "predicted_columns": 2,
      "matched_columns": 1,
      "extra_columns": 1,
      "recall": 0.5,
      "penalty": 0.05,
      "score": 0.45,
      "status": "partial_match",
      "mismatched_columns": [{"gold_fingerprint": "sha256:cd34...", "reason": "values differ"}]
    }
  },
  "aggregate": {
    "total_score": 0.58,
    "mean_recall": 0.72,
    "tasks_scored_nonzero": 19,
    "tasks_missing": 2,
    "by_difficulty": {
      "easy": {"count": 8, "mean_score": 0.91},
      "medium": {"count": 10, "mean_score": 0.62},
      "hard": {"count": 8, "mean_score": 0.31},
      "extreme": {"count": 4, "mean_score": 0.12}
    }
  }
}
```

## 8.4 列签名算法

给定 CSV 某一列的值向量，生成**签名**用于跨 prediction/gold 的无序匹配：

```python
def column_signature(values: list[Any]) -> str:
    """
    1. 归一化每个 cell：normalize_cell（同 OutputWriter）
    2. 排序 normalized values（str 比较）
    3. 用 '\x1f' 连接（ASCII unit separator，避免与数据冲突）
    4. SHA-256 hex
    """
    normalized = [normalize_cell(v) for v in values]
    normalized.sort()
    joined = "\x1f".join(normalized)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()
```

**重复列处理**：用 `collections.Counter[signature]` 记录各签名出现次数。匹配时 `min(pred_count, gold_count)` 计入 matched。

## 8.5 归一化（与 §6.5 完全对齐）

```python
def normalize_cell(v: Any) -> str:
    # 空值
    if v is None or v == "":
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    if isinstance(v, str) and v.strip().lower() in {"null", "nan", "n/a"}:
        return ""

    # 数值：round 2 位后字符串化
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, Decimal, np.floating)):
        return f"{round(float(v), 2):.2f}"

    # 日期
    if isinstance(v, (pd.Timestamp, datetime)):
        if getattr(v, "tzinfo", None) is not None:
            return v.astimezone(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return v.isoformat()

    # 字符串尝试转数字
    s = str(v).strip().replace("\r\n", "\n")
    try:
        # "1,200.50" → "1200.50"（保留去千分位逻辑？争议点）
        cleaned = s.replace(",", "") if re.fullmatch(r"-?[\d,]+(\.\d+)?", s) else s
        f = float(cleaned)
        return f"{round(f, 2):.2f}"
    except (ValueError, TypeError):
        pass

    # 字符串尝试转日期
    for fmt in ISO_DATE_FORMATS:
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is not None:
                return dt.astimezone(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            return dt.isoformat()
        except ValueError:
            continue

    return s   # 保留原文（大小写敏感）
```

**⚠️ 风险点**：千分位分隔符（"1,200"）是否该去掉？规则没明说。**默认去掉**——更宽容的匹配；但若组委会区分，可能导致本地高估。CLI 加 `--strict-commas` 开关让用户选。

## 8.6 评分流程

```python
def score_task(pred_csv: Path, gold_csv: Path, lambda_: float) -> TaskScore:
    if not pred_csv.exists():
        return TaskScore(score=0.0, status="missing_prediction", **zeros)
    
    pred_df = pd.read_csv(pred_csv, dtype=str, keep_default_na=False)
    gold_df = pd.read_csv(gold_csv, dtype=str, keep_default_na=False)
    
    pred_sigs = Counter(column_signature(pred_df[c]) for c in pred_df.columns)
    gold_sigs = Counter(column_signature(gold_df[c]) for c in gold_df.columns)
    
    matched = sum(min(pred_sigs[s], gold_sigs[s]) for s in gold_sigs)
    gold_cols = sum(gold_sigs.values())
    pred_cols = sum(pred_sigs.values())
    extra = max(0, pred_cols - matched)
    
    recall = matched / gold_cols if gold_cols else 0.0
    penalty = lambda_ * (extra / pred_cols) if pred_cols else 0.0
    score = max(0.0, recall - penalty)
    
    return TaskScore(
        score=score,
        recall=recall,
        penalty=penalty,
        matched_columns=matched,
        gold_columns=gold_cols,
        predicted_columns=pred_cols,
        extra_columns=extra,
        status="scored" if recall == 1.0 and extra == 0
               else "partial_match" if matched > 0
               else "no_match",
    )
```

## 8.7 聚合

```python
def aggregate(task_scores: dict[str, TaskScore], task_meta: dict[str, TaskMeta]) -> Aggregate:
    total = sum(s.score for s in task_scores.values()) / len(task_scores)
    
    by_diff = defaultdict(list)
    for tid, score in task_scores.items():
        by_diff[task_meta[tid].difficulty].append(score.score)
    
    return Aggregate(
        total_score=total,
        mean_recall=sum(s.recall for s in task_scores.values()) / len(task_scores),
        tasks_scored_nonzero=sum(1 for s in task_scores.values() if s.score > 0),
        tasks_missing=sum(1 for s in task_scores.values() if s.status == "missing_prediction"),
        by_difficulty={d: {
            "count": len(scores),
            "mean_score": sum(scores) / len(scores),
        } for d, scores in by_diff.items()},
    )
```

## 8.8 Regression 检测

```python
def check_regression(current: dict, previous: Path) -> list[str]:
    """
    对比旧结果，找出：
    - 分数从 >0 降到 0 的任务（严重）
    - 分数下降 >0.2 的任务（警告）
    
    返回违规任务 ID 列表；非空时 CLI 非零退出（除非 --force）
    """
    prev = json.loads(previous.read_text())
    regressions = []
    for tid, cur in current["per_task"].items():
        if tid not in prev["per_task"]:
            continue
        prev_score = prev["per_task"][tid].get("score", 0)
        cur_score = cur.get("score", 0)
        if prev_score > 0 and cur_score == 0:
            regressions.append(f"{tid}: {prev_score:.2f} → 0 (SEVERE)")
        elif cur_score < prev_score - 0.2:
            regressions.append(f"{tid}: {prev_score:.2f} → {cur_score:.2f} (warn)")
    return regressions
```

## 8.9 与 starter kit 的差异

starter kit **不提供** scorer——`data/public/output/task_<id>/gold.csv` 是公开的，但比对逻辑需自行实现。本文件即此实现的规格。

## 8.10 单元测试清单

| 测试 case | 输入 | 期望 |
|---|---|---|
| 完美匹配 | pred == gold | score=1.0 |
| 冗余 1 列 | pred = gold + 1 col | score = 1 − λ·(1/(n+1)) |
| 缺 1 列 | pred 少 1 col | recall = (n-1)/n |
| 列值错 1 个 | pred 某列 1 个 cell 改变 | 该列签名变，matched -1 |
| 列顺序打乱 | pred 列顺序变 | score 不变 |
| 重复列 gold 有 2 | pred 有 1 重复列 | matched 对该签名 = 1（min） |
| 数字精度：`3.14159` vs `3.14` | round(3.14159,2)=3.14 | match |
| 数字精度：`3.145` vs `3.14` | round(3.145,2)=3.14(银行家? 一般=3.15) | **flag**——Python round 用银行家四舍五入；文档不确定评测端策略；需测 |
| 空值 `NaN` vs `""` | 归一化后都 = "" | match |
| 时间带时区 `2026-04-24T10:00+08` vs `2026-04-24T02:00Z` | 都 → "2026-04-24T02:00:00Z" | match |
| prediction.csv 缺失 | — | status=missing_prediction, score=0 |
| pred 为空表 | columns only, no rows | recall=0 |

## 8.11 待决问题

| # | 问题 | 临时策略 |
|---|---|---|
| S1 | λ 具体值（Q2 未答） | 0.1；公开 leaderboard 出结果后反推 |
| S2 | 千分位逗号是否算数字 | 默认算；`--strict-commas` 可关 |
| S3 | 科学记数法 "1e5" 是否等于 "100000"? | 归一化后都 → "100000.00"，应算等 |
| S4 | `round(3.145, 2)` Python = 3.14（银行家），Excel = 3.15。评测端哪种？ | 跑 demo 对拍，观察；flag 记录 |
| S5 | 字符串大小写：规则说"大小写敏感"；列名真的不参与？ | 代码实现按规则；若 demo 异常再改 |
| S6 | 字段内换行 `\n` 的处理 | 保留（规则未说要去） |

## 8.12 使用时机

| 阶段 | 用法 |
|---|---|
| 每次 `scripts/eval_local.sh` 跑完 | 自动调 scorer；输出到 `artifacts/scores.json` |
| 每次提交前 | `scripts/pre_submit_check.sh` 里强制调；regression → 阻塞提交 |
| M6 调优期 | 每次 prompt/模型改动手动调，对比 scores.json |
| 拿到组委会真实分数后 | 对比本地分数 vs leaderboard，反推 λ 和隐藏规则 |
