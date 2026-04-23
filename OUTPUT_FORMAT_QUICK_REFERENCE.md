# Output Format Quick Reference

## TL;DR: Two Output Modes

### Combined (默认 / Default)
**一个JSON包含全部6个阶段**

```bash
# 单个例句
& python scripts/run_single.py --format combined
# → outputs/single_result.json (所有阶段在一个文件)

# 批处理
& python scripts/run_batch.py --format combined
# → outputs/batch_results.jsonl (一行一个完整结果)
```

### Staged (新增)
**每个阶段一个JSON文件 + 元数据**

```bash
# 单个例句
& python scripts/run_single.py --format staged
# → outputs/single_result/ (7个JSON文件)
#   ├─ single_metadata.json      (文本、语言、记号、验证)
#   ├─ single_phoneme.json       (音节与音位)
#   ├─ single_morpheme.json      (词素分割)
#   ├─ single_lexeme.json        (引理化 + 音译)
#   ├─ single_semantic.json      (语义类型/角色/分类)
#   ├─ single_relations.json     (提取的语义关系)
#   └─ single_dag.json           (DAG节点 + 边 + 验证)

# 批处理
& python scripts/run_batch.py --format staged --max-records 5
# → outputs/batch_results/ (每个例句一个子目录，每个7个文件)
```

---

## 何时选择哪个格式？

| 场景 | 推荐 |
|------|------|
| **快速查看完整结果** | Combined ✓ |
| **调试特定管道阶段** | Staged ✓ |
| **流式处理大批量** | Combined ✓ |
| **逐阶段重新处理** | Staged ✓ |
| **减少存储** | Combined ✓ |
| **模块化处理流程** | Staged ✓ |

---

## 文件结构对比

### Combined Mode
```
outputs/
  ├─ single_result.json          ← 1 file (all stages)
  ├─ dev_results.jsonl           ← 1 JSONL file (batch)
  └─ batch_results.jsonl         ← 1 JSONL file (Aristotle processing)
```

### Staged Mode  
```
outputs/
  ├─ single_result/              ← 1 directory
  │  ├─ single_metadata.json
  │  ├─ single_phoneme.json
  │  ├─ single_morpheme.json
  │  ├─ single_lexeme.json
  │  ├─ single_semantic.json
  │  ├─ single_relations.json
  │  └─ single_dag.json
  │
  ├─ dev_results_staged/         ← 1 directory (6 subdirs)
  │  ├─ summary.json             ← Index of all examples
  │  ├─ d1/
  │  │  └─ (7 files)
  │  ├─ d2/
  │  │  └─ (7 files)
  │  └─ ... (6 examples total)
  │
  └─ batch_results/              ← 1 directory (N subdirs)
     ├─ summary.json
     ├─ result_1/
     │  └─ (7 files)
     └─ ... (N results)
```

---

## Python 使用示例

### Combined: Load everything
```python
import json

# Single result
with open("outputs/single_result.json") as f:
    result = json.load(f)
    lemmas = [tok["lemma"] for tok in result["lexeme_layer"]]
    relations = result["relations"]
    dag = result["dag"]
```

### Combined: Stream batch (JSONL)
```python
with open("outputs/batch_results.jsonl") as f:
    for line in f:
        result = json.loads(line)
        text = result["text"]
        tokens = result["tokens"]
        relations = result["relations"]
```

### Staged: Load by stage
```python
import json
from pathlib import Path

result_dir = Path("outputs/single_result")

# Load only what you need
metadata = json.load(open(result_dir / "single_metadata.json"))
phonemes = json.load(open(result_dir / "single_phoneme.json"))
morphemes = json.load(open(result_dir / "single_morpheme.json"))
dag = json.load(open(result_dir / "single_dag.json"))

# Skip other stages if not needed
```

### Staged: Iterate batch with index
```python
import json
from pathlib import Path

batch_dir = Path("outputs/batch_results")
summary = json.load(open(batch_dir / "summary.json"))

for item in summary:
    example_id = item["id"]
    example_dir = batch_dir / example_id
    
    # Process each example independently
    metadata = json.load(open(example_dir / "output_metadata.json"))
    relations = json.load(open(example_dir / "output_relations.json"))
    print(f"{example_id}: {len(relations)} relations extracted")
```

---

## 命令行参数汇总

### run_single.py
```bash
& python scripts/run_single.py
    [--format {combined|staged}]     # 默认: combined
    [--output-dir PATH]              # 默认: outputs/
```

### run_batch.py
```bash
& python scripts/run_batch.py
    [--input PATH]                   # 输入 JSONL (默认: Aristotle.txt)
    [--output PATH]                  # 输出路径 (默认: outputs/batch_results)
    [--format {combined|staged}]     # 默认: combined
    [--max-records N]                # 处理记录数上限 (默认: 20)
```

---

## 实际用例对比

### 用例1: 快速检验单个例句
```bash
# Combined ✓ 简单直接
& python scripts/run_single.py --format combined
# 打开 outputs/single_result.json，看所有结果
```

### 用例2: 调试词元化失败
```bash
# Staged ✓ 快速定位问题
& python scripts/run_single.py --format staged
# 打开 outputs/single_result/single_lexeme.json，检查引理化错误
```

### 用例3: 批量处理古希腊文本
```bash
# Combined ✓ 流式处理效率高
& python scripts/run_batch.py --format combined --max-records 100
# 用 jq 或 Python 脚本逐行流式处理
```

### 用例4: 逐阶段重新分析批量结果
```bash
# Staged ✓ 便于模块化处理
& python scripts/run_batch.py --format staged --max-records 20
# 可以只重新处理关系抽取或 DAG 验证，而不再跑前面阶段
```

---

完整文档见 [OUTPUT_FORMATS.md](OUTPUT_FORMATS.md)
