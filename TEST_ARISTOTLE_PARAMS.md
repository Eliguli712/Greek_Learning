test_aristotle.py — 参数参考

════════════════════════════════════════════════════════════════════════════
参数列表
════════════════════════════════════════════════════════════════════════════

1. --sentences N
   处理前 N 个句子 (默认: 5)
   
   & python test_aristotle.py --sentences 3
   & python test_aristotle.py --sentences 20

2. --format {combined|staged}
   输出格式 (默认: combined)
   
   combined: 单一 JSONL 文件（所有6阶段合并）
   staged:   每个例句一个子目录，共7个JSON（元数据 + 6阶段）
   
   & python test_aristotle.py --format combined    # ← 默认
   & python test_aristotle.py --format staged      # ← 分阶段输出

3. --output-dir PATH
   输出目录 (默认: outputs/aristotle_test)
   
   & python test_aristotle.py --output-dir outputs/my_results
   & python test_aristotle.py --output-dir D:\custom\path

4. --language LANG
   语言标签 (默认: ancient_greek)
   
   & python test_aristotle.py --language ancient_greek  # ← 默认

5. --verbose
   详细输出：显示全文而非截断
   
   & python test_aristotle.py --verbose             # ← 显示完整句子文本

════════════════════════════════════════════════════════════════════════════
常用组合
════════════════════════════════════════════════════════════════════════════

1. 处理 3 个句子，分阶段输出（最常见用例）
   & python test_aristotle.py --sentences 3 --format staged
   → 输出: outputs/aristotle_test/aristotle_results_staged/
   
2. 处理 10 个句子，合并输出到自定义目录
   & python test_aristotle.py --sentences 10 --format combined --output-dir outputs/my_run
   → 输出: outputs/my_run/aristotle_results.jsonl
   
3. 处理 5 个句子，详细输出 + 分阶段
   & python test_aristotle.py --sentences 5 --format staged --verbose
   → 每个句子显示完整文本 + 7 个文件/例句
   
4. 快速查看（仅 2 个句子）
   & python test_aristotle.py --sentences 2
   → 输出: outputs/aristotle_test/aristotle_results.jsonl (简洁格式)

════════════════════════════════════════════════════════════════════════════
输出结构对比
════════════════════════════════════════════════════════════════════════════

COMBINED (合并格式):
  outputs/aristotle_test/
  └─ aristotle_results.jsonl          ← 1 行 = 1 个完整结果 (所有6阶段)

STAGED (分阶段格式):
  outputs/aristotle_test/
  └─ aristotle_results_staged/
     ├─ summary.json                   ← 索引
     ├─ s1/                            ← 第1个句子
     │  ├─ output_metadata.json
     │  ├─ output_phoneme.json
     │  ├─ output_morpheme.json
     │  ├─ output_lexeme.json
     │  ├─ output_semantic.json
     │  ├─ output_relations.json
     │  └─ output_dag.json
     ├─ s2/                            ← 第2个句子
     │  └─ (7 个文件)
     └─ ...

════════════════════════════════════════════════════════════════════════════
快速参考
════════════════════════════════════════════════════════════════════════════

参数名              类型        默认值                   描述
──────────────────────────────────────────────────────────────────────────
--sentences         int         5                       处理句子数
--format            enum        combined                combined | staged
--output-dir        path        outputs/aristotle_test  输出目录
--language          str         ancient_greek           语言标签
--verbose           flag        False                   显示详细信息

════════════════════════════════════════════════════════════════════════════
Python 代码示例
════════════════════════════════════════════════════════════════════════════

# 使用 combined 格式读取
import json
with open('outputs/aristotle_test/aristotle_results.jsonl') as f:
    for line in f:
        result = json.loads(line)
        print(result['text'])
        print(result['relations'])

# 使用 staged 格式读取
from pathlib import Path
import json

results_dir = Path('outputs/aristotle_test/aristotle_results_staged')
for example_dir in sorted(results_dir.glob('s*')):
    metadata = json.load(open(example_dir / 'output_metadata.json'))
    dag = json.load(open(example_dir / 'output_dag.json'))
    relations = json.load(open(example_dir / 'output_relations.json'))
    
    print(f"{example_dir.name}: {len(relations)} relations")
