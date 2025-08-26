# 🔮 未来技术计划（由浅入深｜强获得感优先）

本计划围绕“个人公众号长期写作”的真实场景，按“先可见收益，再逐步深入”的路径推进：
1) 即刻获得感：词频与年度盘点、主题聚类、代表作榜单；
2) 认知与成长：个人与社会关系、价值观与情绪、阶段性关注主题；
3) NLP 与检索：语义检索、RAG 问答；
4) 个性化模型：小样本/LoRA 轻量适配。

数据形态：HTML / Markdown / meta.json / images；范围：400+ 篇已备份文章。

## 阶段 0：数据规范与一致性（1 周）

- 目标：保证所有既有与后续备份的数据结构一致，便于后续分析与建模。
- 可交付物：
  1) `scripts/normalize.py`：批量规范化 `meta.json` 字段（统一 `title/date/url/tags` 等键名、日期格式 ISO-8601）。
  2) `docs/DATA_SCHEMA.md`：定义目录结构与字段规范。
- 验收标准：
  - 随机抽样 50 篇文章，字段齐全且通过 `jsonschema` 校验；
  - 规范化脚本幂等（多次执行结果一致）。
- 工具建议：`python-jsonschema`, `dateparser`。

---

## 阶段 1：即刻获得感（1-2 周）

- 目标：一周内看到“有收获”的可视化与榜单，形成年度与主题的“第一印象”。
- 可交付物：
  1) 词频与关键词：`reports/wordfreq_top.md`（全量/年度Top-N、停用词与自定义词典支持）
  2) 年度盘点：`reports/yearly_overview.md`（年度篇数、关键词云、代表作、情绪/语气分布）
  3) 主题初探：`reports/topics_quick.md`（KeyBERT/TF-IDF 主题词 + 示例段落）
  4) 代表作榜单：`reports/best_of.md`（按互动指标/自评标签/长度权重合成评分）
  5) 可视化：`reports/charts_yearly.html`（年度趋势、词云、主题力导向图）
- 验收标准：
  - 一次性全量统计 < 5 分钟（400+ 篇基准）；
  - 每份报告含“摘要 + 图表 + 可复现命令”；
  - 关键词/榜单能被作者主观认可（回顾 10 篇样例）。
- 工具建议：`pandas`, `jieba`/`pkuseg`, `keybert`, `matplotlib/altair/plotly`, `wordcloud`。
- 示例命令：
  ```bash
  python scripts/stats_wordfreq.py --root Wechat-Backup --out reports/
  python scripts/stats_yearly.py   --index artifacts/index.parquet --out reports/
  python scripts/topics_quick.py   --root Wechat-Backup --out reports/
  ```

---

## 阶段 2：认知与成长挖掘（2-3 周）

- 目标：从文本中提炼“个人与社会关系、价值观与情绪、阶段性主题”的结构化画像。
- 可交付物：
  1) 人物与关系：`reports/people_network.html`（人名/昵称共现图，社交网络中心度）
  2) 地点与场景：`reports/places_timeline.html`（地名识别 + 时间线）
  3) 情绪与语气：`reports/emotion_trends.html`（情绪/极性随时间与主题变化）
  4) 价值观关键词：`reports/values_keywords.md`（“选择/反思/责任/边界”等语义类簇）
  5) 阶段性主题：`reports/stages.md`（按年份/人生节点聚类出的主题变化）
- 验收标准：
  - 共现图与时间线能呈现“重要人物/场景”的直观印象；
  - 情绪/价值观的抽取能复核 30 篇抽样；
  - 阶段性主题与作者自评高度一致。
- 工具建议：`spaCy`/`HanLP`（NER）、`jieba`、`textblob-cn`/`snownlp`（情感）、`networkx`、`altair/plotly`。
- 示例命令：
  ```bash
  python scripts/people_network.py --index artifacts/index.parquet --out reports/
  python scripts/emotion_trends.py --index artifacts/index.parquet --out reports/
  ```

---

## 阶段 3：嵌入与语义检索（RAG 基础）（1-2 周）

- 目标：让“问题→出处段落”成为日常可用的工具，支撑后续对话原型。
- 可交付物：
  1) `scripts/embed_corpus.py`：段落切块 + 嵌入 → `embeddings.faiss` / `chunks.parquet`
  2) `scripts/semantic_search.py`：命令行语义检索 + Top-K 引用 + 高亮
  3) `docs/RAG_USAGE.md`：检索使用与注意事项
- 验收标准：
  - Top-5 段落可用率高（作者主观打分 ≥ 4/5）；
  - 召回片段附带来源 `md/html` 路径；
  - 构建时间 < 10 分钟，可幂等。
- 工具建议：`sentence-transformers`, `faiss-cpu`, `langchain/textsplitter`。
- 示例命令：
  ```bash
  python scripts/embed_corpus.py      --input Wechat-Backup --out-dir artifacts/
  python scripts/semantic_search.py   --query "关于选择与责任的讨论"
  ```

---

## 阶段 4：对话原型（本地 RAG Chat）（1-2 周）

- 目标：在本地提供简洁对话界面，问题→检索→综合回答，并附引用。
- 可交付物：
  1) `app/rag_server.py`：FastAPI 接口 `POST /ask {query}` 返回答案+引用段落
  2) `app/ui_simple.py`：Streamlit/Gradio 本地 UI
  3) `docs/RAG_USAGE.md`：对话原型使用说明
- 验收标准：
  - 典型问题响应 < 2s（CPU）；
  - 每条回答附 2-3 条引用；
  - 支持中文问答与“无法回答”提示。
- 工具建议：`fastapi`, `uvicorn`, `streamlit`/`gradio`，可对接本地 `ollama`。

---

## 阶段 5：个性化微调（LoRA 轻量适配）（2-3 周｜可选）

- 目标：在保护隐私前提下，用少量高质量样本对本地小模型做风格/知识适配。
- 可交付物：
  1) `data/sft/`：高质量样本（摘要、改写、问答），≥1k 轮
  2) `train/lora_finetune.py`：LoRA 训练脚本与配置
  3) `artifacts/adapters/`：适配器权重（本地保存，不上传）
- 验收标准：
  - 内部评测集在风格一致性、引用准确性上优于基线；
  - 延迟与资源成本可接受。
- 工具建议：`transformers`, `peft`, `trl`, `bitsandbytes`（量化可选）。
- 隐私与合规：与原计划相同（本地存储、脱敏脚本、不开源权重）。

---

## 阶段 5：分析报告与可视化（并行进行，1-2 周）

- 目标：将阶段 1/2 的统计与检索能力用于生成“洞察报告”。
- 可交付物：
  1) `scripts/analysis_topics.py`：LDA/KeyBERT 提取主题 → `reports/topics.md`；
  2) `scripts/analysis_style.py`：句式/标点/词频 → `reports/style.md`；
  3) `scripts/plot_dash.py`：Plotly/Altair 生成交互图表。
- 验收标准：
  - 报告可读、可复现，附上生成命令；
  - 图表可离线打开（HTML）。
- 工具建议：`keybert`, `scikit-learn`, `jieba`/`pkuseg`, `plotly`, `altair`。

---

## 基线评测与回归（全阶段贯穿）

- 设立小型评测集 `eval/`：包含检索命中（信息找回）、事实问答（引用准确）、风格改写（主观评分）。
- 每阶段完成后运行 `scripts/run_eval.py` 输出 `eval_report.md`，记录指标与样例。

---

## 资源与运行建议

- 硬件：CPU 即可完成 0-3 阶段；4 阶段建议 12GB+ 显存（或使用 QLoRA/4-bit 量化）。
- 运行环境：Python 3.9+；建议 `venv`/`conda` 隔离；Windows 与 macOS/Linux 均可。
- 任务编排：可用 `make`/`taskfile` 组织常用命令（可选）。

---

## 隐私与合规

- 所有数据与模型权重默认仅存储在本地 `artifacts/`；
- 严禁将原始语料、微调样本、适配器权重上传到公共仓库；
- 提供 `COPYING_LOCAL.md` 提醒使用者遵循 MIT（代码）与 CC BY-NC-SA 4.0（内容）。

---

## 快速命令索引（示例）

```bash
# 0. 规范化与索引
python scripts/normalize.py --root Wechat-Backup
python scripts/build_index.py --root Wechat-Backup --out artifacts/index.parquet

# 1. 即刻获得感（统计/年度/主题）
python scripts/stats_basic.py --index artifacts/index.parquet --out reports/basic_stats.md
python scripts/stats_wordfreq.py --root Wechat-Backup --out reports/
python scripts/stats_yearly.py   --index artifacts/index.parquet --out reports/

# 2. 认知与成长挖掘 / 嵌入与检索
python scripts/embed_corpus.py --input Wechat-Backup --out-dir artifacts/
python scripts/rag_cli.py --query "周易相关的文章有哪些？"

# 3. 启动对话原型
uvicorn app.rag_server:app --port 8001 --reload
python app/ui_simple.py

# 4.（可选）LoRA 微调
python train/lora_finetune.py --config train/configs/lora_8b.yaml
```

---

> 相关：
> - 主 README 的“项目展望”摘要
> - `STATUS.md`（阶段性里程碑记录）
> - `docs/TROUBLESHOOTING.md`（遇到问题时）
