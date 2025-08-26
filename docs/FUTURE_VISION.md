# 🔮 未来技术计划（可执行）

本计划基于当前仓库的功能与数据形态（已备份的微信公众号文章：HTML、Markdown、meta.json、images/），目标是在不增加维护复杂度的前提下，逐步解锁“可分析、可搜索、可对话、可扩展”的能力。

本项目不仅是一个微信公众号文章备份工具，更是个人知识资产的基础设施。基于备份的400+篇个人文章，我们规划了从数据收集到知识应用的完整生态。

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

## 阶段 1：语料索引与基础统计（1-2 周）

- 目标：可快速统计/检索语料，支持后续可视化与分析。
- 可交付物：
  1) `scripts/build_index.py`：生成 `index.parquet`（每篇一行：id、date、title、year、tokens、tags、path）。
  2) `scripts/stats_basic.py`：输出 `reports/basic_stats.md`（篇数/年份分布、平均字数、图片数等）。
- 验收标准：
  - 本地一次性全量跑完时间 < 3 分钟（以 400+ 篇为基准）；
  - 生成的 Parquet 可被 `pandas` 直接加载，字段完整。
- 工具建议：`pandas`, `pyarrow`, `bs4`, `tqdm`。

---

## 阶段 2：嵌入与向量检索（RAG 基础）（1-2 周）

- 目标：构建轻量可复用的段落级向量检索，支撑语义搜索与问答。
- 可交付物：
  1) `scripts/embed_corpus.py`：将 Markdown 切块（按标题/段落/长度阈值）→ 生成 `embeddings.faiss` 与 `chunks.parquet`；
  2) `scripts/rag_cli.py`：命令行检索与 Top-K 段落展示，支持 query→返回片段+来源路径。
- 验收标准：
  - 常见问题（如“某主题/某旅行/某关键词”）在 Top-5 命中率令人满意；
  - 总体嵌入构建时长 < 10 分钟；
  - 重复运行不产生破坏性结果（可幂等）。
- 工具建议：`sentence-transformers`（如 `all-MiniLM-L6-v2`）、`faiss-cpu`、`langchain/textsplitter`。
- 示例（嵌入构建）：
  ```bash
  python scripts/embed_corpus.py --input Wechat-Backup --out-dir artifacts/
  ```

---

## 阶段 3：对话原型（本地 RAG Chat）（1-2 周）

- 目标：基于向量检索搭建本地命令行/简单 Web 的对话原型，回答与个人语料相关的问题并给出出处。
- 可交付物：
  1) `app/rag_server.py`：FastAPI 服务，接口 `POST /ask {query}` 返回答案+引用段落；
  2) `app/ui_simple.py`：Streamlit 或 Gradio 简易界面；
  3) `docs/RAG_USAGE.md`：部署与使用指南（本地/离线优先）。
- 验收标准：
  - 典型问题平均响应 < 2s（CPU 环境）；
  - 每条回答至少附 2-3 条文内引用（路径+行号/片段摘要）；
  - 支持中文问答，明确“无法回答时”给出友好提示。
- 工具建议：`fastapi`, `uvicorn`, `streamlit`/`gradio`, `openai-compatible` 或 `ollama`（本地 LLM）。
- 示例（本地服务）：
  ```bash
  uvicorn app.rag_server:app --port 8001 --reload
  python app/ui_simple.py
  ```

---

## 阶段 4：个性化微调（LoRA 轻量适配）（2-3 周）

- 目标：在保护隐私的前提下，用少量高质量样本对本地小模型进行风格与知识微调，提升回答的“你味道”。
- 可交付物：
  1) `data/sft/`：对话/摘要/改写任务的高质量样本（人工/半自动生成，≥1k 对话轮次）；
  2) `train/lora_finetune.py`：基于 `Llama-3-8B-Instruct` 或同量级模型的 LoRA 训练脚本；
  3) `artifacts/adapters/`：LoRA 适配器权重（本地保存，不上传公库）。
- 验收标准：
  - 内部评测集（50-100 条）在风格一致性、引用准确性上优于基线（无 LoRA）；
  - 推理延迟变化可接受（CPU/GPU 下均衡）。
- 工具建议：`transformers`, `peft`, `trl`, `bitsandbytes`（可选量化）。
- 安全与合规：
  - 严禁上传个人原始语料与微调样本到公共平台；
  - 提供 `scripts/redact.py` 对敏感实体（姓名/地点/证件号等）做脱敏可选开关。

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

# 1. 基础统计
python scripts/stats_basic.py --index artifacts/index.parquet --out reports/basic_stats.md

# 2. 构建嵌入与检索
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
