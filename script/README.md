# 微信公众账号内容管理系统

## 📱 第一步：公众号爬取 (WeChat Scraping)

这个脚本用于备份微信公众账号的已发布文章，包括：
- 文章HTML内容
- 文章Markdown格式
- 文章中的图片（本地化保存）
- 文章元数据（标题、发布时间、URL等）

## 使用方法

### 1. 配置环境

在项目根目录创建 `env.json` 文件，参考 `env.json.EXAMPLE`：

```json
{
  "COOKIE": "你的微信公众平台Cookie",
  "TOKEN": "发表记录页URL里的token",
  "WECHAT_ACCOUNT_NAME": "你的公众号名称",
  "COUNT": "20",
  "SLEEP_LIST": "2.5",
  "SLEEP_ART": "1.5",
  "IMG_SLEEP": "0.08"
}
```

### 2. 运行脚本

### 平台命令对照

- Windows PowerShell
```powershell
py -3 script\wx_publish_backup.py
```

- macOS / Linux（终端）
```bash
python3 script/wx_publish_backup.py
```

### 一键运行示例

```bash
# 克隆并进入项目
git clone https://github.com/xianyu564/scrape-my-wechat-official-account.git
cd scrape-my-wechat-official-account

# 安装依赖
pip install -r requirements.txt

# 复制示例配置并编辑
cp env.json.EXAMPLE env.json
# 用编辑器填入 COOKIE / TOKEN / WECHAT_ACCOUNT_NAME

# 运行
python3 script/wx_publish_backup.py
```

### 错误恢复建议（速查）

- 403 / 预检失败：刷新登录态，重新复制 Cookie 与 token；把 `COUNT` 暂时设为 `1`
- 频繁失败/限速：上调 `SLEEP_LIST` / `SLEEP_ART` / `IMG_SLEEP`，分多次运行
- HTML 空白或图片丢失：优先用本地静态服务器访问；确认 `images/` 已生成
- 断点续传：删除（或备份后清空）对应目录下的 `_state.json` 可全量重抓

## 运行流程

### 预检阶段
脚本首先会进行预检，验证Cookie和Token的有效性：

```
→ 正在预检 appmsgpublish 接口…
✅ 预检通过：publish_list=1，total_count=434
```

### 备份阶段
预检通过后，开始正式备份文章：

```
saved: 2024/2024-12-19_文章标题/article.html
saved: 2024/2024-12-18_另一篇文章/article.html
...
```

### 完成提示
所有文章备份完成后，会显示：

```
✅ 完成 → D:\Experiment\Elephenotype\scrape-my-wechat-official-account\script\backup_文不加点的张衔瑜
```

## 📈 第二步：爬取效率分析 (Scraping Efficiency Analysis)

`generate_clone_report.py` 脚本用于分析已备份文章的下载效率，生成详细的时间和大小报告。

- **报告内容**：
    - **每篇文章下载详情**：包含文章路径、下载结束时间、文章总大小（字节和MB），以及与上一篇文章下载完成时间的时间间隔（秒），以此衡量单篇文章的下载耗时。
    - **年度汇总**：统计每个年度的总下载文件大小（字节和MB）和总下载耗时（秒）。
- **输出格式**：同时生成 `克隆效率.json` 和 `克隆效率.md` 两种格式的报告。

### 运行方法

在项目根目录运行以下命令：
```bash
python scripts/generate_clone_report.py
```
如果你想指定不同的目标目录，可以使用 `--target_dir` 参数：
```bash
python scripts/generate_clone_report.py --target_dir "Wechat-Backup/你的其他文件夹"
```

## 📚 第三步：TOC生成 (TOC Generation)

### 内容目录组织系统

建立双重视图的内容组织系统：

#### 🎯 架构理念

1. **双重视图设计** 
   - **编年目录** (`目录.md`): 按时间顺序，提供完整的历史脉络和统计数据
   - **主题合集** (`合集.md`): 按内容主题，支持专题研究和深度阅读

2. **数据驱动的组织方法**
   - 基于文件系统扫描进行内容发现
   - 从实际文件内容提取元数据（字数、图片数、摘要）
   - 统计数据自动计算和验证

3. **多维度分类系统**
   - 时间维度：年份、季度、月份
   - 主题维度：读书、旅行、生活、学术、艺术等
   - 地理维度：城市、地区、国家标注
   - 内容维度：书籍、影视、人物引用

#### 🔧 Copilot指令使用

获取详细的TOC生成指令，请参考：[`copilot_discover_TOC.prompt.example.md`](copilot_discover_TOC.prompt.example.md)

该指令集包含四个阶段：
1. **内容发现与元数据提取**：递归扫描目录结构，提取文章元数据
2. **编年目录生成**：创建时间序列索引，包含编号和摘要  
3. **主题合集组织**：建立多维度分类系统
4. **质量控制与验证**：数据一致性检查和格式验证

## 📈 第四步：词云分析 (Wordcloud Analysis)

基于 `analysis/` 目录的完整语言学分析功能，提供科学化的内容语言学分析：

### 🔤 高级分词与分析
- **主要分词器**: pkuseg（通用模型，强大的中文分词）
- **备用分词器**: jieba（如果pkuseg不可用）
- **规范化**: 全角→半角，英文小写，保留技术术语
- **混合语言**: 智能处理机器学习、AIGC_2025等术语
- **单字过滤**: 保留有意义的中文单字（人/心/光）

### 📈 科学指标
- **Zipf定律**: 词频排序分析与R²验证
- **Heaps定律**: 词汇增长曲线 (V = K × n^β)
- **TF-IDF**: 与scikit-learn集成的预分词输入
- **词汇多样性**: TTR、词汇密度、内容词比例
- **年度对比**: 年度间词频变化分析

### 🎨 可视化输出
- **词云图**: CJK字体支持，科学配色方案，300 DPI
- **科学图表**: 四面板Zipf分析，带置信区间的Heaps曲线
- **对比图表**: 年度频率对比，增长可视化
- **可重现性**: 固定随机种子保证结果一致

### 🚀 使用方法

```bash
# 完整分析（两个阶段）
cd analysis/
python main.py

# 仅理论分析（设置 RUN_VISUALIZATION = False）
# 仅可视化（设置 RUN_ANALYSIS = False，需要阶段1的结果）
```

### 📊 输出文件

**分析结果**
- `out/summary.json` - 综合统计摘要
- `out/analysis_results.pkl` - 完整分析数据供第二阶段使用

**科学图表**
- `out/fig_zipf_panels.png` - 四面板Zipf分析
- `out/fig_heaps.png` - 带置信区间的Heaps定律
- `out/fig_yearly_comparison.png` - 年度词频对比
- `out/fig_growth.png` - 年度增长图表

**词云图**
- `out/cloud_overall.png` - 整体语料词云
- `out/cloud_YYYY.png` - 各年度词云 (2017-2025)
- `out/cloud_complete.png` - 完整数据集词云

📁 **备份位置**: 所有词云图备份至 [`../.github/assets/wordclouds/`](../.github/assets/wordclouds/) 以供保存和参考。

**报告**
- `out/report.md` - 完整语言学分析报告

更多详细配置和使用说明，请参考 [`../analysis/README.md`](../analysis/README.md)。

## 🚀 第五步：作者行动生成 (Author Action Generation)

将前四步的分析结果转化为作者可执行的具体行动建议，提升写作效率和自我认知：

### 💡 人机协作理念

本系统是一个**人机交互的反思性写作项目**，旨在：
- **提升记忆回忆效率**：通过结构化索引快速定位历史内容
- **增强写作模式自我认知**：通过数据分析识别个人写作偏好和发展轨迹
- **加强主题规划能力**：基于历史主题分布制定未来内容策略
- **提高合规风险意识**：通过内容分析预防潜在的合规问题

### 🎯 与个人信息学的融合

设计原则契合**个人信息学阶段**和**生活日志模型**：
- **收集阶段** (步骤1-2)：自动化内容备份与效率监控
- **反思阶段** (步骤3-4)：结构化组织与深度分析
- **行动阶段** (步骤5)：将洞察转化为可执行的写作行动

### 🔧 Copilot指令使用

获取详细的作者行动生成指令，请参考：[`copilot_action.prompt.example.md`](copilot_action.prompt.example.md)

该指令集帮助作者：
1. **内容回顾与模式识别**：基于TOC和词云分析结果进行写作模式总结
2. **主题缺口分析**：识别未充分探索的主题领域
3. **写作计划制定**：根据历史数据制定未来写作策略
4. **质量提升建议**：基于语言学分析提供文本改进建议

### 📋 输出建议

系统将在内容目录旁生成 `action_suggestion.md`，包含：
- 个性化写作建议
- 主题发展方向
- 内容质量优化要点
- 长期写作规划指导

---

## 输出目录结构

```
Wechat-Backup/<微信公众号名称>/
├── 克隆效率.json # 文章下载效率的JSON报告
├── 克隆效率.md # 文章下载效率的Markdown报告
├── 2025/
│ ├── 2025-08-26_文章标题1/
│ │ ├── 2025-08-26_文章标题1.html # 可双击离线打开
│ │ ├── 2025-08-26_文章标题1.md # 外链保留，适合 GitHub/Obsidian
│ │ ├── meta.json
│ │ └── images/
│ └── ...
└── _state.json # 已抓取链接指纹，供断点续传
```

## 注意事项

1. **Cookie和Token有效期**：需要定期更新，过期后脚本会提示预检失败
2. **网络请求频率**：脚本内置了请求间隔，避免被限制
3. **图片下载**：会自动下载文章中的图片并本地化保存
4. **断点续传**：通过 `_state.json` 记录已备份的文章，支持断点续传

> ✅ **说明**：相较官方网页源码，`HTML` 已包裹基础页面骨架并显式解锁正文容器，避免离线渲染空白；`Markdown` 会把 `<a>` 转换为 `[text](href)`，保留外链。

## 故障排除（简版，完整见 docs）

常见问题速查：
- **预检失败/403**：刷新登录态，重新复制 Cookie/Token；将 `COUNT=1` 重试
- **频繁失败/限速**：上调 `SLEEP_LIST`/`SLEEP_ART`/`IMG_SLEEP`，分批运行
- **HTML 空白/图片缺失**：用本地静态服务器访问；确认 `images/` 已生成
- **断点续传**：删除或清空 `_state.json` 可强制全量重抓

> 详细排障步骤与更多案例：请查看 `docs/TROUBLESHOOTING.md`

## 配置参数说明

- `COUNT`: 每页获取的文章数量（建议20，接口上限）
- `SLEEP_LIST`: 列表页请求间隔（秒）
- `SLEEP_ART`: 单篇文章请求间隔（秒）
- `IMG_SLEEP`: 图片下载间隔（秒）



## 🔑 如何获取 Cookie 与 Token（简版）

1. 登录 [https://mp.weixin.qq.com](https://mp.weixin.qq.com) 后进入 **内容与互动 → 发表记录**（或历史消息列表）。
2. 打开 **开发者工具 (F12) → Network**，刷新页面。
3. 找到指向 `/cgi-bin/appmsgpublish?sub=list...` 的请求，右键 **Copy → Copy as cURL**（或直接复制 **Request Headers** 里的 **Cookie**）。
4. 将 Cookie 粘贴进 `env.json` 的 `"COOKIE"`；URL 里的 `token=...` 值填入 `"TOKEN"`。

> 如果「Copy as cURL」不含 Cookie，可在 DevTools 设置里勾选 **Allow to generate HAR with sensitive data** 再试。

---

## 🖥️ 离线打开 HTML 的方式

* 直接双击 `YYYY-MM-DD_标题/YYYY-MM-DD_标题.html` 即可在本机浏览器打开；
* 某些系统安全策略或浏览器对 `file://` 资源有限制（例如阻止某些相对路径、显示提示），可以用一个**本地静态服务器**更稳妥：

```bash
# 在 Wechat-Backup 根目录启动
python -m http.server 8000
# 然后浏览器访问 http://127.0.0.1:8000/
```

---

## ⚙️ 可调参数（`env.json`）

* `COUNT`：列表分页大小（建议 1\~10；部分账户在 `count=1` 更稳定）
* `SLEEP_LIST` / `SLEEP_ART` / `IMG_SLEEP`：列表页/文章/图片抓取间隔（秒）
* **脚本内置抖动**（jitter），每次实际休眠会在基础值附近随机浮动，降低被风控概率

---

## 🧠 设计要点

* **去重**：用文章链接的 `md5` 作为指纹；`_state.json` 记录已抓取集合，避免重复。
* **图片**：逐个下载到 `images/` 并重写正文 `<img src>`；`wx_fmt` 保留原后缀。
* **HTML 包裹**：统一注入 `<!doctype html>`、`<meta charset="utf-8">`、基础样式，并显式使 `#js_content` / `.rich_media_content` 可见，保证离线渲染。
* **Markdown 转换**：优先把 `<a>` 转为 `[text](href)`，避免纯 `get_text()` 丢失外链。

---

## 🧪 预检与断点续传

* **预检**：程序启动先调用 `probe()`，仅拉取一条，确认能解析到 `publish_page.publish_list`；失败多为 Cookie/Token 过期或权限问题。
* **断点续传**：多次运行会跳过 `_state.json` 已记录的链接指纹；手动清空该文件可强制全量重抓。

---

## 🧯 故障排除（FAQ）

- 预检失败/列表为空：刷新登录态；重取 Cookie/Token；`COUNT=1` 再试
- HTML 空白或图片缺失：本地静态服务器访问；检查 `images/`
- 被限速/403：上调 `SLEEP_*`，保持抖动，分多次运行
- 接口权限不足：与账号类型/认证有关，请以官方权限说明为准

> 更多细节与操作截图：参见 `docs/TROUBLESHOOTING.md`

---

## 🔒 法务与合规提示

* 本工具仅用于**备份你自己账号**已发布的公开文章；
* 遵守微信公众平台相关条款与接口权限规范；
* 切勿将包含敏感信息的 `env.json` 或抓取所得**私密素材**公开发布。

---

## 🤝 贡献指南

欢迎提交 Issue/PR：

* 🐛 Bug 修复、📝 文档改进、💡 新功能提议
* 代码与脚本以 **MIT** 授权；**内容**以 **CC BY-NC-SA 4.0** 授权

---

## 📄 许可证

* **代码**：MIT
* **内容**：CC BY-NC-SA 4.0

---

## 🗺️ 项目展望

* 🤖 个人语料→专属对话模型
* 📊 写作风格与主题演化分析
* 🧩 知识图谱与检索增强
