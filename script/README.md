# 微信公众账号文章备份脚本

## 功能说明

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

```bash
# 在项目根目录运行
python script/wx_publish_backup.py

# 或者在script目录下运行
cd script
python wx_publish_backup.py
```

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

## 输出目录结构

```
Wechat-Backup/<微信公众号名称>/
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

## 故障排除

### 预检失败
- 检查Cookie和Token是否过期
- 确认网络连接正常
- 尝试刷新微信公众平台页面后重新获取

### 备份中断
- 脚本支持断点续传，重新运行即可
- 检查网络连接和请求频率设置

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

* **预检失败 / 列表为空**

  1. 刷新登录态，按上文重新复制 Cookie/Token；
  2. 把 `COUNT` 降到 `1` 重试；
  3. 检查是否误用了浏览器地址栏里的 `document.cookie`（`HttpOnly` Cookie 不会出现）。

* **双击 HTML 显示空白或图片缺失**

  1. 优先用 `python -m http.server` 跑本地服务器再访问；
  2. 确认该文章目录下的 `images/` 已生成且文件存在；
  3. 脚本已去除正文容器的隐藏样式，若仍异常请附上问题 HTML 片段。

* **被限速/403**
  适当**加大** `SLEEP_*`、保持抖动、分多次运行，避免短时间大量请求。

* **接口权限不足**
  微信的部分 **发布/获取**能力对**企业认证**账号更友好；个人或未认证账号可能受限（以官方权限说明为准）。

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

```