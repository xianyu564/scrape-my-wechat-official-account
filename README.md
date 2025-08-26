
<div align="center">
  <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin: 20px 0; font-size: 2.5em;">
    🚀 微信公众号文章备份工具
  </h1>
  
  <p style="font-size: 1.2em; color: #666; max-width: 800px; margin: 0 auto 30px;">
    专业的微信公众号文章抓取和备份解决方案，支持图片本地化、多格式输出、断点续传等功能
  </p>
  
  <div style="display: flex; justify-content: center; gap: 15px; margin-bottom: 30px;">
    <a href="https://opensource.org/licenses/MIT">
      <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="MIT License" />
    </a>
    <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">
      <img src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg?style=for-the-badge" alt="CC BY-NC-SA 4.0" />
    </a>
    <img src="https://img.shields.io/badge/Python-3.7+-blue.svg?style=for-the-badge&logo=python" alt="Python 3.7+" />
    <img src="https://img.shields.io/badge/Status-Active%20Maintenance-brightgreen.svg?style=for-the-badge" alt="Active Maintenance" />
  </div>
</div>

<div align="center" style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 20px; border-radius: 15px; margin: 20px 0; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
  <p style="margin: 0; font-size: 1.1em; color: #333;">
    🎯 <strong>本工具专为《文不加点的张衔瑜》个人公众号设计</strong>
    本公众号已通过个人职业兴趣认证(哲学博士，专注于AI 、计算化学、生物医药、 周易等，探索科技与哲学的边界。)
  </p>
</div>

## 💭 项目动机

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 20px; color: white; margin: 20px 0; box-shadow: 0 10px 20px rgba(0,0,0,0.15);">

<div style="display: flex; align-items: center; gap: 20px; margin-bottom: 20px;">
  <div style="font-size: 3em;">🎁</div>
  <div>
    <h3 style="margin: 0 0 10px 0; color: white;">《文不加点的张衔瑜》的诞生</h3>
    <p style="margin: 0; opacity: 0.9;">这是我成年当天的礼物</p>
  </div>
</div>

<div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px; backdrop-filter: blur(10px); margin-bottom: 20px;">
  <h4 style="margin: 0 0 15px 0; color: white;">📚 八年公众号写作历程</h4>
  <ul style="margin: 0; padding-left: 20px; opacity: 0.9;">
    <li>累计创作了400多篇文章，记录人生各个阶段的思考</li>
    <li>涵盖生活日志、旅行笔记、社会评论等多元内容</li>
  </ul>
</div>

<div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px; backdrop-filter: blur(10px); margin-bottom: 20px;">
  <h4 style="margin: 0 0 15px 0; color: white;">⏰ 时间节点的巧合与遗憾</h4>
  <p style="margin: 0; opacity: 0.9;">
    原本计划通过官方渠道进行备份，于是我自己写作了 <a href="https://github.com/xianyu564/wechat_official_backup" style="color: #ffd700; text-decoration: underline;">@xianyu564/wechat_official_backup</a> 项目。
    然而，微信官方在 <strong>2025年7月</strong> 停用了关键的 <code>freepublishGetarticle</code> 接口（<code>/cgi-bin/freepublish/getarticle</code>），
    这个时间点恰好比我开始备份的想法早了<strong>一个月</strong>。
  </p>
</div>

<div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px; backdrop-filter: blur(10px);">
  <h4 style="margin: 0 0 15px 0; color: white;">🛠️ 解决方案的诞生</h4>
  <p style="margin: 0; opacity: 0.9;">
    面对官方接口的停用，我开发了这个替代方案，通过模拟浏览器行为来获取已发布文章，
  </p>
</div>

</div>

## 📋 目录

- [项目动机](#项目动机)
- [功能特性](#功能特性)
- [快速开始](#快速开始)
- [安装依赖](#安装依赖)
- [配置说明](#配置说明)
- [使用方法](#使用方法)
- [输出结构](#输出结构)
- [项目展望](#项目展望)
- [故障排除](#故障排除)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

## ✨ 功能特性

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0;">

  <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; color: white; box-shadow: 0 8px 16px rgba(0,0,0,0.15);">
    <h3 style="margin: 0 0 15px 0; color: white;">🔍 智能抓取</h3>
    <p style="margin: 0; opacity: 0.9;">自动抓取微信公众号已发布文章，支持分页获取和智能去重。</p>
  </div>

  <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 20px; border-radius: 15px; color: white; box-shadow: 0 8px 16px rgba(0,0,0,0.15);">
    <h3 style="margin: 0 0 15px 0; color: white;">📁 智能组织</h3>
    <p style="margin: 0; opacity: 0.9;">按年份自动组织备份文件结构，便于管理和查找。</p>
  </div>

  <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 20px; border-radius: 15px; color: white; box-shadow: 0 8px 16px rgba(0,0,0,0.15);">
    <h3 style="margin: 0 0 15px 0; color: white;">🖼️ 图片本地化</h3>
    <p style="margin: 0; opacity: 0.9;">自动下载并本地化图片资源，确保备份完整性。</p>
  </div>

  <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); padding: 20px; border-radius: 15px; color: white; box-shadow: 0 8px 16px rgba(0,0,0,0.15);">
    <h3 style="margin: 0 0 15px 0; color: white;">📝 多格式输出</h3>
    <p style="margin: 0; opacity: 0.9;">支持HTML和Markdown格式输出，满足不同使用需求。</p>
  </div>

  <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); padding: 20px; border-radius: 15px; color: white; box-shadow: 0 8px 16px rgba(0,0,0,0.15);">
    <h3 style="margin: 0 0 15px 0; color: white;">⚡ 性能优化</h3>
    <p style="margin: 0; opacity: 0.9;">可配置的抓取速度和分页大小，平衡效率与稳定性。</p>
  </div>

</div>

## 🚀 快速开始

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 20px; color: white; margin: 20px 0; box-shadow: 0 10px 20px rgba(0,0,0,0.15);">

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px;">

<div style="text-align: center; padding: 20px; background: rgba(255,255,255,0.1); border-radius: 15px; backdrop-filter: blur(10px);">
  <div style="font-size: 2em; margin-bottom: 10px;">📥</div>
  <h3 style="margin: 0 0 10px 0; color: white;">1. 克隆仓库</h3>
  <p style="margin: 0; opacity: 0.9; font-size: 0.9em;">获取项目代码到本地，开始你的备份之旅</p>
</div>

<div style="text-align: center; padding: 20px; background: rgba(255,255,255,0.1); border-radius: 15px; backdrop-filter: blur(10px);">
  <div style="font-size: 2em; margin-bottom: 10px;">📦</div>
  <h3 style="margin: 0 0 10px 0; color: white;">2. 安装依赖</h3>
  <p style="margin: 0; opacity: 0.9; font-size: 0.9em;">安装Python依赖包，为工具运行做好准备</p>
</div>

<div style="text-align: center; padding: 20px; background: rgba(255,255,255,0.1); border-radius: 15px; backdrop-filter: blur(10px);">
  <div style="font-size: 2em; margin-bottom: 10px;">⚙️</div>
  <h3 style="margin: 0 0 10px 0; color: white;">3. 配置环境</h3>
  <p style="margin: 0; opacity: 0.9; font-size: 0.9em;">设置Cookie和Token，让工具能够访问你的公众号</p>
</div>

<div style="text-align: center; padding: 20px; background: rgba(255,255,255,0.1); border-radius: 15px; backdrop-filter: blur(10px);">
  <div style="font-size: 2em; margin-bottom: 10px;">🚀</div>
  <h3 style="margin: 0 0 10px 0; color: white;">4. 运行脚本</h3>
  <p style="margin: 0; opacity: 0.9; font-size: 0.9em;">开始备份文章，让每一篇文字都得到妥善保存</p>
</div>

</div>

</div>

### 详细步骤

#### 1. 克隆仓库
```bash
git clone https://github.com/xianyu564/scrape-my-wechat-official-account.git
cd scrape-my-wechat-official-account
```

#### 2. 安装依赖
```bash
pip install -r requirements.txt
```

#### 3. 配置环境
复制 `env.json.EXAMPLE` 为 `env.json`，填写配置信息

#### 4. 运行脚本
```bash
cd script
python wx_publish_backup.py
```

## 🧭 复用者快速起步（Fork/Clone 即用）

> 面向想直接复用本工具的开发者：不改代码，配置好 `env.json` 即可跑通。

### 最小可运行示例

1. 复制示例配置：
   - 将项目根目录下的 `env.json.EXAMPLE` 复制为 `env.json`
   - 至少填写三项：

```json
{
  "WECHAT_ACCOUNT_NAME": "你的公众号名称",
  "COOKIE": "从浏览器开发者工具复制的Cookie",
  "TOKEN": "发表记录页URL中的token值"
}
```

2. 运行（任选其一）：

- Windows PowerShell
```powershell
py -3 script\wx_publish_backup.py
```

- macOS / Linux（终端）
```bash
python3 script/wx_publish_backup.py
```

3. 输出位置：
   - 文章会按年份落到 `Wechat-Backup/<你的公众号名称>/YYYY/` 下
   - 每篇文章目录包含：`*.html`、`*.md`、`meta.json`、`images/`

### 常见上手问题（超简版）
- 403 或“预检失败”：Cookie/Token 过期 → 重新抓取
- HTML 打开空白：用 `python -m http.server` 启动本地静态服务器后访问
- 速度过快被限流：适当调大 `SLEEP_LIST`/`SLEEP_ART`/`IMG_SLEEP`

## 📦 安装依赖

```bash
pip install -r requirements.txt
```

或者手动安装：

```bash
pip install requests beautifulsoup4 lxml
```

## ⚙️ 配置说明

### 基础配置步骤

1. **复制配置文件**：复制 `env.json.EXAMPLE` 为 `env.json`
  * 🔐 安全：env.json 已加入 .gitignore，请勿提交。Cookie 常含 HttpOnly 字段，不会出现在 document.cookie，需在 DevTools/Network 里从真实请求复制。
  *⚠️ 有效期：Cookie/Token 有时效，失效时重登后台并重新复制。
2. **填写配置信息**：根据你的公众号信息填写相应字段

### 配置文件详解

```json
{
  "WECHAT_APPID": "你的微信公众号APPID", //其实不重要，这是上一个项目的遗留，属于公众号开发接口
  "WECHAT_APPSECRET": "你的微信公众号APPSECRET",  // 同上
  "WECHAT_ACCOUNT_NAME": "你的微信公众号名称", //为了命名的，可以是任意名字，不非得跟公众号相同
  "COOKIE": "从浏览器复制的Cookie",
  "TOKEN": "从发表记录页URL获取的token",
  "COUNT": "20",
  "SLEEP_LIST": "0.6",
  "SLEEP_ART": "0.3"
}
```

### 🔑 获取Cookie和Token的详细步骤

#### 1. 获取Cookie
就像获取通行证一样，Cookie是你访问微信公众平台的凭证：

1. **登录微信公众平台**：访问 https://mp.weixin.qq.com
2. **进入内容管理**：点击"内容与互动" → "历史消息"
3. **打开开发者工具**：按F12键，切换到Network标签
4. **复制Cookie**：找到 `appmsg?action=list_ex...` 请求，复制完整的Cookie

#### 2. 获取Token
Token就像是时间的钥匙，让你能够访问特定时间段的文章：

1. **查看发表记录**：在历史消息页面，查看URL地址
2. **提取Token参数**：从URL中找到 `token=` 后面的参数值
3. **复制到配置文件**：将完整的token值复制到 `env.json` 文件中

### ⚠️ 重要提醒

- **Cookie有效期**：Cookie会定期过期，需要及时更新
- **Token唯一性**：每个时间段的Token都是唯一的
- **信息安全**：请妥善保管这些敏感信息，不要泄露给他人

## 💻 使用方法

```bash
cd script
python wx_publish_backup.py
```

## 📁 输出结构

备份文件将保存在 `backup_<微信公众号名字>/` 目录下：

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

## 🤝 贡献指南

我们欢迎所有形式的贡献！请查看 [贡献指南](CONTRIBUTING.md) 了解如何参与项目开发。

### 贡献方式

- 🐛 报告Bug
- 💡 提出新功能建议
- 📝 改进文档
- 🔧 提交代码修复
- 🌟 给项目点星

## 📜 行为准则

本项目采用 [贡献者公约](CODE_OF_CONDUCT.md) 作为行为准则。我们致力于为每个人创造友好、包容的环境。

## 🔒 安全政策

如果您发现了安全漏洞，请查看 [安全政策](SECURITY.md) 了解如何私下报告。

**重要**：请不要在公开渠道中报告安全漏洞。

## 📄 许可证

本项目采用双重许可证：

- **代码**: [MIT License](LICENSE) - 允许自由使用、修改和分发代码
- **内容**: [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](CONTENT_LICENSE.md) - 允许非商业用途的分享和改编

## ⚠️ 注意事项

- `env.json` 包含敏感信息，已添加到 `.gitignore`
- 建议设置合理的抓取间隔，避免被限制访问
- 备份目录会自动创建，无需手动创建
- 请遵守微信公众平台的使用条款

## 📊 项目状态

- **版本**: 1.0.0
- **状态**: 活跃维护
- **Python版本**: 3.7+

## 🔮 项目展望

本项目不仅是一个备份工具，更是个人知识资产的基础设施。基于备份的400+篇个人文章，我们规划了以下发展方向：

### 🤖 个人AI对话模型
- 利用个人写作风格和知识结构训练专属AI助手
- 保持与作者一致的表达方式和思维模式
- 传承个人在AI、计算化学、生物医药、周易等领域的知识

### 📊 语料分析与洞察
- 词频分析和写作风格研究
- 主题演化追踪和知识图谱构建
- 为内容创作和学术研究提供数据支持

> 💡 **了解更多**：详细的项目展望请查看 [STATUS.md](STATUS.md#-项目展望)

## 🔧 故障排除

如果您在使用过程中遇到问题，请查看 [故障排除指南](docs/TROUBLESHOOTING.md) 获取详细的解决方案。

### 常见问题快速解决

- **预检失败**：重新获取Cookie和Token
- **配置文件错误**：检查env.json文件位置和格式
- **网络连接问题**：调整请求间隔或使用代理
- **图片下载失败**：检查网络权限和磁盘空间
- **文件权限问题**：检查目录权限和磁盘空间

## 📞 获取帮助

如果您需要帮助：

1. 查看 [故障排除指南](docs/TROUBLESHOOTING.md) 解决常见问题
2. 查看 [更新日志](CHANGELOG.md) 了解最新变更
3. 搜索现有 [Issues](https://github.com/xianyu564/scrape-my-wechat-official-account/issues)
4. 创建新的 Issue 描述问题

## 🙏 致谢

凡没有把我杀死的，都没有把我杀死。

---

<div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 30px; border-radius: 20px; margin: 30px 0; text-align: center; box-shadow: 0 8px 16px rgba(0,0,0,0.1);">

<h2 style="margin: 0 0 20px 0; color: #333;">⭐️ 如果这个项目对您有帮助，请给我们一个星标！</h2>

<p style="font-size: 1.1em; color: #666; margin-bottom: 25px;">
  您的支持是我们持续改进的最大动力
</p>

</div>
