# 🔧 故障排除指南

本指南帮助您解决微信公众号文章备份工具使用过程中遇到的常见问题。

## 🚨 常见问题

### 1. 预检失败

#### 问题描述
```
→ 正在预检 appmsgpublish 接口…
❌ 预检失败：HTTP 403，请刷新后台后重新复制 Cookie/Token
```

#### 解决方案
1. **刷新微信公众平台**
   - 重新登录 https://mp.weixin.qq.com
   - 确保登录状态正常

2. **重新获取Cookie和Token**
   - 打开"内容与互动" → "历史消息"
   - 在浏览器开发者工具中复制最新的Cookie
   - 从URL中获取token参数

3. **检查权限**
   - 确认您的账号有"历史消息"查看权限
   - 确认公众号已通过认证

### 2. 配置文件错误

#### 问题描述
```
FileNotFoundError: 请创建env.json文件，参考env.json.EXAMPLE
```

#### 解决方案
1. **检查文件位置**
   ```bash
   # 确保env.json在项目根目录
   ls -la env.json
   ```

2. **复制示例文件**
   ```bash
   cp env.json.EXAMPLE env.json
   ```

3. **填写配置信息**
   - 参考README中的配置说明
   - 确保所有必填字段都已填写

### 3. 网络连接问题

#### 问题描述
```
requests.exceptions.ConnectionError: HTTPSConnectionPool
```

#### 解决方案
1. **检查网络连接**
   - 确认能够访问微信公众平台
   - 检查防火墙设置

2. **调整请求间隔**
   - 增加 `SLEEP_LIST` 和 `SLEEP_ART` 的值
   - 避免被微信限制访问

3. **使用代理（如需要）**
   ```python
   # 在脚本中添加代理配置
   proxies = {
       'http': 'http://your-proxy:port',
       'https': 'https://your-proxy:port'
   }
   S.proxies.update(proxies)
   ```

### 4. 图片下载失败

#### 问题描述
```
图片下载失败，跳过处理
```

#### 解决方案
1. **检查网络权限**
   - 确认脚本有网络访问权限
   - 检查图片URL是否可访问

2. **调整下载间隔**
   - 增加 `IMG_SLEEP` 的值
   - 避免被限制图片下载

3. **手动下载**
   - 如果某些图片始终无法下载，可以手动保存
   - 图片会保存在 `images/` 目录中

### 5. 文件权限问题

#### 问题描述
```
PermissionError: [Errno 13] Permission denied
```

#### 解决方案
1. **检查目录权限**
   ```bash
   # 确保有写入权限
   chmod 755 backup_文不加点的张衔瑜/
   ```

2. **以管理员权限运行**
   - Windows: 以管理员身份运行命令提示符
   - Linux/Mac: 使用 `sudo` 运行脚本

3. **检查磁盘空间**
   - 确保有足够的磁盘空间存储备份文件

## 🔍 调试技巧

### 1. 启用详细日志
```python
# 在脚本开头添加
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 2. 检查网络请求
```python
# 在requests.Session()后添加
S.verify = False  # 禁用SSL验证（仅用于调试）
```

### 3. 保存响应内容
```python
# 保存HTML响应用于分析
with open('debug_response.html', 'w', encoding='utf-8') as f:
    f.write(r.text)
```

## 📞 获取帮助

如果以上解决方案无法解决您的问题：

1. **查看现有Issues**
   - 搜索是否有类似问题
   - 查看已关闭的Issues

2. **创建新的Issue**
   - 提供详细的错误信息
   - 包含环境信息（操作系统、Python版本等）
   - 提供重现步骤

3. **通过赞助渠道联系**
   - 获得优先技术支持
   - 直接与维护者沟通

## 🎯 预防措施

### 1. 定期更新
- 保持Python和相关包的最新版本
- 关注项目的更新日志

### 2. 备份配置
- 定期备份 `env.json` 文件
- 保存重要的Cookie和Token信息

### 3. 监控运行状态
- 定期检查备份目录
- 关注脚本的运行日志

---

💡 **提示**：大多数问题都与Cookie/Token过期或网络连接有关。如果遇到问题，首先尝试重新获取最新的认证信息。
