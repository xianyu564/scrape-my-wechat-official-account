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
backup_文不加点的张衔瑜/
├── 2024/
│   ├── 2024-12-19_文章标题/
│   │   ├── 2024-12-19_文章标题.html
│   │   ├── article.md
│   │   ├── meta.json
│   │   └── images/
│   │       ├── img_001.jpg
│   │       └── img_002.png
│   └── 2024-12-18_另一篇文章/
│       └── ...
└── _state.json
```

## 注意事项

1. **Cookie和Token有效期**：需要定期更新，过期后脚本会提示预检失败
2. **网络请求频率**：脚本内置了请求间隔，避免被限制
3. **图片下载**：会自动下载文章中的图片并本地化保存
4. **断点续传**：通过 `_state.json` 记录已备份的文章，支持断点续传

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
