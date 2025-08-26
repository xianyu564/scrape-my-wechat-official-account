# 🤝 贡献指南

感谢您对 **微信公众号文章备份工具** 项目的关注！我们欢迎所有形式的贡献。

> 💡 本项目专为《文不加点的张衔瑜》个人公众号设计，旨在解决微信官方接口停用后的备份需求。

## 如何贡献

### 1. 报告问题

如果您发现了bug或有功能建议，请：

1. 检查是否已有相关issue
2. 创建新的issue，包含：
   - 清晰的标题
   - 详细的描述
   - 重现步骤
   - 预期行为
   - 实际行为
   - 环境信息（操作系统、Python版本等）

### 2. 提交代码

#### 准备工作

1. Fork 本仓库
2. 克隆您的fork到本地
3. 创建新的分支：`git checkout -b feature/your-feature-name`

#### 开发流程

1. **代码风格**：遵循PEP 8规范
2. **测试**：确保代码通过现有测试
3. **文档**：更新相关文档和注释
4. **提交信息**：使用清晰的提交信息

#### 提交规范

```
类型(范围): 简短描述

详细描述（可选）

相关问题: #123
```

类型包括：
- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档更新
- `style`: 代码格式调整
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建过程或辅助工具的变动

#### 提交Pull Request

1. 推送您的分支到您的fork
2. 创建Pull Request
3. 填写PR模板
4. 等待代码审查

## 开发环境设置

### 1. 克隆仓库

```bash
git clone https://github.com/your-username/scrape-my-wechat-official-account.git
cd scrape-my-wechat-official-account
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 设置开发环境

```bash
# 复制环境配置文件
cp env.json.EXAMPLE env.json
# 编辑配置文件
```

### 4. 运行测试

```bash
# 运行所有测试
python -m pytest

# 运行特定测试
python -m pytest tests/test_specific.py
```

## 代码审查

所有代码提交都需要通过代码审查：

1. **自动化检查**：CI/CD流水线会运行测试和代码质量检查
2. **人工审查**：至少需要一名维护者批准
3. **反馈处理**：及时响应审查意见

## 发布流程

### 版本号规范

我们使用 [语义化版本控制](https://semver.org/lang/zh-CN/)：

- `MAJOR.MINOR.PATCH`
- 例如：`1.0.0`、`1.2.3`

### 发布步骤

1. 更新版本号
2. 更新CHANGELOG.md
3. 创建发布标签
4. 发布到PyPI（如果适用）

## 社区准则

- 尊重所有贡献者
- 保持专业和友好的交流
- 欢迎新手和专家
- 提供建设性的反馈

## 获取帮助

如果您需要帮助：

1. 查看 [README.md](README.md)
2. 搜索现有issues
3. 创建新的issue
4. 参与讨论

## 致谢

感谢所有为项目做出贡献的开发者！

---

如果您有任何问题，请随时联系我们！
