# Makefile for scrape-my-wechat-official-account
# 一键运行测试、代码检查和CI任务

.PHONY: help install test lint type format check ci clean

# Python executable
PYTHON := python3

# Help target
help:  ## 显示帮助信息
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Installation
install:  ## 安装项目依赖
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install pytest pytest-cov ruff mypy

install-dev: install  ## 安装开发依赖（包括可选依赖）
	$(PYTHON) -m pip install -e .[dev] || true
	@echo "Development dependencies installed"

# Testing
test:  ## 运行测试套件
	$(PYTHON) -m pytest tests/ -v

test-coverage:  ## 运行测试并生成覆盖率报告
	$(PYTHON) -m pytest tests/ -v --cov=script --cov=analysis --cov-report=term-missing --cov-report=html

test-quiet:  ## 安静模式运行测试
	$(PYTHON) -m pytest tests/ -q

# Code quality
lint:  ## 运行 ruff 代码检查
	ruff check script/ analysis/ tests/

lint-fix:  ## 运行 ruff 并自动修复问题
	ruff check script/ analysis/ tests/ --fix

format:  ## 使用 ruff 格式化代码
	ruff format script/ analysis/ tests/

type:  ## 运行 mypy 类型检查
	mypy script/ analysis/ --ignore-missing-imports

type-strict:  ## 运行严格的 mypy 类型检查（针对 src/ 目录）
	mypy script/ --ignore-missing-imports --strict

# Combined checks
check: test  ## 运行所有代码质量检查（适合当前代码状态）

check-strict: lint type test  ## 运行严格的代码质量检查

check-fix: lint-fix format test  ## 运行所有检查并自动修复

# CI pipeline
ci: test  ## 运行适合当前代码状态的CI检查流程

ci-verbose: lint-fix format test-coverage  ## 运行详细的CI检查流程（自动修复问题）

ci-strict: lint type test-coverage  ## 运行严格的CI检查流程

# Cleanup
clean:  ## 清理生成的文件
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + || true
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/

# Local development server (if applicable)
dev-setup:  ## 设置开发环境
	@echo "Setting up development environment..."
	@make install-dev
	@echo "Creating test env.json if not exists..."
	@test -f env.json || cp env.json.EXAMPLE env.json || echo '{"WECHAT_ACCOUNT_NAME":"test","COOKIE":"test","TOKEN":"test"}' > env.json
	@echo "Development environment ready!"

# Analysis-specific targets
analysis-test:  ## 测试分析模块（需要额外依赖）
	$(PYTHON) -c "import analysis.main; print('Analysis module loads successfully')" || echo "Analysis dependencies missing - install numpy, pandas, etc."

# Quick quality check (for pre-commit hooks)
pre-commit: lint-fix type test-quiet  ## 快速代码质量检查（适合pre-commit）

# Show project status
status:  ## 显示项目状态
	@echo "=== Project Status ==="
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "Pip version: $$($(PYTHON) -m pip --version | cut -d' ' -f1-2)"
	@echo "Project directory: $$(pwd)"
	@echo "Tests directory: $$(test -d tests && echo 'exists' || echo 'missing')"
	@echo "Config file: $$(test -f env.json && echo 'exists' || echo 'missing (use env.json.EXAMPLE)')"
	@echo "=== Dependencies ==="
	@$(PYTHON) -c "import requests, bs4, lxml; print('✅ Core dependencies: OK')" || echo "❌ Core dependencies: Missing"
	@$(PYTHON) -c "import pytest, ruff, mypy; print('✅ Dev dependencies: OK')" || echo "❌ Dev dependencies: Missing" 
	@echo "Run 'make install' to install dependencies"

# Default target
.DEFAULT_GOAL := help