.PHONY: help install dev-install test test-fast lint format typecheck benchmark clean

help: ## 显示所有可用命令
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## 安装核心依赖
	pip install -e .

dev-install: ## 安装开发依赖（含 vllm / agent / dev）
	pip install -e ".[vllm,agent,dev]"

test: ## 运行全部测试
	pytest

test-fast: ## 运行快速测试（跳过 slow / gpu）
	pytest -m "not slow and not gpu"

lint: ## 代码检查
	ruff check src tests benchmarks

format: ## 自动格式化
	ruff format src tests benchmarks
	ruff check --fix src tests benchmarks

typecheck: ## 类型检查
	mypy

benchmark: ## 运行 Benchmark
	python -m benchmarks.run

clean: ## 清理构建产物
	rm -rf build dist *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
