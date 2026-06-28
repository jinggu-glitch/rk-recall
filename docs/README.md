# rk-recall documentation / rk-recall 文档

This directory contains the Sphinx documentation source for rk-recall.
本目录包含 rk-recall 的 Sphinx 文档源码.

## Build / 构建

```bash
pip install sphinx sphinx-rtd-theme
cd docs
make html
# Open _build/html/index.html
```

## Structure / 结构

```
docs/
├── conf.py           # Sphinx configuration / Sphinx 配置
├── index.rst         # Home page / 首页
├── installation.rst  # Installation guide / 安装指南
├── quickstart.rst    # Quick start / 快速开始
├── api.rst           # API reference / API 参考
├── theory.rst        # Theoretical background / 理论背景
└── examples.rst      # Examples / 示例
```
