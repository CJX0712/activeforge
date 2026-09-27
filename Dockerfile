# ActiveForge — 主动学习系统（CPU / 零下载 / 一键复现）
FROM python:3.13-slim

LABEL maintainer="晨星 <CJX0712@users.noreply.github.com>"
LABEL description="ActiveForge: modular active learning system, CPU-only, reproducible"

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 默认入口：运行端到端基准并落盘 benchmark.json
CMD ["python", "-m", "activeforge.cli", "bench", "--out", "benchmark.json"]
