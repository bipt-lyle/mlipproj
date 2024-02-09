# 使用官方 Python 运行时作为父镜像
FROM python:3.8-slim

# 设置环境变量:
# - 不生成 pyc 文件
# - Python 输出直接到终端，以便在日志中看到输出
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 设置工作目录为 /app
WORKDIR /code

# Install any needed packages specified in requirements.txt
# Update pip and install curl
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip


# 将当前目录内容复制到容器中的 /app
COPY . /code

# 安装 requirements.txt 中的所有 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 在容器启动时运行 app.py
CMD gunicorn --bind 0.0.0.0:$PORT app:app

