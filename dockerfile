# 使用官方Python运行时作为父镜像
FROM python:3.8-slim

# 设置环境变量，Python不会写入pyc文件到磁盘
ENV PYTHONDONTWRITEBYTECODE 1
# 设置环境变量，Python输出直接到终端，看起来不会被缓存
ENV PYTHONUNBUFFERED 1

# 设置工作目录为/code
WORKDIR /code

# 将当前目录内容复制到容器中的/code
COPY . /code

# 安装requirements.txt中的所有依赖
RUN pip install --no-cache-dir -r requirements.txt

# 对外暴露端口8000
EXPOSE 8000

# 定义环境变量
# ENV NAME World

# 在容器启动时运行app.py
CMD ["python", "./app.py"]
