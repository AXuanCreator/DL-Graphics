# V1.0

## 特征

Ubuntu 20.04

开放22端口

自动启动SSH，且允许通过root进行ssh连接

内置Miniconda，且自动激活base环境

```dockerfile
# 基于ubuntu 20.04的基础镜像
FROM ubuntu:20.04

# 设置时区为亚洲/香港
ENV TZ=Asia/Hong_Kong
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 更新apt并安装必要的软件包
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    
    bzip2 \
    unzip \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    git \
    mercurial \
    subversion \
    openssh-server \
    nano \
    ufw \
    gcc \
    g++ \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 放开22端口 —— 允许SSH连接
RUN ufw allow 22

# 启动SSH
RUN service ssh start

# 更新源
RUN apt-get update

# 下载并安装Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

# 将Miniconda添加到环境变量中
ENV PATH=/opt/conda/bin:$PATH

# 确保conda更新
RUN conda init --all && \
    conda update -n base -c defaults conda

# 设置conda默认为非交互模式
RUN echo "auto_activate_base: true" >> ~/.condarc

# 安装Python 3.8
RUN conda install -y python=3.8

# 指定容器启动时自动启动Miniconda
CMD ["bash", "-c", "source ~/.bashrc && conda activate base && /bin/bash"]

# 编辑sshd_config以允许root登录
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

```



