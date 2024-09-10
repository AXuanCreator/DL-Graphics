# Ubuntu 22.04

1. 搜索Portainer镜像版本

    ```bash
    docker search portainer
    ```

2. 拉取Portainer，以Portainer社区版为例

    ```bash
    docker pull portainer/portainer-ce
    ```

3. 启用Portainer

    ```bash
    docker run -d -p 9000:9000 -v /var/run/docker.sock:/var/run/docker.sock -v /dockerData/portainer:/data --restart=always --name portainer portainer/portainer-ce:latest
    ```

    

## 使用阿里云镜像

1. 从阿里云镜像仓库拉取Portainer

    ```cmd
    docker pull registry.cn-hangzhou.aliyuncs.com/axuandocker/portainer-ce:cn
    ```

2. 对镜像重新打标签

    ```cmd
    docker tag registry.cn-hangzhou.aliyuncs.com/axuandocker/portainer-ce:cn portainer/portainer-ce:cn
    
    # 删除原有标签
    docker rmi registry.cn-hangzhou.aliyuncs.com/axuandocker/portainer-ce:cn
    ```

3. 运行

    ```cmd
    docker run -d -p 9000:9000 -v /var/run/docker.sock:/var/run/docker.sock -v /dockerData/portainer:/data --restart=always --name portainer portainer/portainer-ce:cn
    ```

