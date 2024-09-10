# Ubuntu 22.04

## 拉取Vaultwander镜像

* 官方

    ```bash
    docker pull vaultwarden/server
    ```

* 阿里云 (可能并非最新，截止2024/6/15)

    ```cmd
    docker pull registry.cn-hangzhou.aliyuncs.com/axuandocker/vaultwarden
    ```

* 更改标签

    ```bash
    docker tag registry.cn-hangzhou.aliyuncs.com/axuandocker/vaultwarden:latest vaultwarden/server:latest
    
    # 删除原有标签
    docker rmi registry.cn-hangzhou.aliyuncs.com/axuandocker/vaultwarden
    ```

    



## 拉取Caddy镜像

* 官方

    ```cmd
    docker pull caddy
    ```

* 阿里云 (可能并非最新，截止2024/6/15)

    ```bash
    docker pull registry.cn-hangzhou.aliyuncs.com/axuandocker/caddy
    ```

* 更改标签

    ```bash
    docker tag registry.cn-hangzhou.aliyuncs.com/axuandocker/caddy:latest caddy:latest
    
    # 删除原有标签
    docker rmi registry.cn-hangzhou.aliyuncs.com/axuandocker/caddy
    ```

    