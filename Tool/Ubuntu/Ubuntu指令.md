# Ubuntu20.04

## 1.删除自定义添加的仓库源

```
/etc/apt/sources.list.d 目录下所有/指定文件删除
```



## 2.切换gcc/g++

```
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/g++-x 100  # 设置优先级

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/g++-X 50  # 低优先级
```

