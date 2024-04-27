

# CUDA

## 版本切换(临时切换)

```bash
export PATH=/usr/local/cuda-xx.x/bin:$PATH

export LD_LIBRARY_PATH=/usr/local/cuda-xx.x/lib64:$LD_LIBRARY_PATH

export CUDA_HOME=/usr/local/cuda-xx.x
```

若不存在lib64请查看 问题.md





## CUDA卸载

### CUDA >= 10.1

```bash
cd /usr/local/cuda-xx.x/bin/
sudo ./cuda-uninstaller
sudo rm -rf /usr/local/cuda-xx.x
```

### CUDA < 10.1

```bash
cd /usr/local/cuda-xx.x/bin/
sudo ./uninstall_cuda_xx.x.pl
sudo rm -rf /usr/local/cuda-xx.x
```





## CUDA安装



### CUDA 11.3

#### Ubuntu 20.04

```bash
# Runfile
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run

sudo sh cuda_11.3.0_465.19.01_linux.run
```

```bash
# lib64软链接
sudo ln -s /usr/local/cuda-11.3/targets/x86_64-linux/lib/ /usr/local/cuda-11.3/lib64

export PATH=/usr/local/cuda-11.3/bin:$PATH

export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH

export CUDA_HOME=/usr/local/cuda-11.3
```





### CUDA 11.4

#### Ubuntu 20.04

```bash
# Runfile
wget https://developer.download.nvidia.com/compute/cuda/11.4.0/local_installers/cuda_11.4.0_470.42.01_linux.run

sudo sh cuda_11.4.0_470.42.01_linux.run
```

```bash
# lib64软链接
sudo ln -s /usr/local/cuda-11.4/targets/x86_64-linux/lib/ /usr/local/cuda-11.4/lib64

export PATH=/usr/local/cuda-11.4/bin:$PATH

export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH

export CUDA_HOME=/usr/local/cuda-11.4
```





### CUDA 11.6

#### Ubuntu 20.04

```bash
# Runfile
wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda_11.6.0_510.39.01_linux.run

sudo sh cuda_11.6.0_510.39.01_linux.run
```

```
# lib64软链接
sudo ln -s /usr/local/cuda-11.6/targets/x86_64-linux/lib/ /usr/local/cuda-11.6/lib64

export PATH=/usr/local/cuda-11.6/bin:$PATH

export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH

export CUDA_HOME=/usr/local/cuda-11.6
```



##### deb local(不太能成功）

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin

sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600

wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda-repo-ubuntu2004-11-6-local_11.6.0-510.39.01-1_amd64.deb

sudo dpkg -i cuda-repo-ubuntu2004-11-6-local_11.6.0-510.39.01-1_amd64.deb

sudo apt-key add /var/cuda-repo-ubuntu2004-11-6-local/7fa2af80.pub

sudo apt-get update

sudo apt-get -y install cuda
# sudo apt-get -y install cuda-11.6
```



### CUDA 11.8

#### Ubuntu 20.04

```bash
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run

sudo sh cuda_11.8.0_520.61.05_linux.run
```

```
# lib64软链接
sudo ln -s /usr/local/cuda-11.8/targets/x86_64-linux/lib/ /usr/local/cuda-11.8/lib64

export PATH=/usr/local/cuda-11.8/bin:$PATH

export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

export CUDA_HOME=/usr/local/cuda-11.8
```



# cuDNN

## Deb(不推荐，因为我没成功)

### Ubuntu 2004 x86_64  9.0.0

```bash
wget https://developer.download.nvidia.com/compute/cudnn/9.0.0/local_installers/cudnn-local-repo-ubuntu2004-9.0.0_1.0-1_amd64.deb

sudo dpkg -i cudnn-local-repo-ubuntu2004-9.0.0_1.0-1_amd64.deb

sudo cp /var/cudnn-local-repo-ubuntu2004-9.0.0/cudnn-*-keyring.gpg /usr/share/keyrings/

sudo apt-get update

sudo apt-get -y install cudnn

# CUDA 11.X
sudo apt-get -y install cudnn-cuda-11

# CUDA 12.X
sudo apt-get -y install cudnn-cuda-12
```



## TAR(推荐)

下载地址

```
https://developer.nvidia.com/rdp/cudnn-archive
```

安装

1.解压

```
tar -xvf [NAME]
```

2.进入文件夹

```
cd ./cudnn*
```

3.软链接

```
sudo cp include/* /usr/local/cuda-[VERSION]/include/

sudo cp lib/lib* /usr/local/cuda-[VERSION]/lib64/

cd /usr/local/cuda-[VERSION]/lib64/

sudo chmod +r libcudnn.so.X.X.X

sudo ln -sf libcudnn.so.X.X.X libcudnn.so.X

sudo ln -sf libcudnn.so.X libcudnn.so  

sudo ldconfig
```





# TensorRT

**不推荐使用DEB安装，使用TAR安装**

## TAR

1.下载地址

```bash
https://developer.nvidia.com/tensorrt-getting-started
```

2.解压

```bash
tar -xvf [NAME]
```

3.将lib添加到环境路径里

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:[Absolute_PATH_TO_tensorRT]
```

4.软链接

```bash
sudo cp -r ./lib/* /usr/lib
sudo cp -r ./include/* /usr/include
```

5.[Optional]Python版本

```
cd ./[PATH_TO_tensorRT]
cd ./python
pip install tensorrt-[VERSION].whl

# Tensorflow模型转换支持
cd ..
cd ./iff
pip install uff-[VERSION].whl

# 自定义结构支持
cd ..
cd ./graphsurgeon
pip install graphsurgeon-[VERSION].whl
```













