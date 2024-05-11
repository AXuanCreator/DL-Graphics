





**Ubuntu20.04**

**CUDA-11.3**



# 1.Cmake安装

## 有Root权限

```
sudo apt install cmake

cmake --version
```

## 无Root权限

* Cmake-3.21.3

**更推荐在DockerFile里面配置cmake**

1. 下载Cmake的Source压缩包

    ```
    https://cmake.org/download/
    ```

2. 解压

    ```
    tar -zxvf cmake-3.21.3.tar.gz
    
    cd cmake-3.21.3
    ```

3. 修改CMakeLists.txt

    ```
    # CMakeLists.txt位于cmake文件夹下
    # 第356行插入
    set(CMAKE_USE_OPENSSL OFF)
    
    ## 其他版本的需找到类似以下结构的地方
    #---------------------------------------------------------------------
      # Create the kwsys library for CMake.
      set(KWSYS_NAMESPACE cmsys)
      set(KWSYS_USE_SystemTools 1)
      set(KWSYS_USE_Directory 1)
      set(KWSYS_USE_RegularExpression 1)
      set(KWSYS_USE_Base64 1)
      set(KWSYS_USE_MD5 1)
      set(CMAKE_USE_OPENSSL OFF)
      set(KWSYS_USE_Process 1)
      set(KWSYS_USE_CommandLineArguments 1)
      set(KWSYS_USE_ConsoleBuf 1)
      set(KWSYS_HEADER_ROOT ${CMake_BINARY_DIR}/Source)
      set(KWSYS_INSTALL_DOC_DIR "${CMAKE_DOC_DIR}")
    ```

4. 安装

    ```
    ./bootstrap
    
    ./configure --prefix=[Install_Path]
    
    make -j `nproc`
    
    make install
    ```

5. 加入环境变量

    ```
    # 单次SSH生效
    export PATH=[Install_Path]/bin:$PATH
    
    cmake --version
    ```



# 2.OpenCV安装

## 存在Root权限

```
sudo apt-get install libopencv-dev

pkg-config --modversion opencv
```

## 无Root权限

* Opencv-3.4.9

**更推荐在DockerFile里面配置opencv**

1. 下载Opencv与Opencv-Contrib

    ```OpenCV
    https://opencv.org/releases/  # OpenCV
    
    https://pan.baidu.com/s/1Xt-gxruQLFmTMLzQd0MRNw  # OpenCV-Contrib  Password : lyh1
    ```

2. 将OpenCV和OpenCV-Contrib解压

3. 创建build文件夹

    ```
    cd opencv-3.4.9
    mkdir build
    cd build
    ```

4. 使用cmake构建

    ```bash
    # 替换[TO OPENCV_CONTRIB]和[WHICH CMAKE]
    cmake -D BUILD_opencv_cudacodec=OFF -D CMAKE_BUILD_TYPE=RELEASE -D WITH_TBB=ON -D WITH_V4L=ON -D BUILD_TIFF=ON -D BUILD_EXAMPLES=OFF -D BUILD_DOCS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF -D WITH_OPENGL=ON -D WITH_EIGEN=ON -D WITH_CUBLAS=ON -D OPENCV_EXTRA_MODULES_PATH=[TO OPENCV_CONTRIB]/opencv_contrib-3.4.9/modules/ -D CMAKE_INSTALL_PREFIX=[WHICH CMAKE] ..
    
    make -j4  # 可以将数值提升到CPU核心数
    
    make install
    ```

5. 配置环境变量

    ```bash
    cd /usr/local/lib
    
    mkdir pkgconfig
    
    cd pkgconfig
    
    touch opencv.pv
    
    # 添加并修改以下内容
    ## Start
    prefix=/usr/local
    exec_prefix=${prefix}
    includedir=${prefix}/include
    libdir=${exec_prefix}/lib
    
    Name: opencv
    Description: The opencv library
    Version:3.4.9
    Cflags: -I${includedir}/opencv3
    Libs: -L${libdir} -lopencv_shape -lopencv_stitching -lopencv_objdetect -lopencv_superres -lopencv_videostab -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_video -lopencv_photo -lopencv_ml -lopencv_imgproc -lopencv_flann  -lopencv_core        
    
    ## End
    
    pkg-config --modversion opencv
    ```

## 问题

### 无法找到一些文件：

```
/liuxianguo1/XYX/Others/opencv_contrib-3.4.9/modules/xfeatures2d/src/vgg.cpp:490:20: fatal error: vgg_generated_120.i: No such file or directory
  490 |           #include "vgg_generated_120.i"
      |                    ^~~~~~~~~~~~~~~~~~~~~
compilation terminated.
make[2]: *** [modules/xfeatures2d/CMakeFiles/opencv_xfeatures2d.dir/build.make:354: modules/xfeatures2d/CMakeFiles/opencv_xfeatures2d.dir/src/vgg.cpp.o] Error 1
make[2]: *** Waiting for unfinished jobs....
[ 88%] Building CXX object modules/ximgproc/CMakeFiles/opencv_ximgproc.dir/src/ridgedetectionfilter.cpp.o
/liuxianguo1/XYX/Others/opencv_contrib-3.4.9/modules/xfeatures2d/src/boostdesc.cpp:654:20: fatal error: boostdesc_bgm.i: No such file or directory
  654 |           #include "boostdesc_bgm.i"
      |                    ^~~~~~~~~~~~~~~~~
```

方案：

```
# 下载配置文件.zip
https://pan.baidu.com/s/1Xt-gxruQLFmTMLzQd0MRNw  # OpenCV-Contrib  Password : lyh1

将其放置在 opencv_contrib/modules/xfeatures2d/src/
```

### Include不到文件

```
/liuxianguo1/XYX/Others/opencv-3.4.9/modules/stitching/include/opencv2/stitching/detail/matchers.hpp:52:12: fatal error: opencv2/xfeatures2d/cuda.hpp: No such file or directory
   52 | #  include "opencv2/xfeatures2d/cuda.hpp"
   
   /liuxianguo1/XYX/Others/opencv-3.4.9/modules/stitching/src/precomp.hpp:91:12: fatal error: opencv2/xfeatures2d/cuda.hpp: No such file or directory
   91 | #  include "opencv2/xfeatures2d/cuda.hpp"
   
   /liuxianguo1/XYX/Others/opencv-3.4.9/modules/stitching/src/matchers.cpp:52:10: fatal error: opencv2/xfeatures2d.hpp: No such file or directory
   52 | #include "opencv2/xfeatures2d.hpp"
      |          ^~~~~~~~~~~~~~~~~~~~~~~~~
      
      /liuxianguo1/XYX/Others/opencv_contrib-3.4.9/modules/xfeatures2d/include/opencv2/xfeatures2d.hpp:42:10: fatal error: opencv2/xfeatures2d.hpp: No such file or directory
   42 | #include "opencv2/xfeatures2d.hpp"
```

解决方案：

将路径换位绝对路径，缺少的文件基本都在`/liuxianguo1/XYX/Others/opencv_contrib-3.4.9/modules/xfeatures2d/include/opencv2/`

```
# 例子
#include"/opencv2/xfeatures2d.hpp"  ==>  
#include"/liuxianguo1/XYX/Others/opencv_contrib-3.4.9/modules/xfeatures2d/include/opencv2/xfeatures2d.hpp"
```



# Openpose构建

1. 安装依赖库

    ```
    sudo apt-get --assume-yes install build-essential
    
    sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libboost-all-dev libhdf5-serial-dev protobuf-compiler
    
    sudo apt-get install --no-install-recommends libboost-all-dev
    
    sudo apt-get install libopenblas-dev liblapack-dev libatlas-base-dev
    
    sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
    ```

2. 下载Openpose并配置model

    ```cmd
    # Github页面
    https://github.com/CMU-Perceptual-Computing-Lab/openpose
    
    git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
    
    cd openpose
    
    # 下载models并替换openpose/models的face、hand、pose
    https://pan.baidu.com/s/19iyTQUV5vFuV3EKMMaBlcg?pwd=al6y
    ```

3. 编译

    ```
    # 将Openpose/CMakeLists.txt 所有 USE_CUDNN ON ==>  USE_CUDNN OFF
    nano CMakeLists.txt
    
    mkdir build
    
    cd build
    
    cmake .. -DBUILD_PYTHON=ON
    
    make -j `nproc`
    ```

4. 链接python(可选)

    * 注意：只会在

    ```
    cd python
    
    make -j `nproc`
    
    sudo make install
    ```

    











# 参考

1. [Linux服务器下openpose搭建_openpose linux-CSDN博客](https://blog.csdn.net/qq_42921511/article/details/121485678)
2. [Linux非管理员安装cmake以及所遇到的坑_非管理员centos安装cmake-CSDN博客](https://blog.csdn.net/qq_42921511/article/details/120584569?spm=1001.2014.3001.5501)
3. [Linux无root权限安装opencv_opencv3.4.1安装 linux-CSDN博客](https://blog.csdn.net/qq_42921511/article/details/120611683)
4. [opencv_contrib 下载驿站（百度云盘下载）_opencv_contrib-3.4.4.tar.gz-CSDN博客](https://blog.csdn.net/weijifen000/article/details/87904707)
5. [ubuntu下命令行安装openpose并使用python调用_openpose ubuntu-CSDN博客](https://blog.csdn.net/qq_41491230/article/details/127906256#:~:text=步骤如下： 1. 获取 openpose git clone https%3A%2F%2Fgithub.com%2FCMU-Perceptual-Computing-Lab%2Fopenpose.git,2. 安装cmake apt install cmake build-essential 3.进入openpose%2C建立build文件夹并使用cmake编译，注意设置参数-DBUILD_PYTHON%3DON后续才可以使用python调用。)







