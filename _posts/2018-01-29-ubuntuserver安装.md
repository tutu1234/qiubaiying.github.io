---
layout:     post
title:      ubuntu server系统安装
subtitle:   OS install on a server
date:       2018-01-29
author:     Hututu
header-img: img/post-bg-2015.jpg
catalog: true
tags:
    - OS
---
## ubuntu server系统安装
### 服务器的系统用`centos`还是`ubuntu`
1. **`Ubuntu/Debian` 的优势**
	+ apt-get 确实比 yum 好用。
	+ 软件包数量多，功能性版本更新快。
	+ 因为上条，研发使用 Ubuntu 比较多，环境能无缝切换。
	+ Ubuntu 对 Linux 初学者非常友好。
2. **`Ubuntu/Debian`的劣势**
	- `Ubuntu`补丁更新慢,官方只管放包不管修
	- `Debian`生命周期不固定，新版本发布以后，上个版本再维护`18`个月，一般生命周期在`5`年。当生命周期过了以后，就没有安全补丁，你的服务器就会裸奔或需要重新安装系统。
3. **`RHEL/CentOS`的优势**
	- `CentOS/RHEL`的生命周期是`7`年，基本可以覆盖硬件的生命周期。`RedHat 5、RedHat 6`的生命周期，延长到`10`年。
	- `CentOS/RHEL`对硬件支持很好，主流硬件厂商早就将服务器拿过去测试，一般不存在硬件兼容性问题。
	- 大量商业软件，比如`Oracle`，都是针对`Redhat`认证的，有大量的帮助文档和使用说明，有良好的技术支持。出了问题，也容易在网上找到类似的答案和经验。
	- Redhat 对安全漏洞的响应更及时。
4. **`RHEL/CentOS`**的劣势
	- 由于生命周期长，线上系统往往版本老，最近的软件在`RH`的官方库里找不到。只能自己编译。还得专人负责补丁更新。
	- 如果研发多使用`Ubuntu`，线上生成环境使用`RHEL`，环境的变更会需要更多的调试，大幅影响研发效率。
5. **结论：**深度学习框架的版本更新速度非常快，推荐使用**`Ubuntu`**，避免`centos`系统某些库由于版本低不支持的麻烦。

### U盘启动盘制作
1. 下载系统镜像文件，校园网可以去北邮人下载，或者去[Ubuntu官网](https://www.ubuntu.com/download?_ga=2.120919240.1958115685.1517229271-1308373443.1499333241)
2. 参考[ubuntu官网教程](https://tutorials.ubuntu.com/tutorial/tutorial-create-a-usb-stick-on-ubuntu#0),制作`U盘`启动盘。
3. 准备`u盘`，使用**`Startup Disk Creator`**工具
![](http://ww1.sinaimg.cn/large/8833244fly1fnxrdawiooj20hg0940tj.jpg)
4. 点击`Other...`选择下载的镜像文件，点击**`Make Startup Disk `**开始制作
![](http://ww1.sinaimg.cn/large/8833244fly1fnxrg31hecj20ng0ggac4.jpg)
5. 等待完成，大功告成！！！

### 服务器系统安装
1. 插上`U盘`，启动按`delete/F12`进入`BIOS`
2. 选择`Boot`选项，从`U盘`启动
3. 安装过程参考[教程](http://blog.topspeedsnail.com/archives/4511)，值得注意的一步是：**选择要安装的服务；我只选项standard system utilities和openssh服务，这是两个最基本的东西，其他服务可以以后在装,空格键选择，回车继续**
![](http://ww1.sinaimg.cn/large/8833244fly1fnxrl69wo0j20m50gm74r.jpg)
4. 安装之后的网络配置注意设置静态`IP`的过程：
	- 获取`root`权限， `sudo -s`
	- 设置`root`密码： ` passwd root`
	- 安装vim： `sudo apt-get install vim`
	- 配置网络文件: `vim /etc/network/interfaces`
	```
    # The loopback network interface
    auto lo
    iface lo inet loopback
    # The primary network interface 注意替换网络接口名(ifconfig查看，每台机器不同)
    auto ens33
    iface ens33 inet static
            address 192.168.1.100
            netmask 255.255.255.0
            network 192.168.1.0（（可以注释掉）
            broadcast 192.168.1.255（可以注释掉）
            gateway 192.168.1.1 （一定要对，不然ping www.baidu.com或者不是一个网段的ip不成功）
            dns-nameservers 8.8.8.8 8.8.4.4
    ```
    
	- 设置`DNS`: `sudo vim /etc/resolv.conf`,插入`nameserver 8.8.8.8`
	- 重启网络服务：`service networking restart 或者 /etc/init.d/networking restart`

### 镜像源配置和软件安装
1. 更换[清华镜像源](https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/),根据所在地区可以选择阿里云和中科大或者其他镜像源。
```
sudo cp /etc/apt/sources.list  /etc/apt/sources.list.bak
sudo vim /etc/apt/sources.list 
删除原文件内容并黏贴新的镜像源
sudo update
sudo upgrade
```
2. 安装配置tmux, htop, git等常用工具
```
安装：sudo apt-get install tmux htop git
```
配置：拷贝[vimrc](http://note.youdao.com/noteshare?id=57100eefbc9f03e1ba600a97dda0d2fa&sub=EFC46B0D72B549E19651C2D7BBDE924F),[.tmux.conf](http://note.youdao.com/noteshare?id=482f19185366941f2945778a89fb5824&sub=543AB1818AF34B7493D2EBA643340427)文件到主目录
3. 安装anaconda
```
    sudo sh Anaconda3-5.0.1-Linux-x86_64
    conda create -n py36 python=3.6
    souece activate py36 (可加到.bashrc中)
```
5. 安装opencv(如3.2.0版本)
    - 安装基本编译工具：
    ```
    sudo apt-get install build-essential cmake pkg-config
    ```
    - 由于OpenCV是计算机视觉库，总需要加载一些图像文件（ JPEG, PNG, TIFF）。使用下面命令安装一些必要的图像库：
    ```
    sudo apt-get install libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev
    ```
    - 除了图片之外，OpenCV还要处理视频文件。使用下面命令安装一些视频编解码库：
    ```
    sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
    sudo apt-get install libxvidcore-dev libx264-dev
    ```
    - OpenCV的GUI模块highgui依赖Gtk。安装gtk库：
    ```
    sudo apt-get install libgtk-3-dev
    ```
    - 下面安装一些可以提高OpenCV性能的库，如矩阵操作：
    ```
    sudo apt-get install libatlas-base-dev gfortran
    ```

    ```
    unzip opencv-3.2.0.zip
    cd opencv-3.2.0
    mkdir build
    cd build
    cmake ..
    sudo make -j8
    sudo make install
    sudo ldconfig
    ```

### 引用
> https://echohn.github.io/2016/02/03/server-os-choose-rhel-cenos-or-debian-ubuntu/
> https://tutorials.ubuntu.com/tutorial/tutorial-create-a-usb-stick-on-ubuntu#0
> http://blog.topspeedsnail.com/archives/4511