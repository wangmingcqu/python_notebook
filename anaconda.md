### 一、conda命令

1、查看当前channel

```shell
conda info
```

2、创建一个新的环境

```shell
conda create -n 环境名
#下述命令会自动安装常用的包
conda create -n 环境名 anaconda
#使用下述命令会自动指定python的版本
conda create -n 环境名 python=3.4

```

3、查看conda所有的环境

```shell
conda info --envs
conda env lsit
```

4、激活环境

```shell
Conda activate 环境名
Source activate 环境名
activate + 环境名
```

5、在你的环境中使用conda或者pip安装包

```shell
Conda install 包名称
或者pip install 包名称 -i https://pypi.tuna.tsinghua.edu.cn/simple（清华镜像）
或者pip install 包名称 -i  https://pypi.doubanio.com/simple/ （豆瓣镜像）

# pip uninstall 要卸载的包名
# 下面以卸载seaborn包为例
pip uninstall seaborn
```

6、查看环境中的包

```shell
conda list
```

7、在环境中运行python程序(windows)

```sehll
//1、切换目录到文件所在目录
①cd + 盘符号
cd F
②cd + 目录
cd F:\示例 切换到示例目录
③输入python +文件名
python 1.py
```

8、退出当前环境

```shell
deactivate 环境名
或者 直接deactivate
```

9、删除环境

```shell
conda remove -n 环境名 --all
```

10、安装相关的paceage

```shell
conda install gensim
```

11、在新环境中安装jupyter

```shell
#默认jupyter安装在了base环境中
#新环境要使用的话必须重新安装一遍
conda install nb_conda
```

12、启动jupyter

```shell
#激活环境之后
jupyter notebook
```

13、直接在anaconda启动python

```shell
#启动python，直接输入
python
#退出python
quit()
```

14、查看显卡驱动

```shell
nvidia-smi
```

15、Anaconda如何换成国内清华源/恢复默认源

```shell
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
# 搜索时显示通道地址（建议添加）
conda config --set show_channel_urls yes


#恢复官方默认源
conda config --remove-key channels
```

 



### 二、基础知识

CUDA（Compute Unified Device Architecture）是NVIDIA推出的并行计算框架。

GPU与pytorch之间转换指令、传递数据的接口——CUDA和显卡驱动。

CUDA负责将pytorch中编好的程序信息整理并且传递给显卡驱动，显卡驱动负责将CUDA发出的指令进行编译，传递给底层层GPU；

即传递链路是：

pytorch ——>CUDA ——>显卡驱动——>GPU。



通常情况下的安装路线：

确保有硬件`GPU`,根据GPU型号安装兼容版本的`GPU驱动`,根据GPU驱动安装兼容版本的`CUDA`,  根据CUDA安装兼容版本的`pytorch/tensorflow`



