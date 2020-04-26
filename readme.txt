如果运行./hello，显示Max error:0.0000000000说明cuda安装正常

tensorflow 2.2 no pair cuda10.2 cuda10.0 only to cuda10.1
先pip3安装tensorflow-gpu，然后运行一下，看看这个软件的版本再觉定安哪个版本的cuda
 opened dynamic library libcudart.so.10.1
 从这里可以看出，它调用了10.1的cuda，所以一定要安装对应版本的cuda，否则tensorflow-gpu运行不起

step.1
wget http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_418.87.00_linux.run
sudo sh cuda_10.1.243_418.87.00_linux.run

由于418的驱动安装不了，所以用440的驱动单独先安装，再安的cuda，安装时不选驱动
ubuntu-drivers devices

step.2 复制cuDNN内容到cuda相关文件夹内

sudo cp cuda/include/cudnn.h    /usr/local/cuda/include      注意，解压后的文件夹名称为cuda ,将对应文件复制到 /usr/local中的cuda内
sudo cp cuda/lib64/libcudnn*    /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h   /usr/local/cuda/lib64/libcudnn*

查看cuda安装正确与否
use nvcc the Nvidia CUDA compiler to compile the code and run the newly compiled binary:

$ nvcc -o hello hello.cu 
$ ./hello 
Max error: 0.000000


.卸载驱动
停X Server
sudo service lightdm stop
sudo /usr/bin/nvidia-uninstall
卸载cuda
cuda的默认安装在 /usr/local/cuda-10.2下，用下面的命令卸载：
sudo /usr/local/cuda-10.2/bin/cuda-uninstall

sudo apt-get --purge remove nvidia*

sudo apt autoremove
To remove CUDA Toolkit:

$ sudo apt-get --purge remove "*cublas*" "cuda*"

To remove NVIDIA Drivers:

$ sudo apt-get --purge remove "*nvidia*"
