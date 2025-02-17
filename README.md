# 為了多圖輸入所做的修改
[cls_cvt.py](./lib/models/cls_cvt.py) 修改了模型架構，

將原先的輸入維度[B, 3, H, W] 改為 [B, D, 3, H, W] ，其中D為維度以用作多圖輸入。

模型則修改為將每個維度分別進行原本的CvT流程後再透過MergeAttention使用注意力機制進行融合。

為了進行相應的數據讀取也修改了相應的[檔案](./lib/dataset/build.py) ，當 DATASET.DATASET 為'multidim_imagenet' 時便使用修改的多圖讀取的方法；為原來的'imagenet'時則使用完來的方法

# 環境架設及使用
## conda
minaconda 安裝  https://docs.anaconda.com/miniconda/install/#quick-command-line-install
```sh
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

conda 常用指令
```sh
conda create -n CvT python=3.9
source ~/miniconda3/bin/activate
conda activate CvT
conda deactivate
```
```sh
python3 -m pip install -r requirements.txt
```

## git&linux (for me 常用指令)
```sh
git remote set-url origin https://github.com/violetzu/CvT.git
git config user.email "liuzii706@gmail.com"
git config user.name "violetzu"
```
```sh
rm -rf ~/miniconda3 (-rf 用作移除目錄及檔案)
source ~/.bashrc
tensorboard --logdir=OUTPUT/imagenet/cvt-13-224x224/
```

## CvT腳本指令
```sh
Usage: run.sh [run_options]
Options:
  -g|--gpus <1> - number of gpus to be used
  -t|--job-type <aml> - job type (train|test)
  -p|--port <9000> - master port
  -i|--install-deps - If install dependencies (default: False)
```
訓練腳本使用範例
```sh
bash run.sh -g 1 -t train --cfg experiments/imagenet/cvt/cvt-13-224x224.yaml
bash run.sh -g 1 -t train --cfg experiments/imagenet/cvt/cvt-dim6-310x321.yaml
tensorboard --logdir=<包含這個檔案的資料夾>

tensorboard --logdir=OUTPUT/multidim_imagenet/cvt-dim6-310x321
```
測試腳本使用範例
```sh
export PRETRAINED_MODEL_FILE=/home/st424/CvT/OUTPUT/imagenet/cvt-13-224x224/model_best.pth
bash run.sh -t test --cfg experiments/imagenet/cvt/cvt-13-224x224.yaml TEST.MODEL_FILE ${PRETRAINED_MODLE_FILE}
```
等價
```sh
bash run.sh -t test --cfg experiments/imagenet/cvt/cvt-13-224x224.yaml TEST.MODEL_FILE /home/st424/CvT/OUTPUT/imagenet/cvt-13-224x224/model_best.pth
```

## Ubuntu20.04 gcc版本不足
```sh
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
``` 

結果未有3.4.25 則執行:
```sh
sudo apt install gcc-11 g++-11
``` 

未有gcc-11 g++-11:
添加PPA 
```sh
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
```

## Torch 1.7.1 +CUDA11
```sh
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```
確認pytorch可使用CUDA
```sh
python3 -c "import torch; print(torch.cuda.is_available())"
```


##　CUDA 11.0 for Ubuntu20.04
https://developer.nvidia.com/cuda-11.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=2004&target_type=deblocal
```sh
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget http://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda-repo-ubuntu2004-11-0-local_11.0.2-450.51.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-0-local_11.0.2-450.51.05-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-0-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```
```sh
nvcc --version
```
如果目前指向的是舊版本的 CUDA，請執行以下命令更新環境變數：
```sh
export PATH=/usr/local/cuda-11.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH
```

永久更新環境變數
如果你希望每次啟動時自動加載 CUDA 11.0，可以將上述環境變數加入到 ~/.bashrc：
```sh
echo 'export PATH=/usr/local/cuda-11.0/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```



## 筆記指令 for me
markdown preview快捷鍵
ctrl-shift-v

使用 pipreqs生成實際用到的 requirements.txt

假设你的项目目录是 my_project，运行以下命令：
这将在项目目录下生成一个只包含实际导入包的 requirements.txt。
如果你希望覆盖现有的 requirements.txt，可以加上 --force：
```sh
pipreqs /path/to/your/project --force
```
```sh
pip install jupyter pipreqs
```
```sh
jupyter nbconvert --to script your_notebook.ipynb
```








