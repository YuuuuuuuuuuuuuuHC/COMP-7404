# MicroISP

## Introduction

A Pytorch implementation of MicroISP for the paper: [MicroISP: Processing 32MP Photos on Mobile Devices with Deep Learning](https://arxiv.org/pdf/2211.06770.pdf).

## Dataset

Please download the dataset [here](https://drive.google.com/file/d/1QIIf9GZjIREnCgaBfSEIcHExQqvzobq2/view?usp=share_link).

## Code Structure

~~~
my_microisp
│  README.md
│  test.py
│  train.py
│  visual.py
│
├─checkpoints
│      microisp_epoch_130.pth
│
├─data
│      load_data.py
│
├─dataset
├─logs
├─models
│      microisp.py
│      vgg.py
│
├─results
└─utils
        ssim.py
        utils.py
~~~

## Usage

### Train

~~~
python train.py [num_train_epochs=] [batch_size=] [learning_rate=] [restore_epoch=] [dataset_dir=]
~~~

### Test

~~~
python test.py [batch_size=] [restore_epoch=] [dataset_dir=]
~~~

### Visual

~~~
python visual.py [restore_epoch=] [dataset_dir=]
~~~

