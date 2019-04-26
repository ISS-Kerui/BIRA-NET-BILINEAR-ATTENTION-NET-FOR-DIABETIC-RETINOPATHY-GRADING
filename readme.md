# BIRA-NET: BILINEAR ATTENTION NET FOR DIABETIC RETINOPATHY GRADING

# Introduction

The proposed BiRA-Net is presented in this Section for DR prediction. The proposed BiRA-Net architecture is shown in Fig. 1, which consists of three key components: (i) ResNet, (ii) Attention Net and (iii) Bilinear Net. First, the processed images are put into the ResNet for feature extraction; then
the Attention Net is applied to concentrate on the suspected area. For more fine-grained classification in this task, a bilinear strategy is adopted, where two RA-Net are trained simultaneously to improve the performance of classification. ![QQ20190426-212322@2x](/Users/zkr/Desktop/QQ20190426-212322@2x.png)

### Installation

The code was tested with Anaconda and Python 2.7. After installing the Anaconda environment:

Clone the repo:

``

```
git clone https://github.com/jfzhang95/pytorch-deeplab-xception.git
cd pytorch-deeplab-xception
```

Install dependencies:

`tensorboardX`

`Pytorch==4.1`

`Numpy`

Download dataset:

Link: https://pan.baidu.com/s/1nz9lMQtgjJbyzjbf22DWyg   password: hqjm 

## Training

Fellow commands below to train the model:

```
usage: train.py [-h] [--batch-size 30]
            [--epochs 10] [--lr 0.0002]
            [--no-cuda] [--seed 1] [--save-epoch 5]
            [--weight-decay 1e-8] [--image-size 610]
```

