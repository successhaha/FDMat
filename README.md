# distribution--------match
Data-feature Distribution Matching by optimal transport for Eﬀicient and
Robust Dataset Selection


### Datasets
It contains a series of other popular computer vision datasets, namely MNIST, QMNIST, FashionMNIST, SVHN, CIFAR10, CIFAR100 and TinyImageNet and ImageNet.

### Models
They are two-layer fully connected MLP, LeNet , AlexNet, VGG, Inception-v3, ResNet, WideResNet and MobileNet-v3.

### Example
Selecting with Glister and training on the coreset with fraction 0.1.

CUDA_VISIBLE_DEVICES=0 python -u main.py --fraction 0.1 --dataset CIFAR10 --data_path ~/datasets --num_exp 5 --workers 10 --optimizer SGD -se 10 --selection Glister --model InceptionV3 --lr 0.1 -sp ./result --batch 128
