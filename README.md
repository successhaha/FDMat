###
Feature Distribution Matching by Optimal Transport for Effective and Robust Coreset Selection
### Introduction
To advance the research of coreset selection in deep learning, we follow a code library named **DeepCore**[14], an extensive and extendable code library, for coreset selection in deep learning, reproducing dozens of popular and advanced coreset selection methods and enabling a fair comparison of different methods in the same experimental settings. **DeepCore** is highly modular, allowing to add new architectures, datasets, methods and learning scenarios easily. It is built on PyTorch.   

### Coreset Methods
We list the methods in DeepCore, they are 1) geometry based methods Contextual Diversity (CD), Herding  and k-Center Greedy; 2) uncertainty scores; 3) error based methods Forgetting  and GraNd score ; 4) decision boundary based methods Cal  and DeepFool ; 5) gradient matching based methods Craig, GradMatch and LCMat ; 6) bilevel optimiza- tion methods Glister ; and 7) Submodularity based Methods (GC) and Facility Location (FL) functions. we also have Random selection as the baseline.
### Datasets
It contains a series of other popular computer vision datasets, namely MNIST, CINIC, QMNIST, FashionMNIST, SVHN, CIFAR10, CIFAR100, TinyImageNet and ImageNet.

### Example
Selecting with FDMat and training on the coreset with fraction 0.1.
```sh
CUDA_VISIBLE_DEVICES=0 python -u main.py --fraction 0.1 --dataset CIFAR10 --data_path ~/datasets --num_exp 5 --workers 10 --optimizer SGD -se 10 --selection mymatch_2 --model ResNet18 --lr 0.1 -sp ./result --batch 128
```
