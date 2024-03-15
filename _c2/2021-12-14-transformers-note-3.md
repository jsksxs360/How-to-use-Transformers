---
title: 第六章：必要的 Pytorch 知识
author: SHENG XU
date: 2021-12-14
category: NLP
layout: post
---

在上一章中，我们介绍了 `Model` 类和 `Tokenizers` 类，尤其是如何运用分词器对文本进行预处理。

Transformers 库建立在 Pytorch 框架之上（Tensorflow 的版本功能并不完善），虽然官方宣称使用 Transformers 库并不需要掌握  Pytorch 知识，但是实际上我们还是需要通过 Pytorch 的 `DataLoader` 类来加载数据、使用 Pytorch 的优化器对模型参数进行调整等等。

因此，本章将介绍 Pytorch 的一些基础概念以及后续可能会使用到的类，让大家可以快速上手使用 Transformers 库建立模型。

## 1. Pytorch 基础

[Pytorch](https://pytorch.org/) 由 Facebook 人工智能研究院于 2017 年推出，具有强大的 GPU 加速张量计算功能，并且能够自动进行微分计算，从而可以使用基于梯度的方法对模型参数进行优化。截至 2022 年 8 月，PyTorch 已经和 Linux 内核、Kubernetes 等并列成为世界上增长最快的 5 个开源社区之一。现在在 NeurIPS、ICML 等等机器学习顶会中，有超过 80% 研究人员用的都是 PyTorch。

> 为了确保商业化和技术治理之间的相互独立，2022 年 9 月 12 日 PyTorch 官方宣布正式加入 Linux 基金会，PyTorch 基金会董事会包括 Meta、AMD、亚马逊、谷歌、微软、Nvidia 等公司。

### 张量

张量 (Tensor) 是深度学习的基础，例如常见的 0 维张量称为标量 (scalar)、1 维张量称为向量 (vector)、2 维张量称为矩阵 (matrix)。Pytorch 本质上就是一个基于张量的数学计算工具包，它提供了多种方式来创建张量：

```python
>>> import torch
>>> torch.empty(2, 3) # empty tensor (uninitialized), shape (2,3)
tensor([[2.7508e+23, 4.3546e+27, 7.5571e+31],
        [2.0283e-19, 3.0981e+32, 1.8496e+20]])
>>> torch.rand(2, 3) # random tensor, each value taken from [0,1)
tensor([[0.8892, 0.2503, 0.2827],
        [0.9474, 0.5373, 0.4672]])
>>> torch.randn(2, 3) # random tensor, each value taken from standard normal distribution
tensor([[-0.4541, -1.1986,  0.1952],
        [ 0.9518,  1.3268, -0.4778]])
>>> torch.zeros(2, 3, dtype=torch.long) # long integer zero tensor
tensor([[0, 0, 0],
        [0, 0, 0]])
>>> torch.zeros(2, 3, dtype=torch.double) # double float zero tensor
tensor([[0., 0., 0.],
        [0., 0., 0.]], dtype=torch.float64)
>>> torch.arange(10)
tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

也可以通过 `torch.tensor()` 或者 `torch.from_numpy()` 基于已有的数组或 Numpy 数组创建张量：

```python
>>> array = [[1.0, 3.8, 2.1], [8.6, 4.0, 2.4]]
>>> torch.tensor(array)
tensor([[1.0000, 3.8000, 2.1000],
        [8.6000, 4.0000, 2.4000]])
>>> import numpy as np
>>> array = np.array([[1.0, 3.8, 2.1], [8.6, 4.0, 2.4]])
>>> torch.from_numpy(array)
tensor([[1.0000, 3.8000, 2.1000],
        [8.6000, 4.0000, 2.4000]], dtype=torch.float64)
```

**注意：**上面这些方式创建的张量会存储在内存中并使用 CPU 进行计算，如果想要调用 GPU 计算，需要直接在 GPU 中创建张量或者将张量送入到 GPU 中：

```python
>>> torch.rand(2, 3).cuda()
tensor([[0.0405, 0.1489, 0.8197],
        [0.9589, 0.0379, 0.5734]], device='cuda:0')
>>> torch.rand(2, 3, device="cuda")
tensor([[0.0405, 0.1489, 0.8197],
        [0.9589, 0.0379, 0.5734]], device='cuda:0')
>>> torch.rand(2, 3).to("cuda")
tensor([[0.9474, 0.7882, 0.3053],
        [0.6759, 0.1196, 0.7484]], device='cuda:0')
```

在后续章节中，我们经常会将编码后的文本张量通过 `to(device)` 送入到指定的 GPU 或 CPU 中。

### 张量计算

张量的加减乘除是按元素进行计算的，例如：

```python
>>> x = torch.tensor([1, 2, 3], dtype=torch.double)
>>> y = torch.tensor([4, 5, 6], dtype=torch.double)
>>> print(x + y)
tensor([5., 7., 9.], dtype=torch.float64)
>>> print(x - y)
tensor([-3., -3., -3.], dtype=torch.float64)
>>> print(x * y)
tensor([ 4., 10., 18.], dtype=torch.float64)
>>> print(x / y)
tensor([0.2500, 0.4000, 0.5000], dtype=torch.float64)
```

Pytorch 还提供了许多常用的计算函数，如 `torch.dot()` 计算向量点积、`torch.mm()` 计算矩阵相乘、三角函数和各种数学函数等：

```python
>>> x.dot(y)
tensor(32., dtype=torch.float64)
>>> x.sin()
tensor([0.8415, 0.9093, 0.1411], dtype=torch.float64)
>>> x.exp()
tensor([ 2.7183,  7.3891, 20.0855], dtype=torch.float64)
```

除了数学运算，Pytorch 还提供了多种张量操作函数，如聚合 (aggregation)、拼接 (concatenation)、比较、随机采样、序列化等，详细使用方法可以参见 [Pytorch 官方文档](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)。

> 对张量进行聚合（如求平均、求和、最大值和最小值等）或拼接操作时，可以指定进行操作的维度 (dim)。例如，计算张量的平均值，在默认情况下会计算所有元素的平均值。：
>
> ```python
> >>> x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.double)
> >>> x.mean()
> tensor(3.5000, dtype=torch.float64)
> ```
>
> 更常见的情况是需要计算某一行或某一列的平均值，此时就需要设定计算的维度，例如分别对第 0 维和第 1 维计算平均值：
>
> ```python
> >>> x.mean(dim=0)
> tensor([2.5000, 3.5000, 4.5000], dtype=torch.float64)
> >>> x.mean(dim=1)
> tensor([2., 5.], dtype=torch.float64)
> ```
>
> 注意，上面的计算自动去除了多余的维度，因此结果从矩阵变成了向量，如果要保持维度不变，可以设置 `keepdim=True`：
>
> ```python
> >>> x.mean(dim=0, keepdim=True)
> tensor([[2.5000, 3.5000, 4.5000]], dtype=torch.float64)
> >>> x.mean(dim=1, keepdim=True)
> tensor([[2.],
>         [5.]], dtype=torch.float64)
> ```
>
> 拼接 `torch.cat` 操作类似，通过指定拼接维度，可以获得不同的拼接结果：
>
> ```python
> >>> x = torch.tensor([[1, 2, 3], [ 4,  5,  6]], dtype=torch.double)
> >>> y = torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.double)
> >>> torch.cat((x, y), dim=0)
> tensor([[ 1.,  2.,  3.],
>         [ 4.,  5.,  6.],
>         [ 7.,  8.,  9.],
>         [10., 11., 12.]], dtype=torch.float64)
> >>> torch.cat((x, y), dim=1)
> tensor([[ 1.,  2.,  3.,  7.,  8.,  9.],
>         [ 4.,  5.,  6., 10., 11., 12.]], dtype=torch.float64)
> ```

组合使用这些操作就可以写出复杂的数学计算表达式。例如对于

$$
z = (x + y) \times (y - 2)
$$

当 $x=2,y=3$ 时，很容易计算出 $z=5$。使用 Pytorch 来实现这一计算过程与 Python 非常类似，唯一的不同是数据使用张量进行保存：

```python
>>> x = torch.tensor([2.])
>>> y = torch.tensor([3.])
>>> z = (x + y) * (y - 2)
>>> print(z)
tensor([5.])
```

使用 Pytorch 进行计算的好处是更高效的执行速度，尤其当张量存储的数据很多时，而且还可以借助 GPU 进一步提高计算速度。下面以计算三个矩阵相乘的结果为例，我们分别通过 CPU 和 NVIDIA Tesla V100 GPU 来进行：

```python
import torch
import timeit

M = torch.rand(1000, 1000)
print(timeit.timeit(lambda: M.mm(M).mm(M), number=5000))

N = torch.rand(1000, 1000).cuda()
print(timeit.timeit(lambda: N.mm(N).mm(N), number=5000))
```

```
77.78975469999999
1.6584811117500067
```

可以看到使用 GPU 能够明显地提高计算效率。

### 自动微分

Pytorch 提供自动计算梯度的功能，可以自动计算一个函数关于一个变量在某一取值下的导数，从而基于梯度对参数进行优化，这就是机器学习中的训练过程。使用 Pytorch 计算梯度非常容易，只需要执行 `tensor.backward()`，就会自动通过反向传播 (Back Propogation) 算法完成，后面我们在训练模型时就会用到该函数。

注意，为了计算一个函数关于某一变量的导数，Pytorch 要求显式地设置该变量是可求导的，即在张量生成时，设置 `requires_grad=True`。我们对上面计算 $z = (x + y) \times (y - 2)$ 的代码进行简单修改，就可以计算当 $x=2,y=3$ 时，$\frac{\text{d}z}{\text{d}x}$ 和 $\frac{\text{d}z}{\text{d}y}$ 的值。

```python
>>> x = torch.tensor([2.], requires_grad=True)
>>> y = torch.tensor([3.], requires_grad=True)
>>> z = (x + y) * (y - 2)
>>> print(z)
tensor([5.], grad_fn=<MulBackward0>)
>>> z.backward()
>>> print(x.grad, y.grad)
tensor([1.]) tensor([6.])
```

很容易手工求解 $\frac{\text{d}z}{\text{d}x} = y-2,\frac{\text{d}z}{\text{d}y} = x + 2y - 2$，当 $x=2,y=3$ 时，$\frac{\text{d}z}{\text{d}x}=1$ 和 $\frac{\text{d}z}{\text{d}y}=6$，与 Pytorch 代码计算结果一致。

### 调整张量形状

有时我们需要对张量的形状进行调整，Pytorch 共提供了 4 种调整张量形状的函数，分别为：

- **形状转换 `view`** 将张量转换为新的形状，需要保证总的元素个数不变，例如：

  ```python
  >>> x = torch.tensor([1, 2, 3, 4, 5, 6])
  >>> print(x, x.shape)
  tensor([1, 2, 3, 4, 5, 6]) torch.Size([6])
  >>> x.view(2, 3) # shape adjusted to (2, 3)
  tensor([[1, 2, 3],
          [4, 5, 6]])
  >>> x.view(3, 2) # shape adjusted to (3, 2)
  tensor([[1, 2],
          [3, 4],
          [5, 6]])
  >>> x.view(-1, 3) # -1 means automatic inference
  tensor([[1, 2, 3],
          [4, 5, 6]])
  ```

  进行 view 操作的张量必须是连续的 (contiguous)，可以调用 `is_conuous` 来判断张量是否连续；如果非连续，需要先通过 `contiguous` 函数将其变为连续的。也可以直接调用 Pytorch 新提供的 `reshape` 函数，它与 `view` 功能几乎一致，并且能够自动处理非连续张量。

- **转置 `transpose`** 交换张量中的两个维度，参数为相应的维度：

  ```python
  >>> x = torch.tensor([[1, 2, 3], [4, 5, 6]])
  >>> x
  tensor([[1, 2, 3],
          [4, 5, 6]])
  >>> x.transpose(0, 1)
  tensor([[1, 4],
          [2, 5],
          [3, 6]])
  ```

- **交换维度 `permute`** 与 `transpose` 函数每次只能交换两个维度不同，`permute` 可以直接设置新的维度排列方式：

  ```python
  >>> x = torch.tensor([[[1, 2, 3], [4, 5, 6]]])
  >>> print(x, x.shape)
  tensor([[[1, 2, 3],
           [4, 5, 6]]]) torch.Size([1, 2, 3])
  >>> x = x.permute(2, 0, 1)
  >>> print(x, x.shape)
  tensor([[[1, 4]],
  
          [[2, 5]],
  
          [[3, 6]]]) torch.Size([3, 1, 2])
  ```

### 广播机制

前面我们都是假设参与运算的两个张量形状相同。在有些情况下，即使两个张量形状不同，也可以通过广播机制 (broadcasting mechanism) 对其中一个或者同时对两个张量的元素进行复制，使得它们形状相同，然后再执行按元素计算。

例如，我们生成两个形状不同的张量：

```python
>>> x = torch.arange(1, 4).view(3, 1)
>>> y = torch.arange(4, 6).view(1, 2)
>>> print(x)
tensor([[1],
        [2],
        [3]])
>>> print(y)
tensor([[4, 5]])
```

它们形状分别为 $(3, 1)$ 和 $(1, 2)$，如果要进行按元素运算，必须将它们都扩展为形状 $(3, 2)$ 的张量。具体地，就是将 $x$ 的第 1 列复制到第 2 列，将 $y$ 的第 1 行复制到第 2、3 行。实际上，我们可以直接进行运算，Pytorch 会自动执行广播：

```python
>>> print(x + y)
tensor([[5, 6],
        [6, 7],
        [7, 8]])
```

### 索引与切片

与 Python 列表类似，Pytorch 也可以对张量进行索引和切片。索引值同样是从 0 开始，切片 $[m:n]$ 的范围是从 $m$ 到 $n$ 前一个元素结束，并且可以对张量的任意一个维度进行索引或切片。例如：

```python
>>> x = torch.arange(12).view(3, 4)
>>> x
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
>>> x[1, 3] # element at row 1, column 3
tensor(7)
>>> x[1] # all elements in row 1
tensor([4, 5, 6, 7])
>>> x[1:3] # elements in row 1 & 2
tensor([[ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
>>> x[:, 2] # all elements in column 2
tensor([ 2,  6, 10])
>>> x[:, 2:4] # elements in column 2 & 3
tensor([[ 2,  3],
        [ 6,  7],
        [10, 11]])
>>> x[:, 2:4] = 100 # set elements in column 2 & 3 to 100
>>> x
tensor([[  0,   1, 100, 100],
        [  4,   5, 100, 100],
        [  8,   9, 100, 100]])
```

### 降维与升维

有时为了计算需要对一个张量进行降维或升维。例如神经网络通常只接受一个批次 (batch) 的样例作为输入，如果只有 1 个输入样例，就需要手工添加一个 batch 维度。具体地：

- **升维** `torch.unsqueeze(input, dim, out=None)` 在输入张量的 dim 位置插入一维，与索引一样，dim 值也可以为负数；
- **降维** `torch.squeeze(input, dim=None, out=None)` 在不指定 dim 时，张量中所有形状为 1 的维度都会被删除，例如 $\text{(A, 1, B, 1, C)}$ 会变成 $\text{(A, B, C)}$；当给定 dim 时，只会删除给定的维度（形状必须为 1），例如对于 $\text{(A, 1, B)}$，`squeeze(input, dim=0)` 会保持张量不变，只有 `squeeze(input, dim=1)` 形状才会变成 $\text{(A, B)}$。

下面是一些示例：

```python
>>> a = torch.tensor([1, 2, 3, 4])
>>> a.shape
torch.Size([4])
>>> b = torch.unsqueeze(a, dim=0)
>>> print(b, b.shape)
tensor([[1, 2, 3, 4]]) torch.Size([1, 4])
>>> b = a.unsqueeze(dim=0) # another way to unsqueeze tensor
>>> print(b, b.shape)
tensor([[1, 2, 3, 4]]) torch.Size([1, 4])
>>> c = b.squeeze()
>>> print(c, c.shape)
tensor([1, 2, 3, 4]) torch.Size([4])
```

## 2. 加载数据

Pytorch 提供了 `DataLoader` 和 `Dataset` 类（或 `IterableDataset`）专门用于处理数据，它们既可以加载 Pytorch 预置的数据集，也可以加载自定义数据。其中数据集类 `Dataset`（或 `IterableDataset`）负责存储样本以及它们对应的标签；数据加载类 `DataLoader` 负责迭代地访问数据集中的样本。

### Dataset

数据集负责存储数据样本，所有的数据集类都必须继承自 `Dataset` 或 `IterableDataset`。具体地，Pytorch 支持两种形式的数据集：

- **映射型 (Map-style) 数据集**

  继承自 `Dataset` 类，表示一个从索引到样本的映射（索引可以不是整数），这样我们就可以方便地通过 `dataset[idx]` 来访问指定索引的样本。这也是目前最常见的数据集类型。映射型数据集必须实现 `__getitem__()` 函数，其负责根据指定的 key 返回对应的样本。一般还会实现 `__len__()` 用于返回数据集的大小。

  > `DataLoader` 在默认情况下会创建一个生成整数索引的索引采样器 (sampler) 用于遍历数据集。因此，如果我们加载的是一个非整数索引的映射型数据集，还需要手工定义采样器。

- **迭代型 (Iterable-style) 数据集**

  继承自 `IterableDataset`，表示可迭代的数据集，它可以通过 `iter(dataset)` 以数据流 (steam) 的形式访问，适用于访问超大数据集或者远程服务器产生的数据。 迭代型数据集必须实现 `__iter__()` 函数，用于返回一个样本迭代器 (iterator)。

  > **注意：**如果在 `DataLoader`  中开启多进程（num_workers > 0），那么在加载迭代型数据集时必须进行专门的设置，否则会重复访问样本。例如：
  >
  > ```python
  > from torch.utils.data import IterableDataset, DataLoader
  > 
  > class MyIterableDataset(IterableDataset):
  > 
  >     def __init__(self, start, end):
  >         super(MyIterableDataset).__init__()
  >         assert end > start
  >         self.start = start
  >         self.end = end
  > 
  >     def __iter__(self):
  >         return iter(range(self.start, self.end))
  > 
  > ds = MyIterableDataset(start=3, end=7) # [3, 4, 5, 6]
  > # Single-process loading
  > print(list(DataLoader(ds, num_workers=0)))
  > # Directly doing multi-process loading
  > print(list(DataLoader(ds, num_workers=2)))
  > ```
  >
  > ```
  > [tensor([3]), tensor([4]), tensor([5]), tensor([6])]
  > 
  > [tensor([3]), tensor([3]), tensor([4]), tensor([4]), tensor([5]), tensor([5]), tensor([6]), tensor([6])]
  > ```
  >
  > 可以看到，当 DataLoader 采用 2 个进程时，由于每个进程都获取到了单独的数据集拷贝，因此会重复访问每一个样本。要避免这种情况，就需要在 DataLoader 中设置 `worker_init_fn` 来自定义每一个进程的数据集拷贝：
  >
  > ```python
  > from torch.utils.data import get_worker_info
  > 
  > def worker_init_fn(worker_id):
  >     worker_info = get_worker_info()
  >     dataset = worker_info.dataset  # the dataset copy in this worker process
  >     overall_start = dataset.start
  >     overall_end = dataset.end
  >     # configure the dataset to only process the split workload
  >     per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
  >     worker_id = worker_info.id
  >     dataset.start = overall_start + worker_id * per_worker
  >     dataset.end = min(dataset.start + per_worker, overall_end)
  > 
  > # Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
  > print(list(DataLoader(ds, num_workers=2, worker_init_fn=worker_init_fn)))
  > # With even more workers
  > print(list(DataLoader(ds, num_workers=20, worker_init_fn=worker_init_fn)))
  > ```
  >
  > ```
  > [tensor([3]), tensor([5]), tensor([4]), tensor([6])]
  > 
  > [tensor([3]), tensor([4]), tensor([5]), tensor([6])]
  > ```

下面我们以加载一个图像分类数据集为例，看看如何创建一个自定义的映射型数据集：

```python
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```

可以看到，我们实现了 `__init__()`、`__len__()` 和 `__getitem__()` 三个函数，其中：

- `__init__()` 初始化数据集参数，这里设置了图像的存储目录、标签（通过读取标签 csv 文件）以及样本和标签的数据转换函数；
- `__len__()` 返回数据集中样本的个数；
- `__getitem__()` 映射型数据集的核心，根据给定的索引 `idx` 返回样本。这里会根据索引从目录下通过 `read_image` 读取图片和从 csv 文件中读取图片标签，并且返回处理后的图像和标签。

### DataLoaders

前面的数据集 `Dataset` 类提供了一种按照索引访问样本的方式。不过在实际训练模型时，我们都需要先将数据集切分为很多的 mini-batches，然后按批 (batch) 将样本送入模型，并且循环这一过程，每一个完整遍历所有样本的循环称为一个 epoch。

> 训练模型时，我们通常会在每次 epoch 循环开始前随机打乱样本顺序以缓解过拟合。

Pytorch 提供了 `DataLoader` 类专门负责处理这些操作，除了基本的 `dataset`（数据集）和 `batch_size` （batch 大小）参数以外，还有以下常用参数：

- `shuffle`：是否打乱数据集；
- `sampler`：采样器，也就是一个索引上的迭代器；
- `collate_fn`：批处理函数，用于对采样出的一个 batch 中的样本进行处理（例如前面提过的 Padding 操作）。

例如，我们按照 `batch = 64` 遍历 Pytorch 自带的图像分类 FashionMNIST 数据集（每个样本是一张 $28\times 28$ 的灰度图，以及分类标签），并且打乱数据集：

```python
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

img = train_features[0].squeeze()
label = train_labels[0]
print(img.shape)
print(f"Label: {label}")
```

```
Feature batch shape: torch.Size([64, 1, 28, 28])
Labels batch shape: torch.Size([64])
torch.Size([28, 28])
Label: 8
```

**数据加载顺序和 `Sampler` 类**

对于迭代型数据集来说，数据的加载顺序直接由用户控制，用户可以精确地控制每一个 batch 中返回的样本，因此不需要使用 `Sampler` 类。

对于映射型数据集来说，由于索引可以不是整数，因此我们可以通过 `Sampler` 对象来设置加载时的索引序列，即设置一个索引上的迭代器。如果设置了 `shuffle` 参数，DataLoader 就会自动创建一个顺序或乱序的 sampler，我们也可以通过 `sampler` 参数传入一个自定义的 `Sampler` 对象。

常见的 `Sampler` 对象包括序列采样器 `SequentialSampler` 和随机采样器 `RandomSampler`，它们都通过传入待采样的数据集来创建：

```python
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler, RandomSampler
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_sampler = RandomSampler(training_data)
test_sampler = SequentialSampler(test_data)

train_dataloader = DataLoader(training_data, batch_size=64, sampler=train_sampler)
test_dataloader = DataLoader(test_data, batch_size=64, sampler=test_sampler)

train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
test_features, test_labels = next(iter(test_dataloader))
print(f"Feature batch shape: {test_features.size()}")
print(f"Labels batch shape: {test_labels.size()}")
```

```
Feature batch shape: torch.Size([64, 1, 28, 28])
Labels batch shape: torch.Size([64])
Feature batch shape: torch.Size([64, 1, 28, 28])
Labels batch shape: torch.Size([64])
```

**批处理函数 `collate_fn`**

批处理函数 `collate_fn` 负责对每一个采样出的 batch 中的样本进行处理。默认的 `collate_fn` 会进行如下操作：

- 添加一个新维度作为 batch 维；
- 自动地将 NumPy 数组和 Python 数值转换为 PyTorch 张量；
- 保留原始的数据结构，例如输入是字典的话，它会输出一个包含同样键 (key) 的字典，但是将值 (value) 替换为 batched 张量（如何可以转换的话）。

例如，如果样本是包含 3 通道的图像和一个整数型类别标签，即 `(image, class_index)`，那么默认的 `collate_fn` 会将这样的一个元组列表转换为一个包含 batched 图像张量和 batched 类别标签张量的元组。

我们也可以传入手工编写的 `collate_fn` 函数以对数据进行自定义处理，例如前面我们介绍过的 padding 操作。

## 3. 训练模型

Pytorch 所有的模块（层）都是 `nn.Module` 的子类，神经网络模型本身就是一个模块，它还包含了很多其他的模块。

### 构建模型

我们还是以前面加载的 FashionMNIST 数据库为例，构建一个神经网络模型来完成图像分类。模型同样继承自 `nn.Module` 类，通过 `__init__()` 初始化模型中的层和参数，在 `forward()` 中定义模型的操作，例如：

```python
import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.Dropout(p=0.2)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
```

```
Using cpu device
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=256, bias=True)
    (3): ReLU()
    (4): Linear(in_features=256, out_features=10, bias=True)
    (5): Dropout(p=0.2, inplace=False)
  )
)
```

可以看到，我们构建的模型首先将二维图像通过 `Flatten` 层压成一维向量，然后经过两个带有 `ReLU` 激活函数的全连接隐藏层，最后送入到一个包含 10 个神经元的分类器以完成 10 分类任务。我们还通过在最终输出前添加 Dropout 层来缓解过拟合。

最终我们构建的模型会输出一个 10 维向量（每一维对应一个类别的预测值），与先前介绍过的 pipeline 模型一样，这里输出的是 logits 值，我们需要再接一个 Softmax 层来计算最终的概率值。下面我们构建一个包含四个伪二维图像的 mini-batch 来进行预测：

```python
import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.Dropout(p=0.2)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)

X = torch.rand(4, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
print(pred_probab.size())
y_pred = pred_probab.argmax(-1)
print(f"Predicted class: {y_pred}")
```

```
Using cpu device
torch.Size([4, 10])
Predicted class: tensor([3, 8, 3, 3])
```

可以看到，模型成功输出了维度为 $(4, 10)$ 的预测结果（每个样本输出一个 10 维的概率向量）。最后我们通过 $\text{argmax}$ 操作，将输出的概率向量转换为对应的标签。

### 优化模型参数

在准备好数据、搭建好模型之后，我们就可以开始训练和测试（验证）模型了。正如前面所说，模型训练是一个迭代的过程，每一轮 epoch 迭代中模型都会对输入样本进行预测，然后对预测结果计算损失 (loss)，并求 loss 对每一个模型参数的偏导，最后使用优化器更新所有的模型参数。

> **损失函数 (Loss function)** 用于度量预测值与答案之间的差异，模型的训练过程就是最小化损失函数。Pytorch 实现了很多常见的[损失函数](https://pytorch.org/docs/stable/nn.html#loss-functions)，例如用于回归任务的均方误差 (Mean Square Error) [nn.MSELoss](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss)、用于分类任务的负对数似然 (Negative Log Likelihood)  [nn.NLLLoss](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss)、同时结合了 `nn.LogSoftmax` 和 `nn.NLLLoss` 的交叉熵损失 (Cross Entropy)  [nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss) 等。
>
> **优化器 (Optimization)** 使用特定的优化算法（例如随机梯度下降），通过在每一个训练阶段 (step) 减少（基于一个 batch 样本计算的）模型损失来调整模型参数。Pytorch 实现了很多[优化器](https://pytorch.org/docs/stable/optim.html)，例如 SGD、ADAM、RMSProp 等。

每一轮迭代 (Epoch) 实际上包含了两个步骤：

- **训练循环 (The Train Loop)** 在训练集上进行迭代，尝试收敛到最佳的参数；
- **验证/测试循环 (The Validation/Test Loop)** 在测试/验证集上进行迭代以检查模型性能有没有提升。

具体地，在训练循环中，优化器通过以下三个步骤进行优化：

- 调用 `optimizer.zero_grad()` 重设模型参数的梯度。默认情况下梯度会进行累加，为了防止重复计算，在每个训练阶段开始前都需要清零梯度；
- 通过 `loss.backwards()` 反向传播预测结果的损失，即计算损失对每一个参数的偏导；
- 调用 `optimizer.step()` 根据梯度调整模型的参数。

下面我们选择交叉熵作为损失函数、选择 AdamW 作为优化器，完整的训练循环和测试循环实现如下：

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

learning_rate = 1e-3
batch_size = 64
epochs = 3

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.Dropout(p=0.2)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader, start=1):
        X, y = X.to(device), y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(dim=-1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
```

```
Using cpu device
Epoch 1
-------------------------------
loss: 0.935758  [ 6400/60000]
loss: 0.991128  [12800/60000]
loss: 0.655021  [19200/60000]
loss: 0.938772  [25600/60000]
loss: 0.480326  [32000/60000]
loss: 0.526776  [38400/60000]
loss: 1.046211  [44800/60000]
loss: 0.749002  [51200/60000]
loss: 0.550378  [57600/60000]
Test Error: 
 Accuracy: 83.7%, Avg loss: 0.441249 

Epoch 2
-------------------------------
loss: 0.596351  [ 6400/60000]
loss: 0.614368  [12800/60000]
loss: 0.588207  [19200/60000]
loss: 0.698899  [25600/60000]
loss: 0.433412  [32000/60000]
loss: 0.533789  [38400/60000]
loss: 0.772370  [44800/60000]
loss: 0.486120  [51200/60000]
loss: 0.534202  [57600/60000]
Test Error: 
 Accuracy: 85.4%, Avg loss: 0.396990 

Epoch 3
-------------------------------
loss: 0.547906  [ 6400/60000]
loss: 0.591556  [12800/60000]
loss: 0.537591  [19200/60000]
loss: 0.722009  [25600/60000]
loss: 0.319590  [32000/60000]
loss: 0.504153  [38400/60000]
loss: 0.797246  [44800/60000]
loss: 0.553834  [51200/60000]
loss: 0.400079  [57600/60000]
Test Error: 
 Accuracy: 87.2%, Avg loss: 0.355058 

Done!
```

可以看到，通过 3 轮迭代 (Epoch)，模型在训练集上的损失逐步下降、在测试集上的准确率逐步上升，证明优化器成功地对模型参数进行了调整，而且没有出现过拟合。

> **注意：**一定要在预测之前调用 `model.eval()` 方法将 dropout 层和 batch normalization 层设置为评估模式，否则会产生不一致的预测结果。

## 4. 保存及加载模型

在之前的文章中，我们介绍过模型类 `Model` 的保存以及加载方法，但如果我们只是将预训练模型作为一个模块（例如作为编码器），那么最终的完整模型就是一个自定义 Pytorch 模型，它的保存和加载就必须使用 Pytorch 预设的接口。

### 保存和加载模型权重

Pytorch 模型会将所有参数存储在一个状态字典 (state dictionary) 中，可以通过 `Model.state_dict()` 加载。Pytorch 通过 `torch.save()` 保存模型权重：

```python
import torch
import torchvision.models as models

model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')
```

为了加载保存的权重，我们首先需要创建一个结构完全相同的模型实例，然后通过 `Model.load_state_dict()` 函数进行加载：

```python
model = models.vgg16() # we do not specify pretrained=True, i.e. do not load default weights
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
```

### 保存和加载完整模型

上面存储模型权重的方式虽然可以节省空间，但是加载前需要构建一个结构完全相同的模型实例来承接权重。如果我们希望在存储权重的同时，也一起保存模型结构，就需要将整个模型传给 `torch.save()` ：

```python
import torch
import torchvision.models as models

model = models.vgg16(pretrained=True)
torch.save(model, 'model.pth')
```

这样就可以直接从保存的文件中加载整个模型（包括权重和结构）：

```python
model = torch.load('model.pth')
```

## 代码

本章核心代码存储于：[How-to-use-Transformers/train_model_FashionMNIST.py](https://github.com/jsksxs360/How-to-use-Transformers/blob/main/train_model_FashionMNIST.py)

## 参考

[[1]](https://pytorch.org/docs/stable/) Pytorch 官方文档  
[[2]](https://pytorch.org/docs/stable/) Pytorch 在线教程  
[[3]](https://book.douban.com/subject/35531447/) 车万翔, 郭江, 崔一鸣. 《自然语言处理：基于预训练模型的方法》

