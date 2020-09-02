# 实验说明

## 修改说明

因为之前查明了，是过度使用非线性函数`ReLU`导致网络中出现大量的零向量，进而导致网络训练效果非常差。所以，现在把网络中所有非必要的`ReLU`函数均去掉了。主要是那些，使用在`一维卷积`$CatConv$后的去掉了。

## 实验结果

|epoch|mAP|Rank1|Rank5|
|:--:|:--:|:--:|:--:|
|520|80.2%|86.1%|94.9%|
|480|80.2%|86.0%|95.1%|
|440|80.2%|86.3%|94.9%|
|400|80.2%|86.2%|95.2%|

## 结果对比

这是在目前最佳实验的基础上做的修改。主要是，删除了第`0`阶特征向量中用到的`ReLU`。但是，发现此时性能出现了大约`1%`的下滑。最起码说明，第`0`阶特征向量参与特征表示和反向传播对模型起到反作用的。想办法做一些改进。