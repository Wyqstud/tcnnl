# 实验结果

|epoch|mAP|rank1|rank5|
|:--:|:--:|:--:|:--:|
|520|$\color{red}81.0$|86.5|94.6|
|480|$\color{red}81.0$|86.5|$\color{red}94.8$|
|440|80.8|86.5|94.7|
|400|80.7|86.5|$\color{red}94.8$|
|360|80.7|86.4|94.6|
|320|80.5|$\color{red}86.7$|94.7|

# 实验说明

在本实验中，没有把第`0`维向量参与特征向量的计算，并且保持了特征融合过后的非线性激活函数`ReLU`。即 下列代码中的`relu`保留。
```
if seq_len != 1:
	feature = self.cat_vect(gap_feat_vect)
	feature = self.relu(feature)
```
并且把原来权向量的计算办法改动了。原来是，
$$
para_0 = ReLU(W_1 \times Sigmoid(W_2 \times f_0)) \tag{1}
$$
$$
para_1 = ReLU(W_1 \times Sigmoid(W_2 \times f_1)  \tag{2}
$$
$$
(\alpha \times f_0 + \gamma \times f_1)^2 = (para_1 \times f_0 + para_0 \times f_1)^2 \tag{3}
$$
现在把公式(1)改成了，
$$
(\alpha \times f_0 + \gamma \times f_1)^2 = (para_0 \times f_0 +para_1 \times f_1)^2 \tag{4}
$$
修改的原因是，这里全连接层的参数$W_1$和$W_2$的参数都是共享的。并且权重特征向量是经过`view(b*t, 1)`过后得到的，将无关系的`batch`和存在关系的`seq_len`放在同一地位，给全连接层训练。逻辑上，有点说不通。所以，事实上我希望公式3表达的思想不能实现。即，是否按照顺序不是很重要了。

# 实验分析

可能不合理的地方：
1、BN层和卷积层放在一起用的时候，一般是设置偏置`bias = 0`的。但是，在我设计的`block`里面的还保留有偏置的计算。
2、想要实现你所想的思路，要用两个不同的全连接层，或者用`for`循环一批批处理数据。不能用`view(b*t, -1)`的形式来做。要保证输入`FC`中的数据之间是相互独立的。

## 结论

把第`0`阶特征向量剔除在特征表示之外，应该是一个合理的做法。

## tricks

如果以后要还碰到这种，可能因为`ReLU`导致输出为零的情况，可以把`ReLU`放在卷积之前。能达到类似的效果。

# 修改方向

1、通过批量处理，让网络实现你希望的那样。即公式(1)(2)(3)。别用`view`重构数据
2、在实现之前想的分$\rightarrow$合$\rightarrow$结构