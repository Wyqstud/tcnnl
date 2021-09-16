# test

# 说明
现在想把这个工作改投到期刊上，需要做一些改进和扩容的工作。改进和扩容的工作可以分成以下三个方向进行：

1）讨论加residual block的意义和重要性。

2）讨论采用之前那个训练方法的重要性。

3）讨论采用不同的reference的重要行。

4）讨论spatical-wise information 和 channel-wise information 结合的更优方法。

5）讨论TRA和SRA更优的结合方式。

6）讨论降低运算量的方法。

7）讨论层与层之间的`linkage`，用KL散度。


# exp1

虽然，当前的试验结果是最优的。但是，当去掉了residual block Rank1性能差不多下滑了1%。不知道是模型的波动还是什么其他的问题。
但是，在之前的实验中，证明了不用residual block 也是能达到比较好的性能的。可以对比一下，这两者的差别。


# exp 2

试图通过调整三层训练的权重来让这三层特征向量得到不同程度的训练。但是，发现修改之后并没有很好的效果。输出三层特征向量对应的全连接层的概率输出，发现第三层的结果普遍要高于前两层。
这只能说明，确实将三层特征取平均，能更好发挥模型特征互补的效果。感觉还是有些不太完美。
为什么，第三层的输出值这么小，还能做到这么好的互补效果呢？第三层找到的到底是什么样的特征？


# exp 3

因为发现之前，前三层的输出特征向量都非常的小，想着这个可能是没有将原来的特征向量加上去的原因。在本实验中，在原来 $\alpha \times F_i + \theta \times F_j$的基础上，将运算公式改成了 $\alpha \times F_i + \theta \times F_j + F_i + F_j$。相比原来，模型的性能在`rank1`上有了差不多 $0.2\%$ 的提升。

这至少说明这么改进应该是有作用的。可以测试一下，此时模型的输出值是一个什么情况。看一下值小的问题是不是确实别解决了，还是说值是没有变化的。

通过对比发现，模型对应的特征值输出确实有了变化，略微变大了一些。

# exp4

基于`exp3`我继续尝试了下，在`SRA`和`TRA`都采用了这种`shortcut`的`setting`。但是，效果不佳。说明过度的`shortcut`有可能干扰模型的学习。

# exp5

基于`exp3`，我们可以继续尝试一下采用不同`reference`的结果。



