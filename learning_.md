1.位置编码的计算

数学公式：
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

在使用python进行计算的时候 对公式进行转换    转换为 （postion * e^((-2i*log(10000))/d_model)） ==> div * position  div为公共部分  然后再分别对奇偶地址进行求解

2.求解注意力得分
  Q与K的转制点积再除去根号d_  在进行softmax后再乘 V  
  
3.将输入的 Q、K、V 向量分别投影到多个子空间（多个头）中，并行计算注意力，最后将各头的结果 拼接 + 再投影 合并成最终输出。

4.Decoder中的mask是两个mask合并形成的  padding mask 与 causal mask 合并之后得到mask用于decoder

5.Embedding 不是通过 one-hot 编码“降维”得到的，而是通过查表操作从一个可学习的向量矩阵中检索 token 的语义向量表示。
  数学上等价于 one-hot × W（W为embedding矩阵），但实现上完全跳过了 one-hot。

6，残差网络 避免梯度消失  --》dropout   将x1逐层传递防止梯度消失
   归一化（LayerNorm）  做标准化   限制区间  避免梯度爆炸

7.前馈神经网络（Feed Forward）Relu（wx+b）（Relu为激活函数）， 前面每一步都在做线性变换（线性变换就是空间中的移动 扩大和缩小），wx+b，线性变换的叠加永远都是线性变换，通过Feed Forward中的Relu做一次非线性变换，这样空间变换可以无限拟合任何一种状态

8.为什么使用多头注意力机制？1. 捕捉不同子空间的信息   2. 增强模型的表达能力  3. 提升稳定性和训练效率  4. 并行计算

9. 为什么 Transformer 需要做 KV 缓存？
在推理阶段（Inference），Transformer（尤其是 decoder，比如 GPT 类模型）是 一个词一个词地生成输出 的。每次生成时都会之前生成的所有 token 作为输入。

10.除根dk严谨点的解释: 假设qk独立，q正态分布，k正态分布，则qk的方差是维度dk，qk/根dk的方差是1。这样一来qk内积的更不容易很大或很小，其梯度也不容易会过大过小，以及发生softmax上下溢出。保证模型收敛稳定性。

11. 为什么使用layernorm BatchNorm（批归一化）归一化方式：对每个特征维度在 batch 中求均值和方差   LayerNorm（层归一化） 归一化方式：对单个样本内部所有特征做归一化  
