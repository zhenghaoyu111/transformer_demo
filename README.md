# Transformer 模型实现

这是一个完整的Transformer模型实现，使用PyTorch框架构建。该实现包含了Transformer的所有核心组件，可以用于序列到序列的任务，如机器翻译、文本生成等。

## 功能特性

- ✅ **多头注意力机制** (Multi-Head Attention)
- ✅ **位置编码** (Positional Encoding)
- ✅ **编码器-解码器架构** (Encoder-Decoder)
- ✅ **残差连接** (Residual Connections)
- ✅ **层归一化** (Layer Normalization)
- ✅ **前馈神经网络** (Feed-Forward Networks)
- ✅ **训练和推理模式**
- ✅ **完整的训练脚本**

## 文件结构

```
.
├── transformer.py          # Transformer模型实现
├── train_transformer.py    # 训练脚本
├── requirements.txt        # 依赖包列表
├── README.md              # 说明文档
└── hello.py               # 原始文件
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 基本使用

```python
from transformer import Transformer

# 创建模型
model = Transformer(
    src_vocab_size=1000,    # 源词汇表大小
    tgt_vocab_size=1000,    # 目标词汇表大小
    d_model=512,            # 模型维度
    num_heads=8,            # 注意力头数
    num_encoder_layers=6,   # 编码器层数
    num_decoder_layers=6,   # 解码器层数
    d_ff=2048,              # 前馈网络维度
    dropout=0.1             # Dropout率
)

# 前向传播
src = torch.randint(1, 1000, (2, 10))  # 源序列
tgt = torch.randint(1, 1000, (2, 8))   # 目标序列
output = model(src, tgt)
print(f"输出形状: {output.shape}")

# 推理
inference_output = model.inference(src)
print(f"推理输出: {inference_output.shape}")
```

### 2. 训练模型

```bash
python train_transformer.py
```

训练脚本会：
- 创建一个简单的数据集进行演示
- 训练Transformer模型
- 保存训练好的模型
- 绘制训练损失曲线
- 测试推理功能

## 模型架构详解

### 1. 位置编码 (PositionalEncoding)

使用正弦和余弦函数生成位置编码，让模型能够理解序列中token的位置信息：

```python
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### 2. 多头注意力 (MultiHeadAttention)

将输入分成多个头，每个头独立计算注意力，然后拼接结果：

```python
MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### 3. 编码器层 (EncoderLayer)

每个编码器层包含：
- 多头自注意力机制
- 前馈神经网络
- 残差连接和层归一化

### 4. 解码器层 (DecoderLayer)

每个解码器层包含：
- 多头自注意力机制（带因果mask）
- 多头交叉注意力机制
- 前馈神经网络
- 残差连接和层归一化

## 核心组件

### PositionalEncoding
- 实现位置编码，让模型理解序列位置信息
- 使用正弦和余弦函数生成编码

### MultiHeadAttention
- 实现缩放点积注意力机制
- 支持多头并行计算
- 包含mask功能（padding mask和因果mask）

### FeedForward
- 两层前馈神经网络
- 使用ReLU激活函数
- 包含dropout正则化

### EncoderLayer
- 编码器单层实现
- 包含自注意力和前馈网络
- 残差连接和层归一化

### DecoderLayer
- 解码器单层实现
- 包含自注意力、交叉注意力和前馈网络
- 残差连接和层归一化

### Transformer
- 完整的Transformer模型
- 支持训练和推理模式
- 包含词嵌入、位置编码、编码器、解码器和输出层

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `src_vocab_size` | 源词汇表大小 | - |
| `tgt_vocab_size` | 目标词汇表大小 | - |
| `d_model` | 模型维度 | 512 |
| `num_heads` | 注意力头数 | 8 |
| `num_encoder_layers` | 编码器层数 | 6 |
| `num_decoder_layers` | 解码器层数 | 6 |
| `d_ff` | 前馈网络维度 | 2048 |
| `max_seq_length` | 最大序列长度 | 100 |
| `dropout` | Dropout率 | 0.1 |

## 使用示例

### 机器翻译任务

```python
# 假设我们有英文到中文的翻译数据
src_vocab_size = 5000  # 英文词汇表大小
tgt_vocab_size = 8000  # 中文词汇表大小

model = Transformer(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    d_model=512,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6
)

# 训练时
src = torch.tensor([[1, 2, 3, 4, 5]])  # 英文序列
tgt = torch.tensor([[1, 6, 7, 8, 9]])  # 中文序列
output = model(src, tgt)

# 推理时
translation = model.inference(src)
```

### 文本生成任务

```python
# 用于文本生成，只需要编码器部分
model = Transformer(
    src_vocab_size=vocab_size,
    tgt_vocab_size=vocab_size,
    d_model=512,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6
)

# 生成文本
input_text = torch.tensor([[1, 2, 3, 4]])
generated_text = model.inference(input_text, max_length=50)
```

## 注意事项

1. **词汇表大小**：需要根据实际任务调整源和目标词汇表大小
2. **序列长度**：注意设置合适的最大序列长度
3. **模型大小**：根据硬件资源调整模型参数
4. **训练数据**：需要准备合适的训练数据
5. **超参数调优**：根据具体任务调整学习率、batch size等超参数

## 扩展功能

可以基于此实现扩展以下功能：
- 学习率调度器
- 早停机制
- 模型检查点保存
- 注意力权重可视化
- 束搜索解码
- 多GPU训练支持

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [PyTorch官方文档](https://pytorch.org/docs/stable/index.html) 