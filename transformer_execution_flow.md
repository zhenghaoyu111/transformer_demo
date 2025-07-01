# Transformer 代码执行顺序图

## 1. 训练模式执行流程

```mermaid
graph TD
    A[输入: src, tgt] --> B[forward方法]
    B --> C[生成mask]
    C --> C1[generate_src_mask]
    C --> C2[generate_tgt_mask]
    C1 --> D[编码阶段]
    C2 --> D
    
    D --> D1[encode方法]
    D1 --> D2[词嵌入 + 缩放]
    D2 --> D3[位置编码]
    D3 --> D4[Dropout]
    D4 --> D5[编码器层循环]
    
    D5 --> D6[EncoderLayer]
    D6 --> D7[MultiHeadAttention]
    D7 --> D8[scaled_dot_product_attention]
    D8 --> D9[FeedForward]
    D9 --> D10[残差连接 + 层归一化]
    D10 --> D11{还有编码器层?}
    D11 -->|是| D6
    D11 -->|否| E[解码阶段]
    
    E --> E1[decode方法]
    E1 --> E2[词嵌入 + 缩放]
    E2 --> E3[位置编码]
    E3 --> E4[Dropout]
    E4 --> E5[解码器层循环]
    
    E5 --> E6[DecoderLayer]
    E6 --> E7[自注意力]
    E7 --> E8[交叉注意力]
    E8 --> E9[FeedForward]
    E9 --> E10[残差连接 + 层归一化]
    E10 --> E11{还有解码器层?}
    E11 -->|是| E6
    E11 -->|否| F[输出层]
    
    F --> F1[output_layer]
    F1 --> G[返回logits]
```

## 2. 推理模式执行流程

```mermaid
graph TD
    A[输入: src] --> B[inference方法]
    B --> C[设置eval模式]
    C --> D[torch.no_grad]
    D --> E[编码阶段]
    
    E --> E1[generate_src_mask]
    E1 --> E2[encode方法]
    E2 --> E3[词嵌入 + 缩放]
    E3 --> E4[位置编码]
    E4 --> E5[Dropout]
    E5 --> E6[编码器层循环]
    E6 --> F[初始化目标序列]
    
    F --> F1[创建start_token序列]
    F1 --> G[自回归生成循环]
    
    G --> G1[generate_tgt_mask]
    G1 --> G2[decode方法]
    G2 --> G3[词嵌入 + 缩放]
    G3 --> G4[位置编码]
    G4 --> G5[Dropout]
    G5 --> G6[解码器层循环]
    G6 --> G7[output_layer]
    G7 --> G8[argmax选择token]
    G8 --> G9[添加到序列]
    G9 --> G10{检查结束条件}
    G10 -->|继续| G1
    G10 -->|结束| H[返回生成序列]
```

## 3. 详细组件执行顺序

### 3.1 MultiHeadAttention 执行流程

```mermaid
graph TD
    A[输入: query, key, value] --> B[线性变换]
    B --> B1[w_q: query投影]
    B --> B2[w_k: key投影]
    B --> B3[w_v: value投影]
    
    B1 --> C[重塑维度]
    B2 --> C
    B3 --> C
    C --> D[scaled_dot_product_attention]
    
    D --> D1[计算注意力分数]
    D1 --> D2[应用mask]
    D2 --> D3[softmax计算权重]
    D3 --> D4[应用dropout]
    D4 --> D5[与value相乘]
    D5 --> E[拼接多头输出]
    
    E --> F[w_o: 输出投影]
    F --> G[返回attention_output, attention_weights]
```

### 3.2 EncoderLayer 执行流程

```mermaid
graph TD
    A[输入: x, mask] --> B[自注意力]
    B --> B1[MultiHeadAttention]
    B1 --> B2[残差连接]
    B2 --> B3[层归一化]
    B3 --> C[前馈网络]
    
    C --> C1[FeedForward]
    C1 --> C2[残差连接]
    C2 --> C3[层归一化]
    C3 --> D[返回输出]
```

### 3.3 DecoderLayer 执行流程

```mermaid
graph TD
    A[输入: x, enc_output, src_mask, tgt_mask] --> B[自注意力]
    B --> B1[MultiHeadAttention with tgt_mask]
    B1 --> B2[残差连接]
    B2 --> B3[层归一化]
    B3 --> C[交叉注意力]
    
    C --> C1[MultiHeadAttention with src_mask]
    C1 --> C2[残差连接]
    C2 --> C3[层归一化]
    C3 --> D[前馈网络]
    
    D --> D1[FeedForward]
    D1 --> D2[残差连接]
    D2 --> D3[层归一化]
    D3 --> E[返回输出]
```

## 4. 关键方法调用链

### 4.1 训练时完整调用链

```
model.forward(src, tgt)
├── generate_src_mask(src)
├── generate_tgt_mask(tgt)
├── encode(src, src_mask)
│   ├── src_embedding(src) * sqrt(d_model)
│   ├── positional_encoding(src)
│   ├── dropout(src)
│   └── encoder_layers循环
│       └── EncoderLayer.forward(x, mask)
│           ├── MultiHeadAttention.forward(x, x, x, mask)
│           │   ├── scaled_dot_product_attention(Q, K, V, mask)
│           │   └── w_o(attention_output)
│           ├── FeedForward.forward(x)
│           └── LayerNorm + 残差连接
├── decode(tgt, enc_output, src_mask, tgt_mask)
│   ├── tgt_embedding(tgt) * sqrt(d_model)
│   ├── positional_encoding(tgt)
│   ├── dropout(tgt)
│   └── decoder_layers循环
│       └── DecoderLayer.forward(x, enc_output, src_mask, tgt_mask)
│           ├── 自注意力 (MultiHeadAttention)
│           ├── 交叉注意力 (MultiHeadAttention)
│           └── FeedForward
└── output_layer(dec_output)
```

### 4.2 推理时完整调用链

```
model.inference(src)
├── generate_src_mask(src)
├── encode(src, src_mask)  # 同训练时
├── 初始化 tgt = [start_token]
└── 自回归循环:
    ├── generate_tgt_mask(tgt)
    ├── decode(tgt, enc_output, src_mask, tgt_mask)  # 同训练时
    ├── output_layer(dec_output[:, -1:, :])
    ├── argmax选择next_token
    ├── torch.cat([tgt, next_token], dim=1)
    └── 检查结束条件
```

## 5. 数据流转换

### 5.1 张量形状变化

```
输入: src [batch_size, src_len], tgt [batch_size, tgt_len]

词嵌入后: [batch_size, seq_len, d_model]
位置编码后: [batch_size, seq_len, d_model] (形状不变，内容变化)

多头注意力内部:
- 线性变换: [batch_size, seq_len, d_model]
- 重塑: [batch_size, num_heads, seq_len, d_k]
- 注意力计算: [batch_size, num_heads, seq_len, seq_len]
- 拼接: [batch_size, seq_len, d_model]

输出: [batch_size, tgt_len, tgt_vocab_size]
```

### 5.2 关键计算步骤

1. **注意力分数计算**: `Q * K^T / sqrt(d_k)`
2. **注意力权重**: `softmax(scores)`
3. **注意力输出**: `attention_weights * V`
4. **残差连接**: `x + sublayer(x)`
5. **层归一化**: `LayerNorm(x + sublayer(x))`

这个执行顺序图展示了Transformer从输入到输出的完整数据流，包括训练和推理两种模式下的详细执行路径。 