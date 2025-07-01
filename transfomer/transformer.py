import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """
    位置编码层 (Positional Encoding)
    
    作用：为序列中的每个位置添加位置信息，因为Transformer没有循环结构，需要显式的位置信息
    
    实现原理：
    1. 使用正弦和余弦函数生成位置编码
    2. 不同频率的正弦波可以表示不同的位置信息
    3. 偶数位置使用sin，奇数位置使用cos
    
    数学公式：
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        # 步骤1: 创建位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        
        # 步骤2: 创建位置索引 [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 步骤3: 计算频率项，用于生成不同频率的正弦波
        # div_term = 1 / (10000^(2i/d_model))，其中i是维度索引
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 步骤4: 应用正弦和余弦函数
        # 偶数维度使用sin函数
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数维度使用cos函数
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 步骤5: 调整维度顺序为 [1, max_len, d_model]，便于广播
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # 步骤6: 注册为buffer，不参与梯度更新，但会随模型保存
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：将位置编码加到输入上
        
        步骤：
        1. 获取对应长度的位置编码
        2. 将位置编码加到输入张量上（残差连接）
        """
        # 使用getattr访问buffer，避免类型检查问题
        pe: torch.Tensor = getattr(self, 'pe')[:x.size(0), :]
        return x + pe  # 残差连接：输入 + 位置编码


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制 (Multi-Head Attention)
    
    作用：允许模型同时关注输入序列的不同位置和不同表示子空间
    
    实现原理：
    1. 将输入投影到多个子空间（多头）
    2. 在每个子空间中计算注意力
    3. 将所有头的输出拼接并投影回原始维度
    
    核心思想：并行计算多个注意力机制，捕获不同类型的依赖关系
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0  # 确保d_model能被num_heads整除
        
        self.d_model = d_model      # 模型维度
        self.num_heads = num_heads  # 注意力头数
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # 步骤1: 定义线性变换层
        # 将输入投影到Query、Key、Value空间
        #Q类似一个查询 K是知识库的钥匙  Q和K越匹配即相似度越高 就能从这里找到更多与Q相关的信息
        self.w_q = nn.Linear(d_model, d_model)  # Query投影
        self.w_k = nn.Linear(d_model, d_model)  # Key投影
        self.w_v = nn.Linear(d_model, d_model)  # Value投影
        self.w_o = nn.Linear(d_model, d_model)  # 输出投影
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        缩放点积注意力 (Scaled Dot-Product Attention)
        
        作用：计算注意力权重并应用到值向量上
        
        实现步骤：
        1. 计算Q和K的点积
        2. 缩放点积结果（除以sqrt(d_k)）
        3. 应用mask（如果提供）
        4. 使用softmax计算注意力权重
        5. 应用dropout
        6. 将权重与V相乘得到输出
        """
        # 步骤1: 计算注意力分数 Q * K^T
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 步骤2: 应用mask（将masked位置设为负无穷）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 步骤3: 使用softmax计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        
        # 步骤4: 应用dropout防止过拟合
        attention_weights = self.dropout(attention_weights)
        
        # 步骤5: 将权重与值向量相乘得到输出
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        """
        多头注意力的前向传播
        
        实现步骤：
        1. 线性变换：将输入投影到Q、K、V
        2. 重塑维度：将结果分割成多个头
        3. 计算注意力：对每个头应用缩放点积注意力
        4. 拼接输出：将所有头的输出拼接
        5. 最终投影：将拼接结果投影回原始维度
        """
        batch_size = query.size(0)
        
        # 步骤1: 线性变换并重塑为多头格式
        # 原始形状: [batch_size, seq_len, d_model]
        # 目标形状: [batch_size, num_heads, seq_len, d_k]
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 步骤2: 计算注意力
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 步骤3: 拼接多头输出
        # 从 [batch_size, num_heads, seq_len, d_k] 重塑为 [batch_size, seq_len, d_model]
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # 步骤4: 最终线性变换
        output = self.w_o(attention_output)
        
        return output, attention_weights


class FeedForward(nn.Module):
    """
    前馈神经网络 (Feed-Forward Network)
    
    作用：为每个位置独立应用非线性变换，增加模型的表达能力
    
    实现原理：
    1. 两个线性变换层，中间有ReLU激活函数
    2. 第一个变换将维度从d_model扩展到d_ff
    3. 第二个变换将维度从d_ff压缩回d_model
    4. 使用dropout防止过拟合
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # 扩展维度
        self.linear2 = nn.Linear(d_ff, d_model)  # 压缩维度
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        前向传播：线性变换 -> ReLU -> Dropout -> 线性变换
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """
    Transformer编码器层
    
    作用：处理输入序列，提取特征表示
    
    结构：
    1. 多头自注意力机制
    2. 前馈神经网络
    3. 残差连接和层归一化
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)  # 第一个层归一化
        self.norm2 = nn.LayerNorm(d_model)  # 第二个层归一化
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        编码器层的前向传播
        
        实现步骤：
        1. 自注意力 + 残差连接 + 层归一化
        2. 前馈网络 + 残差连接 + 层归一化
        """
        # 步骤1: 自注意力 + 残差连接 + 层归一化
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))  # 残差连接 + 层归一化
        
        # 步骤2: 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))  # 残差连接 + 层归一化
        
        return x


class DecoderLayer(nn.Module):
    """
    Transformer解码器层
    
    作用：基于编码器输出生成目标序列
    
    结构：
    1. 多头自注意力机制（关注已生成的目标序列）
    2. 多头交叉注意力机制（关注编码器输出）
    3. 前馈神经网络
    4. 残差连接和层归一化
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)  # 自注意力后的层归一化
        self.norm2 = nn.LayerNorm(d_model)  # 交叉注意力后的层归一化
        self.norm3 = nn.LayerNorm(d_model)  # 前馈网络后的层归一化
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        解码器层的前向传播
        
        实现步骤：
        1. 自注意力（关注目标序列本身）
        2. 交叉注意力（关注编码器输出）
        3. 前馈网络
        每个步骤都包含残差连接和层归一化
        """
        # 步骤1: 自注意力 + 残差连接 + 层归一化
        # 自注意力让解码器关注已生成的目标序列
        attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 步骤2: 交叉注意力 + 残差连接 + 层归一化
        # 交叉注意力让解码器关注编码器的输出
        attn_output, _ = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # 步骤3: 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class Transformer(nn.Module):
    """
    完整的Transformer模型
    
    作用：实现序列到序列的转换，常用于机器翻译、文本生成等任务
    
    架构：
    1. 编码器：处理输入序列，提取特征
    2. 解码器：基于编码器输出生成目标序列
    3. 词嵌入和位置编码
    4. 输出层
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, 
                 max_seq_length=100, dropout=0.1):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        
        # 步骤1: 词嵌入层 - 将词汇ID转换为向量表示
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)  # 源语言嵌入
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)  # 目标语言嵌入
        
        # 步骤2: 位置编码 - 为序列添加位置信息
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # 步骤3: 编码器层 - 堆叠多个编码器层
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_encoder_layers)
        ])
        
        # 步骤4: 解码器层 - 堆叠多个解码器层
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_decoder_layers)
        ])
        
        # 步骤5: 输出层 - 将解码器输出映射到词汇表
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        
        # 步骤6: Dropout层 - 防止过拟合
        self.dropout = nn.Dropout(dropout)
        
        # 步骤7: 初始化参数
        self._init_parameters()
    
    def _init_parameters(self):
        """
        初始化模型参数
        
        作用：使用Xavier初始化确保梯度传播稳定
        """
        for p in self.parameters():
            if p.dim() > 1:  # 只初始化权重矩阵，不初始化偏置
                nn.init.xavier_uniform_(p)
    
    def generate_src_mask(self, src):
        """
        生成源序列的padding mask
        
        作用：在注意力计算时忽略padding token
        
        实现：创建布尔掩码，padding token（值为0）的位置为False
        """
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
        return src_mask
    
    def generate_tgt_mask(self, tgt):
        """
        生成目标序列的padding mask和因果mask
        
        作用：
        1. padding mask：忽略padding token
        2. causal mask：确保解码器只能看到当前位置之前的信息（防止信息泄露）
        """
        tgt_len = tgt.size(1)
        
        # 步骤1: 创建padding mask
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
        
        # 步骤2: 创建因果mask（下三角矩阵）
        # 确保位置i只能关注位置0到i的信息
        causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, dtype=torch.bool)).unsqueeze(0).unsqueeze(1)
        causal_mask = causal_mask.to(tgt.device)
        
        # 步骤3: 组合两种mask
        return tgt_mask & causal_mask
    
    def encode(self, src, src_mask):
        """
        编码器前向传播
        
        作用：将输入序列编码为特征表示
        
        实现步骤：
        1. 词嵌入 + 缩放
        2. 位置编码
        3. Dropout
        4. 通过编码器层
        """
        # 步骤1: 词嵌入并缩放
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        
        # 步骤2: 添加位置编码
        src = self.positional_encoding(src)
        
        # 步骤3: 应用dropout
        src = self.dropout(src)
        
        # 步骤4: 通过编码器层
        for encoder_layer in self.encoder_layers:
            src = encoder_layer(src, src_mask)
        
        return src
    
    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        """
        解码器前向传播
        
        作用：基于编码器输出生成目标序列
        
        实现步骤：
        1. 词嵌入 + 缩放
        2. 位置编码
        3. Dropout
        4. 通过解码器层
        """
        # 步骤1: 词嵌入并缩放
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        
        # 步骤2: 添加位置编码
        tgt = self.positional_encoding(tgt)
        
        # 步骤3: 应用dropout
        tgt = self.dropout(tgt)
        
        # 步骤4: 通过解码器层
        for decoder_layer in self.decoder_layers:
            tgt = decoder_layer(tgt, enc_output, src_mask, tgt_mask)
        
        return tgt
    
    def forward(self, src, tgt):
        """
        完整的前向传播
        
        作用：训练时的前向传播
        
        实现步骤：
        1. 生成mask
        2. 编码输入序列
        3. 解码生成输出序列
        4. 输出层映射
        """
        if tgt is None:
            return self.inference(src)
        
        # 步骤1: 生成mask
        src_mask = self.generate_src_mask(src)
        tgt_mask = self.generate_tgt_mask(tgt)
        
        # 步骤2: 编码
        enc_output = self.encode(src, src_mask)
        
        # 步骤3: 解码
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        
        # 步骤4: 输出层
        output = self.output_layer(dec_output)
        
        return output
    
    def inference(self, src, max_length=50, start_token=1, end_token=2):
        """
        推理模式（自回归生成）
        
        作用：在推理时逐个生成目标序列的token
        
        实现步骤：
        1. 编码输入序列
        2. 从start_token开始
        3. 逐个生成下一个token
        4. 直到生成end_token或达到最大长度
        """
        self.eval()
        with torch.no_grad():
            # 步骤1: 编码输入序列
            src_mask = self.generate_src_mask(src)
            enc_output = self.encode(src, src_mask)
            
            # 步骤2: 初始化目标序列
            batch_size = src.size(0)
            tgt = torch.full((batch_size, 1), start_token, dtype=torch.long, device=src.device)
            
            # 步骤3: 自回归生成
            for _ in range(max_length - 1):
                # 生成当前序列的mask
                tgt_mask = self.generate_tgt_mask(tgt)
                
                # 解码
                dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
                
                # 只取最后一个位置的输出进行预测
                output = self.output_layer(dec_output[:, -1:, :])
                
                # 选择概率最高的token
                next_token = output.argmax(dim=-1)
                
                # 将新token添加到序列中
                tgt = torch.cat([tgt, next_token], dim=1)
                
                # 检查是否所有序列都结束
                if (next_token == end_token).all():
                    break
            
            return tgt


def create_padding_mask(seq, pad_token=0):
    """
    创建padding mask的辅助函数
    
    作用：标识序列中哪些位置是padding，哪些是真实内容
    """
    return (seq != pad_token).unsqueeze(1).unsqueeze(2)


def create_causal_mask(size):
    """
    创建因果mask的辅助函数
    
    作用：确保位置i只能看到位置0到i的信息，防止信息泄露
    """
    mask = torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(1)
    return mask


# 示例使用
if __name__ == "__main__":
    # 模型参数
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 512
    num_heads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    d_ff = 2048
    dropout = 0.1
    
    # 创建模型
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff,
        dropout=dropout
    )
    
    # 示例输入
    batch_size = 2
    src_len = 10
    tgt_len = 8
    
    src = torch.randint(1, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_len))
    
    # 前向传播
    output = model(src, tgt)
    print(f"输入形状: src={src.shape}, tgt={tgt.shape}")
    print(f"输出形状: {output.shape}")
    
    # 推理
    inference_output = model.inference(src)
    print(f"推理输出形状: {inference_output.shape}")
    
    print("Transformer模型实现完成！") 
