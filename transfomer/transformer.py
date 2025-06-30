import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """位置编码层"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 使用getattr访问buffer，避免类型检查问题
        pe: torch.Tensor = getattr(self, 'pe')[:x.size(0), :]
        return x + pe


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """缩放点积注意力"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性变换
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 注意力计算
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 拼接多头输出
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # 最终线性变换
        output = self.w_o(attention_output)
        
        return output, attention_weights


class FeedForward(nn.Module):
    """前馈神经网络"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """Transformer编码器层"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 自注意力 + 残差连接 + 层归一化
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    """Transformer解码器层"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # 自注意力 + 残差连接 + 层归一化
        attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 交叉注意力 + 残差连接 + 层归一化
        attn_output, _ = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class Transformer(nn.Module):
    """完整的Transformer模型"""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, 
                 max_seq_length=100, dropout=0.1):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        
        # 词嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # 编码器和解码器层
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_encoder_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_decoder_layers)
        ])
        
        # 输出层
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 初始化参数
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_src_mask(self, src):
        """生成源序列的padding mask"""
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def generate_tgt_mask(self, tgt):
        """生成目标序列的padding mask和因果mask"""
        tgt_len = tgt.size(1)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        
        # 因果mask（下三角矩阵）
        causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, dtype=torch.bool)).unsqueeze(0).unsqueeze(1)
        causal_mask = causal_mask.to(tgt.device)
        
        return tgt_mask & causal_mask
    
    def encode(self, src, src_mask):
        """编码器前向传播"""
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        src = self.positional_encoding(src)
        src = self.dropout(src)
        
        for encoder_layer in self.encoder_layers:
            src = encoder_layer(src, src_mask)
        
        return src
    
    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        """解码器前向传播"""
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.positional_encoding(tgt)
        tgt = self.dropout(tgt)
        
        for decoder_layer in self.decoder_layers:
            tgt = decoder_layer(tgt, enc_output, src_mask, tgt_mask)
        
        return tgt
    
    def forward(self, src, tgt):
        """完整的前向传播"""
        if tgt is None:
            return self.inference(src)
        
        src_mask = self.generate_src_mask(src)
        tgt_mask = self.generate_tgt_mask(tgt)
        
        # 编码
        enc_output = self.encode(src, src_mask)
        
        # 解码
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        
        # 输出层
        output = self.output_layer(dec_output)
        
        return output
    
    def inference(self, src, max_length=50, start_token=1, end_token=2):
        """推理模式"""
        self.eval()
        with torch.no_grad():
            src_mask = self.generate_src_mask(src)
            enc_output = self.encode(src, src_mask)
            
            batch_size = src.size(0)
            tgt = torch.full((batch_size, 1), start_token, dtype=torch.long, device=src.device)
            
            for _ in range(max_length - 1):
                tgt_mask = self.generate_tgt_mask(tgt)
                dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
                output = self.output_layer(dec_output[:, -1:, :])
                
                next_token = output.argmax(dim=-1)
                tgt = torch.cat([tgt, next_token], dim=1)
                
                # 检查是否所有序列都结束
                if (next_token == end_token).all():
                    break
            
            return tgt


def create_padding_mask(seq, pad_token=0):
    """创建padding mask"""
    return (seq != pad_token).unsqueeze(1).unsqueeze(2)


def create_causal_mask(size):
    """创建因果mask"""
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
