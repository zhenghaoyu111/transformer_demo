#!/usr/bin/env python3
"""
Transformer模型演示脚本
展示模型的基本功能和使用方法
"""

import torch
import torch.nn as nn
from transformer import Transformer, PositionalEncoding, MultiHeadAttention, FeedForward

def demo_basic_usage():
    """演示基本使用方法"""
    print("=" * 60)
    print("Transformer模型基本使用演示")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建一个小型Transformer模型
    model = Transformer(
        src_vocab_size=100,    # 源词汇表大小
        tgt_vocab_size=100,    # 目标词汇表大小
        d_model=128,           # 模型维度（较小以便演示）
        num_heads=4,           # 注意力头数
        num_encoder_layers=2,  # 编码器层数
        num_decoder_layers=2,  # 解码器层数
        d_ff=512,              # 前馈网络维度
        dropout=0.1            # Dropout率
    ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建示例输入
    batch_size = 2
    src_len = 8
    tgt_len = 6
    
    src = torch.randint(1, 100, (batch_size, src_len)).to(device)
    tgt = torch.randint(1, 100, (batch_size, tgt_len)).to(device)
    
    print(f"\n输入形状:")
    print(f"  源序列 (src): {src.shape}")
    print(f"  目标序列 (tgt): {tgt.shape}")
    print(f"  源序列内容: {src[0].cpu().numpy()}")
    print(f"  目标序列内容: {tgt[0].cpu().numpy()}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(src, tgt[:, :-1])  # 去掉最后一个token作为输入
        print(f"\n输出形状: {output.shape}")
        print(f"输出logits形状: {output.shape}")
        
        # 获取预测结果
        predictions = output.argmax(dim=-1)
        print(f"预测结果: {predictions[0].cpu().numpy()}")
        print(f"实际目标: {tgt[0, 1:].cpu().numpy()}")  # 去掉第一个token（通常是开始标记）


def demo_inference():
    """演示推理功能"""
    print("\n" + "=" * 60)
    print("Transformer模型推理演示")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = Transformer(
        src_vocab_size=100,
        tgt_vocab_size=100,
        d_model=128,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=512,
        dropout=0.1
    ).to(device)
    
    # 创建输入序列
    src = torch.randint(1, 100, (1, 5)).to(device)  # 单个序列
    print(f"输入序列: {src[0].cpu().numpy()}")
    
    # 推理
    model.eval()
    with torch.no_grad():
        output = model.inference(
            src, 
            max_length=10, 
            start_token=1, 
            end_token=2
        )
        print(f"生成序列: {output[0].cpu().numpy()}")


def demo_attention_visualization():
    """演示注意力权重（简化版）"""
    print("\n" + "=" * 60)
    print("注意力机制演示")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = Transformer(
        src_vocab_size=50,
        tgt_vocab_size=50,
        d_model=64,
        num_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        d_ff=256,
        dropout=0.0  # 关闭dropout以便观察注意力
    ).to(device)
    
    # 创建简单的输入
    src = torch.tensor([[1, 2, 3, 4, 5]]).to(device)
    tgt = torch.tensor([[1, 6, 7, 8]]).to(device)
    
    print(f"源序列: {src[0].cpu().numpy()}")
    print(f"目标序列: {tgt[0].cpu().numpy()}")
    
    # 前向传播并获取注意力权重
    model.eval()
    with torch.no_grad():
        # 编码
        src_embedded = model.src_embedding(src) * (model.d_model ** 0.5)
        src_encoded = model.positional_encoding(src_embedded)
        
        # 通过编码器层
        src_mask = model.generate_src_mask(src)
        for encoder_layer in model.encoder_layers:
            src_encoded = encoder_layer(src_encoded, src_mask)
        
        print(f"\n编码器输出形状: {src_encoded.shape}")
        print(f"编码器输出统计:")
        print(f"  均值: {src_encoded.mean().item():.4f}")
        print(f"  标准差: {src_encoded.std().item():.4f}")
        print(f"  最大值: {src_encoded.max().item():.4f}")
        print(f"  最小值: {src_encoded.min().item():.4f}")


def demo_model_components():
    """演示模型各个组件"""
    print("\n" + "=" * 60)
    print("模型组件演示")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 位置编码演示
    print("1. 位置编码演示")
    pos_encoding = PositionalEncoding(d_model=8, max_len=10)
    x = torch.randn(5, 8)  # 序列长度为5，维度为8
    encoded = pos_encoding(x)
    print(f"   原始输入形状: {x.shape}")
    print(f"   位置编码后形状: {encoded.shape}")
    print(f"   位置编码差异: {(encoded - x).abs().mean().item():.4f}")
    
    # 2. 多头注意力演示
    print("\n2. 多头注意力演示")
    attention = MultiHeadAttention(d_model=64, num_heads=4)
    query = torch.randn(2, 6, 64)  # batch_size=2, seq_len=6, d_model=64
    key = torch.randn(2, 6, 64)
    value = torch.randn(2, 6, 64)
    
    output, attention_weights = attention(query, key, value)
    print(f"   输入形状: {query.shape}")
    print(f"   输出形状: {output.shape}")
    print(f"   注意力权重形状: {attention_weights.shape}")
    print(f"   注意力权重和: {attention_weights.sum(dim=-1).mean().item():.4f} (应该接近1.0)")
    
    # 3. 前馈网络演示
    print("\n3. 前馈网络演示")
    ff = FeedForward(d_model=64, d_ff=256)
    x = torch.randn(2, 6, 64)
    output = ff(x)
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {output.shape}")


def main():
    """主函数"""
    print("Transformer模型完整演示")
    print("=" * 60)
    
    try:
        # 基本使用演示
        demo_basic_usage()
        
        # 推理演示
        demo_inference()
        
        # 注意力机制演示
        demo_attention_visualization()
        
        # 模型组件演示
        demo_model_components()
        
        print("\n" + "=" * 60)
        print("所有演示完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        print("请确保已安装所有依赖包: pip install -r requirements.txt")


if __name__ == "__main__":
    main() 
