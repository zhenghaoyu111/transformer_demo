import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformer import Transformer

class SimpleDataset(Dataset):
    """简单的数据集用于演示"""
    def __init__(self, num_samples=1000, seq_length=10, vocab_size=100):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        
        # 生成随机序列对
        self.src_data = torch.randint(1, vocab_size, (num_samples, seq_length))
        self.tgt_data = torch.randint(1, vocab_size, (num_samples, seq_length))
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]


def train_transformer():
    """训练Transformer模型"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 模型参数
    src_vocab_size = 100
    tgt_vocab_size = 100
    d_model = 256
    num_heads = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    d_ff = 1024
    dropout = 0.1
    
    # 训练参数
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.0001
    
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
    ).to(device)
    
    # 创建数据集和数据加载器
    train_dataset = SimpleDataset(num_samples=1000, seq_length=10, vocab_size=src_vocab_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding token
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练循环
    model.train()
    losses = []
    
    print("开始训练...")
    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (src, tgt) in enumerate(progress_bar):
            src, tgt = src.to(device), tgt.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            output = model(src, tgt[:, :-1])  # 去掉最后一个token作为输入
            
            # 计算损失
            loss = criterion(output.reshape(-1, tgt_vocab_size), tgt[:, 1:].reshape(-1))
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.show()
    
    # 保存模型
    torch.save(model.state_dict(), 'transformer_model.pth')
    print("模型已保存到 transformer_model.pth")
    
    # 测试推理
    print("\n测试推理...")
    model.eval()
    with torch.no_grad():
        test_src = torch.randint(1, src_vocab_size, (1, 10)).to(device)
        print(f"输入序列: {test_src[0].cpu().numpy()}")
        
        output = model.inference(test_src, max_length=15)
        print(f"输出序列: {output[0].cpu().numpy()}")
    
    return model


def load_and_test_model():
    """加载并测试已训练的模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模型参数（需要与训练时保持一致）
    src_vocab_size = 100
    tgt_vocab_size = 100
    d_model = 256
    num_heads = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    d_ff = 1024
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
    ).to(device)
    
    # 加载模型权重
    try:
        model.load_state_dict(torch.load('transformer_model.pth', map_location=device))
        print("模型加载成功！")
        
        # 测试推理
        model.eval()
        with torch.no_grad():
            test_src = torch.randint(1, src_vocab_size, (1, 10)).to(device)
            print(f"输入序列: {test_src[0].cpu().numpy()}")
            
            output = model.inference(test_src, max_length=15)
            print(f"输出序列: {output[0].cpu().numpy()}")
            
    except FileNotFoundError:
        print("未找到模型文件，请先运行训练")


if __name__ == "__main__":
    print("Transformer训练脚本")
    print("=" * 50)
    
    # 训练模型
    model = train_transformer()
    
    print("\n" + "=" * 50)
    print("测试加载模型")
    load_and_test_model() 
