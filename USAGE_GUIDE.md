# LLaMA.cpp 使用指南

## 1. 下载模型

### 国内镜像源下载
```bash
# 创建模型目录
mkdir -p models

# 方法1: 使用 wget 下载 (需要科学上网或使用代理)
wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4_k_m.gguf -O models/phi-3-mini.gguf

# 方法2: 手动下载
# 访问 https://modelscope.cn/models 搜索 "GGUF"
# 或访问 https://huggingface.co/TheBloke 下载模型
```

## 2. 基本使用

### 文本生成
```bash
./build/bin/llama-cli -m models/phi-3-mini.gguf -p "Hello, world!" -n 100
```

### 聊天模式
```bash
./build/bin/llama-cli -m models/phi-3-mini.gguf -i
```

### 启动服务器
```bash
./build/bin/llama-server -m models/phi-3-mini.gguf --host 0.0.0.0 --port 8080
```

## 3. 常用参数

- `-m`: 模型文件路径
- `-p`: 输入提示
- `-n`: 生成token数
- `-ngl`: GPU层数
- `--temp`: 温度 (0.0-1.0)
- `--top-p`: 核采样
- `-i`: 交互模式

## 4. 推荐模型

- **Phi-3 Mini**: 1.8GB，入门推荐
- **Qwen2.5-1.5B**: 1.5GB，中文支持好
- **Llama-2-7B-Chat**: 4GB，通用性强

## 5. 性能优化

```bash
# 使用GPU加速
./build/bin/llama-cli -m model.gguf -ngl 35 -p "Hello"

# 查看可用设备
./build/bin/llama-cli --list-devices
```

现在可以开始使用了！ 