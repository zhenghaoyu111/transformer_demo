# llama.cpp

  #llama

这是一个基于 GGML 的高性能本地大语言模型推理库。

  

## 项目简介

  

llama.cpp 是一个用 C/C++ 编写的轻量级、高性能的大语言模型推理库。它支持多种模型格式，包括 GGUF、GGML 等，并且可以在 CPU 和多种 GPU 平台上高效运行。

  
  
  

**llam论文中的一些：**

scaling law：尺寸定律 尺度律

尺度律揭示了：只要模型规模、数据量和计算资源成比例增长，大语言模型的性能就可以持续提高。

  

目前因为数据量以及资源的影响 --》 强化学习 自己造数据自己学习 --》openai o1模型

  
  

**llama结构中的一些理解：**

1.大语言模型（LLM）的使用流程，一般分成两个阶段：提示（Prompt） 阶段 和 生成（Generation）阶段。

2.llama与transfomer比较 为什么要增加tokeizer？

因为 Transformer 不能直接理解文字，只能处理数字向量。Tokenizer 是将文字变成模型可读数字序列的“翻译器”。

3.Tokenizer后面使用embedding的作用是

Tokenizer 把文本变成离散的 token ID，Embedding 层则把这些 ID 转换成模型能处理的连续向量，是连接语言和神经网络之间的“桥梁”。

4.transfomer block

输入向量 (来自 embedding 或前一层输出)

│

┌───────▼───────┐

│ 多头注意力（Self-Attention）

└───────▲───────┘

│（加残差 + 归一化）

▼

┌────────────────┐

│ 前馈神经网络（Feed Forward）

└────────────────┘

│（加残差 + 归一化）

▼

输出向量 → 下一层

5.LayerNorm与RMSNorm的区别：

RMSNorm 是 LayerNorm 的简化版，不进行中心化（不减均值），只使用均方根归一化，因此更快、更轻量，性能相当甚至更稳定。

6.为什么llama选择使用RMSNorm？

LLaMA 使用 RMSNorm，是为了更快、更省、更稳地训练大模型，同时保持与 LayerNorm 类似甚至更好的效果。

7.为什么 LLaMA 选择 embedding 输出后就立即用 RMSNorm？

1️⃣简化模型初始化，防止前期激活值过大

2️⃣ RMSNorm 更轻、更快、更稳定，适合 embedding 层

3️⃣ Meta 试验验证：Embedding + RMSNorm 效果更好

8.KVCache KVCache 是在生成文本时，缓存已经计算好的 K（Key）和 V（Value）向量，避免重复计算，提高推理速度和效率。

9.RoPE 为什么使用RoPE？

RoPE（旋转位置编码） 是一种先进的位置编码方法，能让模型更自然、更平滑地理解“顺序”和“相对位置信息”，尤其适合处理长上下文，所以被广泛用于现代大语言模型。

10.ROPE与position encoding的区别以及为什么在llama中使用RoPE？

区别：RoPE（旋转位置编码）是一种相对位置编码方式，它通过对 Q/K 向量进行旋转而非加法注入位置信息，具备无参数、计算高效、支持长文本外推等优点，相比传统的绝对位置编码（如 sin/cos），在大模型中表现更好、更灵活。

原因：LLaMA 使用 RoPE 是因为它能更高效地建模相对位置信息，支持超长上下文、推理更快、更稳定，并能简化模型结构，是大语言模型架构中的主流选择。

11.ROPE的劣势

缺乏显式的位置感知 复杂性更高 可能不适合固定上下文建模 高频成分衰减

12.关于位置编码的一些idea

1️⃣self-attention是不关注位置信息的或者说是对位置信息不敏感的，例如一句话“早上好”和“早好上”经过没有位置信息的self-attention计算之后每一个汉字对应的向量是不变的但是这两句话因为调换了位置信息所以实际上的语意是不同的 但是每一个汉字对应的向量没变，表示位置信息改变没有产生影响，也就是self- attention本身是对位置不敏感的。因此要在计算时或者计算之前加上位置信息。

2️⃣RoPE编码是在计算QK的时候加上的相对位置编码，需要在每一次self——attention都要加，而position-encoding是在embedding之后增加的绝对位置编码，llma源码中将旋转自编码写在attention中每一层都有一个旋转自编码.与GLM的区别就是llama每一层都有旋转位置编码而GLM只在第一层有。

3️⃣RoPE通过欧拉公式引入复数，使用复数增加一个相对位置关系 （通过欧拉公式把相对位置信息“编码”为向量的旋转） 详细公式讲解：https://www.bilibili.com/video/BV1Tr421p7By?spm_id_from=333.788.player.switch&vd_source=5294196a6fd908a2e92087e1b853b70f

4️⃣如何将二维的旋转位置编码扩展到多维？将向量中每两个元素进行旋转位置

5️⃣代码实现：先求角度theta以及每一个字的顺序，然后将theta和顺序融合==》或者旋转角度

神经元是无序的，不依赖维度顺序 所以只需要任意取出来一半元素进行旋转操作

6️⃣高频处理短距离依赖 低频处理长距离依赖 频率越低对相对位置不敏感

  

**关于llma2.c的个人理解**

  

1. llma2.c的组成部分以及输入输出：

主要由tokenizer transformer sampler三部分组成

1️⃣Tokenizer负责将输入文本渲染的提示转换为一系列称为token的数字序列，每个数字都是指向词汇表中的数字索引。词汇表存储了模型支持的语言所使用的所有字符和子词——Llama2 英文模型有 32,000 个独特的token。

2️⃣Transformer 负责将token通过训练好的权重运行，权重存储了模型所学习到的知识。每个token逐个通过模型单独运行。对于每个处理的token，Transformer 输出一个称为 logits 的数字向量。logits 向量中的每个条目代表词汇表中的 32,000 个token槽位之一。注意最后的logits维度是 32000 x 1

3️⃣Sampler负责使用贪婪 arg-max 函数或基于概率的采样方法将 logits 转换回token。处理完输入的每个token后，采样器输出的token随后被反馈到 Transformer 中，作为生成下一个token的输入。这个过程一直持续到达到结束token。在每一步中，采样器输出的token也被解码回其文本表示，并输出给用户。

  

输入：system prompt 和 user prompt 之后将两个输入以及历史对话等内容进行拼接 将拼接后的内容输入到tokenizer中去

输出：采样器输出的token也被解码回其文本表示，并输出给用户。

  
  

2. Tokenizer Encoder

Tokenizer分词器使用在初始化过程中加载的词汇表以及字节对编码（BPE）算法，将Rendered Prompt中的每个字符编码成一系列输入token

  

BPE算法 BPE（Byte Pair Encoding） 是一种基于统计合并频率的分词算法，它从单个字符开始，逐步合并频繁出现的字符对，生成更大的子词单位。

  

流程：

在初始化过程中，分词器从 bin 文件中加载字节对编码词汇表。在处理提示时，rendered prompt被发送到分词器。在内部，分词器首先按字符串条目对词汇表进行排序，仅进行一次，以加快token查找速度。输入的每个字符首先被转换为对应的字符token，然后使用 BPE 算法进行合并，直到所有成对字符都被计算在内。 （char -> 单词的合并）返回的“压缩”后的token集作为输入token集。

  

3. transfomer

Transformer 的主要目标是逐个处理每个token（从起始token开始），并生成与每个token对应的 logits，其中token的 logits 是一个 32,000 个数字的向量，对应于词汇表中的每个槽位。

  

1️⃣Pre-normalization RSMNorm

Pre-Norm：在残差连接之前对输入做归一化（通常是 LayerNorm）。优点是梯度更加稳定，在深层网络里能缓解梯度消失/爆炸问题。Post-Norm：在残差连接之后做归一化。

  

为什么 RMSNorm 更好？

✅ 更快：少了均值计算（和中心化操作），速度更快。

✅ 更简单：只归一化幅值，不改变输入向量的方向（方向信息在某些任务里很重要）。

✅ 更适合深层 Transformer：实践证明它能提升训练稳定性（尤其是搭配 Pre-Norm 结构时）。

  

与layernorm的区别：

✅ LayerNorm：先减去均值再除以标准差，既归一化了方向又归一化了幅值，有两个参数（\gamma, \beta）。

✅ RMSNorm：只除以均方根（RMS），不减均值，只归一化幅值，参数更少（只有\gamma），计算更快。

RMSNorm 是简化版的 LayerNorm，省略了中心化（不减去均值）和方差归一化，速度更快，方向信息更完整，适合大模型。

  

2️⃣SwiGLU

为什么用SwiGLU？

1️更强的非线性表示能力 2️更好的梯度流动 3️ 性能提升

  

3️⃣cacahe KV 用于推理加速

核心思想：已经算过的 Key 和 Value 缓存下来，下一步生成时 直接复用，只算新的 Query 和 Key/Value。

原理：在每一层多头注意力中，新token的Query要和所有之前的 K 计算 把历史 Key 、Value 缓存起来

✅ 新 token 只需要：

• 计算自己的 K_t, V_t

• 把 K_t, V_t 拼到缓存里

• 与缓存的 K_{1:t}、V_{1:t} 做 Attention Q_t 点积 [K_{1:t}]^\topßß

4️⃣auto-regressive transfomers

在 Transformer 结构里，自回归模型就是把 标准 Transformer 编码器-解码器结构简化成 只用解码器（Decoder），每次输出一个 token，并把这个 token 作为下一步输入，直到生成结束。

5️⃣基于人类反馈的强化学习（RLHF）

6️⃣对齐（Alignment）

7️⃣GQA的使用(llama1与llama2的主要区别) 提升attention计算的效率

核心思想：Key 和 Value 不需要跟 Query 保持一一对应。


#**代码部分
tokenizer代码
```
class LLaMATokenizer:
    def __init__(self, model_path: str):
        """
        初始化 LLaMA Tokenizer
        :param model_path: SentencePiece 模型文件路径，例如 llama.model
        """
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> list[int]:
        """
        将文本编码为 token ids
        :param text: 要编码的文本
        :param add_bos: 是否在开头添加 BOS (beginning-of-sequence) token
        :param add_eos: 是否在结尾添加 EOS (end-of-sequence) token
        :return: token id 列表
        """
        ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.sp.bos_id()] + ids
        if add_eos:
            ids = ids + [self.sp.eos_id()]
        return ids

    def decode(self, token_ids: list[int]) -> str:
        #将 token ids 解码回文本
        :param token_ids: token id 列表
        :return: 解码后的字符串
        return self.sp.decode(token_ids)

    def bos_id(self) -> int:
        """获取 BOS token id"""
        return self.sp.bos_id()

    def eos_id(self) -> int:
        """获取 EOS token id"""
        return self.sp.eos_id()

    def pad_id(self) -> int:
        """获取 PAD token id"""
        return self.sp.pad_id()
```

RMSNorm部分
```
class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):

        """
        LLaMA2 的 RMSNorm 实现
        :param dim: 输入维度
        :param eps: 防止除零的小常数

        """

        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 可训练缩放参数 γ

  

    def forward(self, x):

        # 计算均方根

        rms = x.norm(dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)
        norm_x = x / (rms + self.eps)  # 归一化
        return self.weight * norm_x    # 乘以缩放参数
```

transfomer部分代码
```
class RMSNorm(nn.Module):

    """LLaMA 使用的 RMSNorm"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.norm(dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)
        return self.weight * x / (rms + self.eps)

class SwiGLU(nn.Module):

    """前馈网络激活函数"""

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2

  

  

class FeedForward(nn.Module):

    """前馈网络（FFN）"""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.activation = SwiGLU()
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class MultiHeadAttention(nn.Module):

    """多头自注意力"""
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0, "Embedding size must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

  

    def forward(self, x, mask=None):

        B, T, C = x.shape
        qkv = self.qkv_proj(x)  # [B, T, 3 * C]
        q, k, v = qkv.chunk(3, dim=-1)  # 分离 Q, K, V

  

        # reshape to [B, heads, T, head_dim]
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

  

        # scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:

        attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_probs, v)  # [B, heads, T, head_dim]

  

        # concatenate heads

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_output)

  

  

class TransformerBlock(nn.Module):

    """LLaMA2 的 Transformer Block"""

    def __init__(self, dim, num_heads, hidden_dim):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = FeedForward(dim, hidden_dim)

  

    def forward(self, x, mask=None):

        # 自注意力 + 残差连接
        x = x + self.attn(self.attn_norm(x), mask)
        # 前馈网络 + 残差连接
        x = x + self.ffn(self.ffn_norm(x))
        return x
```

transfomer中注意力部分代码（包含旋转位置代码）
```
class RotaryEmbedding:

    """LLaMA2 使用的 RoPE（旋转位置编码）"""

    def __init__(self, dim, base=10000):

        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

  

    def apply(self, q, k, seq_len):

        # 构建旋转位置编码

        t = torch.arange(seq_len, device=q.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

  

        # 分离奇偶维度

        cos_emb = emb.cos().unsqueeze(0).unsqueeze(0)
        sin_emb = emb.sin().unsqueeze(0).unsqueeze(0)

  

        def rotate(x):

            x1, x2 = x[..., ::2], x[..., 1::2]
            x_rot = torch.stack((-x2, x1), dim=-1).reshape_as(x)
            return x_rot

  

        q = q * cos_emb + rotate(q) * sin_emb
        k = k * cos_emb + rotate(k) * sin_emb

        return q, k

  

  

class MultiHeadAttention(nn.Module):

    def __init__(self, dim, num_heads, rope_base=10000):

        super().__init__()
        assert dim % num_heads == 0, "Hidden dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5  # 缩放因子
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.rope = RotaryEmbedding(self.head_dim, base=rope_base)

    def forward(self, x, mask=None):

        B, T, C = x.shape  # Batch, Time, Channels

        # QKV projection

        qkv = self.qkv_proj(x)  # [B, T, 3 * C]
        q, k, v = qkv.chunk(3, dim=-1)

        # Split heads

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, T, head_dim]
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

  

        # Apply RoPE
        q, k = self.rope.apply(q, k, seq_len=T)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_probs, v)  # [B, heads, T, head_dim]

        # Combine heads

        attn_output = attn_output.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(attn_output)
```

SwiGLU部分代码
```
class SwiGLU(nn.Module):

    """LLaMA2 的 SwiGLU 激活函数"""

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)  # 沿最后一维切成两半
        return F.silu(x1) * x2       # SiLU 激活后乘上另一半
```

前馈神经网络
```
class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.activation = SwiGLU()
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))
```

 **LLaMA2 Block 中的调用**
 ```
 class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = FeedForward(dim, hidden_dim)

    def forward(self, x, mask=None):
        x = x + self.attn(self.attn_norm(x), mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x
```


#**llma2本地部署

安装依赖
macos
```
# 安装编译工具和 CMake
brew install cmake
brew install wget  # 如果你需要下载模型
```
下载 LLaMA C 实现 (llama.cpp)
```
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
```
 编译 llama2.c
 ```
 mkdir build && cd build
cmake ..
make -j4  # -j4 表示4线程编译，可按 CPU 核心数调整
 ```
 ## **下载 LLaMA2 模型权重**

  

(1) 获取 Meta 官方权重

- 去 [Meta 官网](https://ai.meta.com/llama/) 申请下载 **LLaMA2 权重**
- 权重通常是 .pth 格式 + tokenizer.model
---

(2) 转换权重为 gguf 格式
llama.cpp 只支持 **.gguf** 格式权重
安装 Python 转换工具
```
pip install torch transformers sentencepiece
```
转换权重
```
# 在 llama.cpp/scripts 下执行
python3 convert-hf-to-gguf.py \
    --outtype f16 \
    --outfile llama-2-7b.gguf \
    meta-llama/Llama-2-7b-hf
```

本地部署为 API
llama.cpp 自带简单 HTTP API：
```
make server
./server -m ./llama-2-7b.gguf --port 8080
```
现在你可以用 cURL 调用：
```
curl -X POST http://localhost:8080/completion -d '{"prompt":"你好","n_predict":128}'
```
