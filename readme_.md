# llama.cpp

这是一个基于 GGML 的高性能本地大语言模型推理库。

## 项目简介

llama.cpp 是一个用 C/C++ 编写的轻量级、高性能的大语言模型推理库。它支持多种模型格式，包括 GGUF、GGML 等，并且可以在 CPU 和多种 GPU 平台上高效运行。



**llam论文中的一些：**
scaling law：尺寸定律  尺度律
尺度律揭示了：只要模型规模、数据量和计算资源成比例增长，大语言模型的性能就可以持续提高。

目前因为数据量以及资源的影响 --》 强化学习  自己造数据自己学习  --》openai o1模型


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
      │  多头注意力（Self-Attention） 
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
8.KVCache    KVCache 是在生成文本时，缓存已经计算好的 K（Key）和 V（Value）向量，避免重复计算，提高推理速度和效率。
9.RoPE  为什么使用RoPE？
  RoPE（旋转位置编码） 是一种先进的位置编码方法，能让模型更自然、更平滑地理解“顺序”和“相对位置信息”，尤其适合处理长上下文，所以被广泛用于现代大语言模型。
10.ROPE与position encoding的区别以及为什么在llama中使用RoPE？
区别：RoPE（旋转位置编码）是一种相对位置编码方式，它通过对 Q/K 向量进行旋转而非加法注入位置信息，具备无参数、计算高效、支持长文本外推等优点，相比传统的绝对位置编码（如 sin/cos），在大模型中表现更好、更灵活。
原因：LLaMA 使用 RoPE 是因为它能更高效地建模相对位置信息，支持超长上下文、推理更快、更稳定，并能简化模型结构，是大语言模型架构中的主流选择。

**关于llma2.c的个人理解**

1. llma2.c的组成部分以及输入输出：
    主要由tokenizer transformer sampler三部分组成
    1️⃣Tokenizer负责将输入文本渲染的提示转换为一系列称为token的数字序列，每个数字都是指向词汇表中的数字索引。词汇表存储了模型支持的语言所使用的所有字符和子词——Llama2 英文模型有 32,000 个独特的token。
    2️⃣Transformer 负责将token通过训练好的权重运行，权重存储了模型所学习到的知识。每个token逐个通过模型单独运行。对于每个处理的token，Transformer 输出一个称为 logits 的数字向量。logits 向量中的每个条目代表词汇表中的 32,000 个token槽位之一。注意最后的logits维度是 32000 x 1
    3️⃣Sampler负责使用贪婪 arg-max 函数或基于概率的采样方法将 logits 转换回token。处理完输入的每个token后，采样器输出的token随后被反馈到 Transformer 中，作为生成下一个token的输入。这个过程一直持续到达到结束token。在每一步中，采样器输出的token也被解码回其文本表示，并输出给用户。

    输入：system prompt 和 user prompt    之后将两个输入以及历史对话等内容进行拼接 将拼接后的内容输入到tokenizer中去
    输出：采样器输出的token也被解码回其文本表示，并输出给用户。


2. Tokenizer Encoder
   Tokenizer分词器使用在初始化过程中加载的词汇表以及字节对编码（BPE）算法，将Rendered Prompt中的每个字符编码成一系列输入token

   BPE算法  BPE（Byte Pair Encoding） 是一种基于统计合并频率的分词算法，它从单个字符开始，逐步合并频繁出现的字符对，生成更大的子词单位。

   流程：
    在初始化过程中，分词器从 bin 文件中加载字节对编码词汇表。在处理提示时，rendered prompt被发送到分词器。在内部，分词器首先按字符串条目对词汇表进行排序，仅进行一次，以加快token查找速度。输入的每个字符首先被转换为对应的字符token，然后使用 BPE 算法进行合并，直到所有成对字符都被计算在内。 （char -> 单词的合并）返回的“压缩”后的token集作为输入token集。

3. transfomer
    Transformer 的主要目标是逐个处理每个token（从起始token开始），并生成与每个token对应的 logits，其中token的 logits 是一个 32,000 个数字的向量，对应于词汇表中的每个槽位。

    1️⃣Pre-normalization  RSMNorm
    2️⃣SwiGLU
    3️⃣Rotary Embeddings
    4️⃣masked部分为什么不计算权重以及kv score
    5️⃣auto-regressive transfomers 
    6️⃣基于人类反馈的强化学习（RLHF）
    7️⃣对齐（Alignment）
    8️⃣GQA的使用(llama1与llama2的主要区别)


