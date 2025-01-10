# transformer_pytorch

**English:**

The complete original version of the Transformer program, supporting padding operations, written in PyTorch, suitable for students who are new to Transformer. The code syntax is relatively simple.

I wrote this program to solidify my understanding of the Transformer and to demonstrate my ability to write code based on research papers.

**Chinese:**

完整的原版 Transformer 程序，支持 padding 操作，使用 PyTorch 编写，适合初次接触 Transformer 的同学，代码语法较为简单。

我写这个程序是为了巩固我对 Transformer 的理解，并且证明我有根据论文编写代码的能力。

```markdown
Paper: Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need[J]. Advances in neural information processing systems, 2017, 30.

Paper site: [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
```

### English Introduction

1. **Positional Encoding:**
   - The code provides two methods for positional encoding: `LearnablePositionalEncoding` and `PositionalEncoding`.
   - `LearnablePositionalEncoding` learns positional embeddings as model parameters.
   - `PositionalEncoding` generates positional encodings using sine and cosine functions.
  
2. **Multi-Head Attention:**
   - Implements multi-head attention mechanism. 
   - Divides the input into multiple heads and computes attention separately for each head.

3. **Feedforward Network:**
   - Defines a simple feedforward neural network with a ReLU activation function.

4. **Masking Functions:**
   - Provides functions for generating attention masks to handle padding in sequences.
   - `attn_mask(len)` generates a mask to prevent attention to subsequent positions in the sequence.
   - `padding_mask(pad_q, pad_k)` generates a mask to prevent attention to padded elements in the sequence.

5. **Encoder and Decoder Layers:**
   - Implements Encoder and Decoder layers, each containing multi-head attention and feedforward sub-layers.
   - The `EncoderLayer` applies self-attention followed by a position-wise feedforward network.
   - The `DecoderLayer` applies self-attention over the target sequence followed by cross-attention over the encoder output.

6. **Encoder and Decoder Stacks:**
   - Constructs the encoder and decoder stacks using multiple layers of encoder/decoder blocks.

7. **Transformer Model:**
   - Combines the encoder and decoder modules to form a complete Transformer model.
   - The model takes source and target sequences as input and outputs predicted target sequences.
   - Also provides a function `get_mask` to generate masks for handling padding during training.

### Chinese Introduction

1. **位置编码：**
   - 代码提供了两种位置编码的方法：LearnablePositionalEncoding 和 PositionalEncoding。
   - `LearnablePositionalEncoding` 学习位置嵌入作为模型参数。
   - `PositionalEncoding` 使用正弦和余弦函数生成位置编码。

2. **多头注意力：**
   - 实现了多头注意力机制。
   - 将输入分成多个头，并为每个头单独计算注意力。

3. **前馈网络：**
   - 定义了一个简单的前馈神经网络，使用 ReLU 激活函数。

4. **掩码函数：**
   - 提供了用于生成注意力掩码以处理序列中填充的函数。
   - `attn_mask(len)` 生成一个掩码，防止对序列中后续位置的注意力。
   - `padding_mask(pad_q, pad_k)` 生成一个掩码，防止对序列中填充的元素的注意力。

5. **编码器和解码器层：**
   - 实现编码器和解码器层，每个层包含多头注意力和前馈子层。
   - `EncoderLayer` 应用自注意力，然后是位置逐元素前馈网络。
   - `DecoderLayer` 在目标序列上应用自注意力，然后是对编码器输出的交叉注意力。

6. **编码器和解码器堆叠：**
   - 使用多层编码器/解码器块构建编码器和解码器堆叠。

7. **Transformer 模型：**
   - 将编码器和解码器模块组合成完整的 Transformer 模型。
   - 模型接受源序列和目标序列作为输入，并输出预测的目标序列。
   - 还提供了一个函数 `get_mask`，用于生成处理填充时所需的掩码。

*** If there are any questions or bugs, you can create an issue or contact 1793706453@qq.com. ***
