# Minimal-Siyu

**每天一小时，磨练内功，保持手感。**

三条线：**编程** | **常用库** | **数学**

---

# 📅 本周 TODO（20251125周二 - 20251130周日）

| 天 | 日期 | 任务 | 具体内容 | 状态 |
|----|------|------|----------|------|
| Day 1 | 周二 | LeetCode | Hot 100 - 数组/双指针 2题 | [ ] |
| Day 2 | 周三 | LeetCode | Hot 100 - 数组/双指针 2题 | [ ] |
| Day 3 | 周四 | 常用库 | Softmax + CE 手写 + `F.cross_entropy` 调用 | [ ] |
| Day 4 | 周五 | 常用库 | PyTorch `autograd` 源码阅读 | [ ] |
| Day 5 | 周六 | 数学 | 链式法则推导 + Softmax 梯度 | [ ] |
| Day 6 | 周日 | 数学 | 反向传播公式复习 | [ ] |

---

# 每日节奏（6天制）

| 天 | 任务 | 时长 |
|----|------|------|
| Day 1 | LeetCode | 1h |
| Day 2 | LeetCode | 1h |
| Day 3 | 常用库：手写 + 调用 | 1h |
| Day 4 | 常用库：源码阅读 | 1h |
| Day 5 | 数学：公式推导 | 1h |
| Day 6 | 数学：概念复习 | 1h |

---

# 一个月

> 目标：建立节奏，每天都有明确动作

## 编程

LeetCode Hot 100 重复刷，直到闭眼能写。

## 常用库

两条腿走路：**手写原理** + **熟练调用**

| 模块 | 手写理解 | 库调用熟练 |
|------|----------|------------|
| Softmax + CE | 数值稳定实现 | `F.cross_entropy`, `F.softmax` |
| Attention | 单头→多头→mask | `nn.MultiheadAttention`, transformers |
| AdamW | 动量 + weight decay | `torch.optim.AdamW`, 参数组配置 |
| autograd | 反向传播链式法则 | `backward()`, `grad`, `no_grad` |

## 数学

复习/推导一个公式：

- [ ] 链式法则
- [ ] SVD 分解
- [ ] 高斯分布采样（Box-Muller）
- [ ] Softmax 梯度

---

# 一年

> 目标：核心模块闭眼能写，面试能讲清楚

## 编程

**Python**
- LeetCode Hot 100 反复刷到本能
- 中等稳定 AC，Hard 能做
- 周赛保持手感

**C++（选修但推荐）**
- 现代 C++ 特性：`auto`, `lambda`, `smart pointer`, `move semantics`
- STL 容器和算法熟练使用
- 用 C++ 造轮子：链表、哈希表、LRU Cache
- pybind11：Python 调用 C++ 扩展

## 常用库

**PyTorch 核心（手写 + 调用）**

| 手写理解 | 库调用熟练 |
|----------|------------|
| Conv2d (Im2Col) | `nn.Conv2d`, `F.conv2d` |
| BatchNorm (train/eval) | `nn.BatchNorm2d`, `.train()/.eval()` |
| Dropout | `nn.Dropout`, `F.dropout` |
| 完整 BP | `loss.backward()`, `optimizer.step()` |

**Transformer（手写 + 调用）**

| 手写理解 | 库调用熟练 |
|----------|------------|
| RoPE | transformers `LlamaRotaryEmbedding` |
| KV Cache | `use_cache=True`, `past_key_values` |
| GQA | Llama2/3 config |
| BPE Tokenizer | `AutoTokenizer`, `encode/decode` |

**训练/微调**
- [ ] LoRA（手写 + `peft` 库）
- [ ] 混合精度（`torch.cuda.amp`, `autocast`, `GradScaler`）
- [ ] 梯度累积（`accumulation_steps`）
- [ ] DeepSpeed / FSDP 基本配置

**生成/采样**
- [ ] Temperature / Top-K / Top-P（手写 + `generate()` 参数）
- [ ] Beam Search

**Diffusion**
- [ ] DDPM 原理（前向加噪 + 逆向去噪）
- [ ] UNet 结构（手写简化版）
- [ ] Scheduler（DDPM, DDIM, Euler）
- [ ] diffusers 库：`StableDiffusionPipeline`, `AutoencoderKL`, `UNet2DConditionModel`
- [ ] ControlNet / LoRA for SD
- [ ] 文生图 / 图生图 pipeline

**推理加速**
- [ ] vLLM 基本使用
- [ ] 量化：GPTQ, AWQ, bitsandbytes
- [ ] ONNX 导出

**强化学习**
- [ ] Q-Learning / DQN
- [ ] Policy Gradient
- [ ] PPO
- [ ] RLHF 流程（trl 库）

**几何（如需要）**
- [ ] 四元数 SLERP
- [ ] SE(3) 变换
- [ ] 卡尔曼滤波
- [ ] ICP

**源码阅读**
- PyTorch: `autograd` / `nn.Module` / `optim`
- Transformers: `modeling_llama.py`
- vLLM: `attention` / `cache`
- Diffusion: `UNet` / `scheduler`

## 数学

**必须能推导**
- [ ] 反向传播链式法则
- [ ] Softmax 梯度
- [ ] Attention 梯度
- [ ] Adam 动量更新
- [ ] 重要性采样（PPO 用）

**必须能解释**
- [ ] 为什么除以 √d_k
- [ ] 为什么 AdamW 比 Adam 好
- [ ] 为什么 LayerNorm 而不是 BatchNorm
- [ ] KL 散度的意义

---

# 十年

> 目标：底层思维，不随技术迭代而过时

## 编程

- 算法复杂度分析成为本能
- 系统设计能力（大规模、分布式）
- 新语言/范式快速上手
- **C++ 深入**：模板元编程、内存模型、并发

## 常用库

- 理解库背后的设计哲学
- 能读懂任何框架源码
- 能从零造轮子
- 跟踪前沿：新架构、新训练方法、新推理优化

**长期关注**
- 编译器/IR（TVM, Triton, MLIR）
- 并行计算（CUDA, 分布式）
- 推理优化（量化, 稀疏, 投机解码）
- C++ 算子开发（PyTorch C++ extension, CUDA kernel）

## 数学

**几何与流形**
- 李群李代数（SE3, SO3）
- 微分几何（黎曼度量、测地线）
- 自然梯度的几何意义

**优化与动力学**
- 凸优化、对偶理论
- 最优传输（Wasserstein）
- Neural ODE / SDE
- 控制理论（李雅普诺夫稳定性）

**概率与信息**
- 随机过程（Diffusion 的数学）
- 信息几何（Fisher 信息）
- 因果推断
- 博弈论

---

**原则**：
1. 不查资料，限时写，写不出来就是薄弱点
2. 每天都做，不是「有空做」
3. 不求新，求熟到本能
4. 手写理解原理，调用提升效率，两条腿走路
