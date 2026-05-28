# 2026-W23：vLLM × LoRA × SFT —— 在自己机器上「能跑、能调、能省」的大模型三件套

- 主题来源：Notion「20260525 vLLM 本地部署 LoRA 与 SFT」
- 视频链接：TBD
- 录制时长目标：10–15 分钟（领域开篇 / 系列引子）
- 一句话定位：把「微调一个开源大模型并部署到本地」这件事，拆成 **数据 → SFT 训练 → LoRA 省显存 → vLLM 高吞吐推理** 四段流水线，每一段挑出 1 篇必读经典 + 1 篇 2024 之后的代表作。

---

## 0. 这一期想做的事

观众类型：手上有一两张消费级 GPU（RTX 3090 / 4090 / A6000）、想在本地跑通「Llama / Qwen 微调 + 部署」全流程的研究者或工程师。

观众看完应当带走的三件事：

1. **一条流水线**：从一段对话数据开始，到一个能并发响应几十个请求的本地服务，中间每一步在做什么
2. **三个名字背后的论文**：vLLM 是 PagedAttention，LoRA 是 Low-Rank Adaptation，SFT 是 InstructGPT 那条 step 1
3. **一份书单**：每条线一篇必读 + 一篇近一年的代表作；最后再给「multi-LoRA serving」这条把三者粘起来的交叉线

不做的事：

- 不展开 RLHF / DPO（留给「对齐」系列）
- 不做量化深讲（GPTQ / AWQ / bitsandbytes 单开一期）
- 不讲分布式训练（FSDP / DeepSpeed 单开一期）

---

## 1. 一张总图：训练 → 微调 → 推理 的三段流水线

```
预训练大模型 W₀ ──► [SFT 阶段] ──► [LoRA 省显存] ──► [vLLM 部署]
   (Llama-3 / Qwen-2)        指令-回答对           ΔW = BA, r≪d        PagedAttention + Continuous Batching
```

- **W₀**：开源基座，几十亿到几百亿参数
- **SFT**：让模型「学会按指令说话」，损失就是下一 token 交叉熵，只是数据被组织成 `(instruction, response)` 对
- **LoRA**：训练时不改 W₀，只学一对低秩矩阵 `ΔW = B·A`，把可训参数量从 d² 降到 2dr
- **vLLM**：把模型装载到 GPU，按请求动态拼 batch、用 OS 风格的 paging 管 KV cache

> 这一期就沿着这条流水线走，每段挑代表论文。

---

## 2. SFT 线 —— 把基座变成「会聊天的助手」

### 2.1 必读经典

- **InstructGPT**（Ouyang et al., NeurIPS 2022, *Training language models to follow instructions with human feedback*）
  - 三步走的开端：**Step 1 = SFT**（约 13K 人工示范）→ Step 2 = RM → Step 3 = PPO
  - 1.3B 的 InstructGPT 在人评上压过 175B 原版 GPT-3，第一次让大家相信「微调 < 100 倍参数翻盘」是可能的
  - **arXiv：2203.02155**

- **FLAN**（Wei et al., ICLR 2022, *Finetuned Language Models Are Zero-Shot Learners*）
  - 把 60+ NLP 任务统统改写成自然语言指令，做大规模指令微调
  - 第一次系统验证「**指令格式本身**比任务多样性更重要」
  - **arXiv：2109.01652**

### 2.2 引爆开源 SFT 的两篇

- **Stanford Alpaca**（Taori et al., 2023, 技术报告）
  - 用 175 条种子指令 + GPT-3.5 自蒸馏出 52K 数据，3 小时 8×A100 微调 LLaMA-7B
  - 让「能 SFT 一个像样的助手」从大厂下放到小实验室

- **Vicuna**（Chiang et al., 2023）
  - 7 万条 ShareGPT 多轮对话 SFT，第一次把多轮对话 SFT 调成主流配方

### 2.3 「数据质量 > 数量」的代表作

- **LIMA: Less Is More for Alignment**（Zhou et al., NeurIPS 2023）
  - 只用 **1000 条** 精挑细选的 SFT 数据，微调 LLaMA-65B
  - 在盲评中能打平甚至略胜 Alpaca-52K 和 RLHF 后的 DaVinci-003
  - 提出「**Superficial Alignment Hypothesis**」：能力都在预训练里学完了，SFT 只是教模型用什么风格交付
  - **arXiv：2305.11206**

### 2.4 一篇近一年的代表作

- **LIMO: Less is More for Reasoning**（Ye et al., 2025）
  - 把 LIMA 的「少而精」哲学搬到推理任务，挑 817 条高质量数学题做 SFT，激活 DeepSeek-Math 等模型的推理能力
  - 验证 LIMA 假设在数学/推理域同样成立

### 2.5 一句话推荐

> 视频里挑 **InstructGPT (Step 1) → Alpaca → LIMA** 三篇，画一条「13K 人工 → 52K 蒸馏 → 1K 精选」的数据演进曲线。

---

## 3. LoRA 线 —— 不动原模型，只学一个「补丁」

### 3.1 必读经典

- **LoRA: Low-Rank Adaptation of Large Language Models**（Hu et al., ICLR 2022）
  - 核心假设：微调时权重的更新量 ΔW 内在秩很低
  - 把 ΔW 写成 `B·A`，其中 `B ∈ R^{d×r}`、`A ∈ R^{r×k}`，秩 `r` 取 4–16 就够
  - GPT-3 175B 上可训参数减少 10000 倍、显存省 3 倍，**推理时合并回 W 不引入额外延迟**
  - **arXiv：2106.09685**

数学版一行：
\[
W = W_0 + \alpha\,B A,\quad B\in\mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times k},\ r\ll \min(d,k)
\]

### 3.2 把 LoRA 推到「单卡微调 65B」的关键一步

- **QLoRA**（Dettmers et al., NeurIPS 2023）
  - 三个组合拳：**4-bit NF4 量化**基座 + **Double Quantization**（量化常数本身再量化）+ **Paged Optimizer**（统一内存避免显存溢出）
  - 单张 48GB GPU 微调 LLaMA-65B，效果和 16bit 全参数微调持平
  - 这是「在自己 4090 上跑通 13B 微调」的工程拐点
  - **arXiv：2305.14314**

### 3.3 同一时期的几个改进方向

| 方向 | 论文 | 一句话改进点 |
| ---- | ---- | ---- |
| 自适应秩 | **AdaLoRA**（Zhang et al., ICLR 2023, 2303.10512）| 不同层分配不同秩，重要的层多给参数 |
| 长序列 | **LongLoRA**（Chen et al., ICLR 2024, 2309.12307）| 用 shift short attention 把上下文从 4K 扩到 100K |
| 学习率 | **LoRA+**（Hayou et al., 2024, 2402.12354）| 给 A 和 B 设不同学习率，收敛更快 |

### 3.4 一篇近一年的代表作

- **DoRA: Weight-Decomposed Low-Rank Adaptation**（Liu et al., **ICML 2024 Oral**）
  - 把权重拆成 **方向 V** + **幅度 m**：方向继续用 LoRA 学，幅度单独学一个向量
  - 比 LoRA 多 0.01% 参数量，但在常识推理、视觉指令、视频文本上一致超 LoRA，甚至可以减半秩还更好
  - 推理时仍可合并回原权重
  - **arXiv：2402.09353**，NVIDIA 出品

### 3.5 一句话推荐

> 视频里画一条「**LoRA → QLoRA → DoRA**」三步曲：
> LoRA 解决「能不能不全调」，QLoRA 解决「显存够不够」，DoRA 解决「精度差不差」。

---

## 4. vLLM 线 —— 把模型「服务化」起来

### 4.1 必读经典

- **vLLM / PagedAttention**（Kwon et al., **SOSP 2023**, *Efficient Memory Management for Large Language Model Serving with PagedAttention*）
  - 观察：KV cache 是推理时的显存大头，传统连续分配浪费 60–80%
  - 思路：照搬 OS 虚拟内存的**分页**思想，把每条序列的 KV cache 切成固定大小（默认 16 token / block）的块，按需分配，碎片接近 0
  - 同一 prompt 的多次采样可以**copy-on-write 共享前缀块**
  - 比 FasterTransformer / Orca 提升 2–4× 吞吐
  - 团队：Berkeley Sky Lab，作者 Woosuk Kwon、Zhuohan Li、Ying Sheng、Lianmin Zheng、Hao Zhang、Ion Stoica
  - **arXiv：2309.06180**

KV cache 显存的直觉账：

```
KV_bytes = 2 × layers × hidden × seq_len × batch × dtype_bytes
         = 2 × 32   × 4096   × 2048    × 32    × 2     ≈ 32 GB   (Llama-7B)
```

> 这就是为什么 7B 模型权重只有 14 GB，但服务起来动辄 OOM。

### 4.2 vLLM 站在巨人的肩上

- **Orca**（Yu et al., **OSDI 2022**）
  - 提出 **iteration-level scheduling**（也叫 continuous batching）：每生成一个 token 就检查队列，能塞新请求就立刻塞进来
  - 同时提出 selective batching：attention 不 batch、其它算子 batch
  - 在 GPT-3 175B 上比 FasterTransformer 高 36.9× 吞吐
  - vLLM 直接继承了 continuous batching，再加 PagedAttention

- **FlashAttention** 系列
  - **FA-1**（Dao et al., NeurIPS 2022, 2205.14135）：tiling + online softmax，2–4× 加速
  - **FA-2**（Dao, 2023, 2307.08691）：更好的并行划分，再 2×
  - **FA-3**（Shah et al., 2024, 2407.08608）：吃满 Hopper 的 TMA + WGMMA + FP8，H100 上 1.5–2×
  - vLLM 的 attention kernel 默认就是 FA 系列

### 4.3 一篇近一年的代表作

- **vAttention**（Prabhu et al., **ASPLOS 2025**）
  - 反直觉地宣布：**不需要 PagedAttention 也能做动态 KV cache**
  - 利用 CUDA 低层的虚拟内存 API（`cuMemCreate`/`cuMemMap`）让 KV cache 在物理上仍然连续
  - 好处：注意力 kernel 不用为分页改写，直接复用 FlashAttention 等现成 kernel
  - 这是 PagedAttention 的「反方案」，2025 年争论最大的一篇 LLM serving 文章
  - **arXiv：2405.04437**

### 4.4 一句话推荐

> 视频里讲清两件事：
> **iteration-level scheduling**（Orca）解决「等队友的浪费」，**PagedAttention**（vLLM）解决「KV 碎片的浪费」。
> 想留个钩子讲下期，就提一句：**vAttention** 在挑战 PagedAttention 是不是必要。

---

## 5. 三者交汇 —— multi-LoRA serving

> 这是这一期最有意思的「彩蛋」：当一个团队训了一堆 LoRA（比如客服 / 代码 / 翻译各一个），怎么用一份基座模型同时服务它们？

### 5.1 必读

- **Punica: Multi-Tenant LoRA Serving**（Chen et al., **MLSys 2024**）
  - 自定义 CUDA kernel **SGMV (Segmented Gather Matrix-Vector multiplication)**
  - 一份基座 + N 份 LoRA，可以在同一 batch 里**同时服务不同 adapter** 的请求
  - 比把多份合并权重分开部署快 12×
  - **arXiv：2310.18547**

- **S-LoRA: Serving Thousands of Concurrent LoRA Adapters**（Sheng et al., **MLSys 2024**）
  - **Unified Paging**：把 LoRA adapter 权重和 KV cache 装进同一个分页内存池
  - 异构 batch：不同请求用不同秩的 LoRA 也能一起算
  - 单 GPU 上同时挂 **数千个 adapter**，吞吐比 vLLM-packed 高 4×
  - **arXiv：2311.03285**

### 5.2 vLLM 现状

- 这两篇的思想已经合进 vLLM 主分支：`--enable-lora` + `LoRARequest("name", id, "/path")` 即可在线热加载
- 这是「一台机器上线 N 个客户的微调模型」的现行最佳实践

### 5.3 一句话推荐

> 把第 3 节（LoRA）和第 4 节（vLLM）粘起来 ——
> 「训完十个 LoRA，怎么不开十份 vLLM？答案叫 S-LoRA / Punica。」

---

## 6. 一张大表（视频里的关键 Figure）

> 录视频时这张表会做成竖屏可读的版本，存到 `figures/landscape.png`。

| 维度 | SFT | LoRA | vLLM |
| ---- | ---- | ---- | ---- |
| 解决什么 | 学不会指令 | 全参数显存爆 | 推理吞吐低 |
| 核心思路 | (指令, 回答) 对做 next-token | ΔW = BA，r ≪ d | KV cache 分页 + 连续 batching |
| 代表论文 | InstructGPT (2022) / LIMA (2023) | LoRA (2021) / QLoRA (2023) / DoRA (2024) | Orca (2022) / vLLM (2023) / vAttention (2025) |
| 关键开源 | trl, axolotl, LLaMA-Factory | peft, bitsandbytes | vLLM, SGLang, TensorRT-LLM |
| 训/推 | 训练时 | 训练时 | 推理时 |
| 对显存影响 | 不变 | ↓↓（搭配 4-bit ↓↓↓） | KV cache 占用 ↓↓ |

---

## 7. 5 分钟讲稿（口播草稿）

> 此版只填了开场和过渡，细节录前再补。

**开场（30 秒）**

> 大家好，今天这期不挑一篇论文，挑三个名字：vLLM、LoRA、SFT。
> 这三个名字在干同一件事的不同段落 —— 把一个开源大模型，从下载下来，到能在你机器上像 ChatGPT 那样响应。
> 这一期我会沿着 **数据 → SFT → LoRA → vLLM** 这条流水线走一遍，每一段告诉你：要看哪一篇必读，要看哪一篇最新。

**过渡 1：什么是 SFT（约 1 分钟）**

> SFT 全称 Supervised Fine-Tuning，最早是 InstructGPT 那篇文章里的 step 1。
> 思想朴素到一句话能讲完：把数据组织成「指令 + 回答」对，损失还是下一 token 交叉熵。
> 真正难的是数据。Alpaca 用 5 万条蒸馏，LIMA 用 1000 条精选，结论是**质量比数量重要**。
> 看完这一节，你看到一个 SFT 数据集，第一反应应该是「这 1000 条干净不干净」，而不是「条数够不够」。

**过渡 2：从 SFT 到 LoRA（约 1 分钟）**

> SFT 的下一个问题是：65B 模型 16-bit 全参数微调要 800GB 显存，谁都跑不动。
> LoRA 的解法非常巧 —— 它说，反正你要学的是一个**很小的更新量**，那我把这个更新量写成两个低秩矩阵相乘 `B·A`，秩取 8 或 16。
> 可训参数从几十亿降到几百万，原模型权重一动不动，**推理时还能合并回去**。
> 再往前一步是 QLoRA：把基座先量化到 4-bit，叠加 LoRA，48GB 单卡能搞 65B。
> 想再进一步看精度，看 DoRA —— ICML 2024 Oral，把权重拆成方向加幅度，比 LoRA 又稳又准。

**过渡 3：从 LoRA 到 vLLM（约 1.5 分钟）**

> 微调好了，怎么把它跑起来？这就轮到 vLLM。
> vLLM 这篇 SOSP 2023 的论文叫 PagedAttention，核心观察是：推理时 KV cache 是显存大头，过去连续分配浪费 60% 以上。
> 它把 KV cache 切成固定大小的块，照搬 OS 的虚拟内存分页 —— 碎片接近 0，不同请求之间还能共享前缀。
> 再加上从 Orca 继承来的 continuous batching：哪个请求先生成完，立刻把队列里的新请求顶上去，GPU 一刻不闲。
> 结果就是一台机器吞吐能比 HuggingFace pipeline 高 20 倍以上。

**过渡 4：把三者粘起来（约 1 分钟）**

> 最有意思的是这三件事粘在一起的地方。
> 假设你训了十个 LoRA，难道开十份 vLLM？
> 不用 —— Punica 和 S-LoRA 这两篇 MLSys 2024 的工作，写了专门的 CUDA kernel，让一份基座 + 上千个 LoRA 在同一个 batch 里跑。
> 这套思想已经合进 vLLM 主分支：加一个 `--enable-lora` 就能在线热加载 adapter。
> 这就是为什么这一期把三个名字放在一起讲：它们已经长在一起了。

**收尾（30 秒）**

> 总结一句话：
> SFT 教模型怎么说话，LoRA 让微调便宜得起，vLLM 让推理跑得快。
> 下一期，我们挑这条流水线里的一段，把代码从头到尾敲一遍。

---

## 8. 配图与素材清单

> 录前补到 `figures/`：

- `pipeline.png`：第 1 节的「四段流水线」总图
- `lora_diagram.png`：LoRA 原文 Figure 1（W₀ + B·A 那张）
- `qlora_arch.png`：QLoRA 原文 Figure 1（NF4 + 反量化 + LoRA）
- `dora_decomposition.png`：DoRA 原文 Figure 1（方向 / 幅度拆分）
- `paged_attention.png`：vLLM 原文 Figure 6（block table → physical blocks）
- `continuous_batching.png`：Anyscale 博客那张静态 vs 连续 batching 对比
- `slora_unified_paging.png`：S-LoRA 原文 Figure 4（KV cache + adapter 共享内存池）
- `landscape.png`：第 6 节的三列大表

---

## 9. 后续期次的伏笔

| 期次 | 主题 | 这一期里如何挖坑 |
| ---- | ---- | ---- |
| W24 | LoRA 手撕：peft 源码 + 数学推导 | 第 3.1 节末尾留一句「下期我们手写一个 LoRALinear」 |
| W25 | QLoRA 量化深讲（NF4 / DQ / 反量化） | 第 3.2 节「QLoRA 三件套」做成预告 |
| W26 | vLLM 源码导读：PagedAttention kernel | 第 4.1 节末尾埋 GitHub 路径：`vllm/attention/ops/paged_attn.py` |
| W27 | 对齐系列开篇：DPO / PPO / RLHF | 第 0 节明确「不展开 RLHF，留给对齐系列」 |
| W28 | 投机解码（Speculative Decoding）专题 | 第 4 节末尾提一句「下期讲怎么用小模型抢跑」 |

---

## 附录：关键论文一句话清单（供视频补充资料 / 视频简介）

**SFT**

- *Training language models to follow instructions with human feedback*. Ouyang et al., NeurIPS 2022.（**InstructGPT**, arXiv:2203.02155）
- *Finetuned Language Models Are Zero-Shot Learners*. Wei et al., ICLR 2022.（**FLAN**, arXiv:2109.01652）
- *Stanford Alpaca: An Instruction-following LLaMA model*. Taori et al., 2023.
- *LIMA: Less Is More for Alignment*. Zhou et al., NeurIPS 2023.（arXiv:2305.11206）
- *LIMO: Less is More for Reasoning*. Ye et al., 2025.

**LoRA / PEFT**

- *LoRA: Low-Rank Adaptation of Large Language Models*. Hu et al., ICLR 2022.（arXiv:2106.09685）
- *QLoRA: Efficient Finetuning of Quantized LLMs*. Dettmers et al., NeurIPS 2023.（arXiv:2305.14314）
- *AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning*. Zhang et al., ICLR 2023.（arXiv:2303.10512）
- *LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models*. Chen et al., ICLR 2024.（arXiv:2309.12307）
- *DoRA: Weight-Decomposed Low-Rank Adaptation*. Liu et al., **ICML 2024 Oral**.（arXiv:2402.09353）
- *A Survey on LoRA of Large Language Models*. Mao et al., 2024.（arXiv:2407.11046，综述）

**vLLM / 推理服务**

- *Orca: A Distributed Serving System for Transformer-Based Generative Models*. Yu et al., OSDI 2022.
- *Efficient Memory Management for Large Language Model Serving with PagedAttention*. Kwon et al., **SOSP 2023**.（**vLLM**, arXiv:2309.06180）
- *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. Dao et al., NeurIPS 2022.（arXiv:2205.14135）
- *FlashAttention-2*. Dao, 2023.（arXiv:2307.08691）
- *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*. Shah et al., 2024.（arXiv:2407.08608）
- *vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention*. Prabhu et al., **ASPLOS 2025**.（arXiv:2405.04437）

**Multi-LoRA Serving（三者交汇）**

- *Punica: Multi-Tenant LoRA Serving*. Chen et al., **MLSys 2024**.（arXiv:2310.18547）
- *S-LoRA: Serving Thousands of Concurrent LoRA Adapters*. Sheng et al., **MLSys 2024**.（arXiv:2311.03285）
