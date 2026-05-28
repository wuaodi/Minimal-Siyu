# Basic — 基础内容选题地图

> `episodes/` 之外的「素材池」。
>
> 这里收集的不是学习清单，而是**可以做成视频的基础主题**：经典算法、常用库的关键模块、数学公式与几何工具。每一项都是潜在的一期 5–15 分钟视频。

工作流：从下面的列表里挑一项 → 在 `episodes/` 下新建一期 → 写完后把这里对应条目链到该期目录。

---

## 一、编程

### Python · LeetCode 经典题

挑选 LeetCode Hot 100 中**最值得讲的题**做成视频，每集一题，套路：「一句话讲清思路 + 最小代码 + 一个易错点」。

候选切片：

- 双指针 / 滑动窗口典型题
- 动态规划入门四件套（背包、LIS、LCS、编辑距离）
- 图论：最短路、并查集、拓扑排序
- 二分答案的几道经典题
- 单调栈 / 单调队列

### C++ 现代特性

每个特性单独一期，重点放在「为什么这样设计」而不是语法堆砌。


| 主题            | 视频切入角度                            |
| ------------- | --------------------------------- |
| `auto` / 类型推导 | 何时该用、何时坑                          |
| Lambda        | 捕获语义 + STL 算法配合                   |
| 智能指针          | `unique_ptr` vs `shared_ptr`，循环引用 |
| 移动语义          | 拷贝 vs 移动，一个 demo 看清楚              |
| pybind11      | Python 调 C++：一个最小可运行例子            |


### 自造轮子系列

适合做成「跟我手写一遍」型节目：链表、哈希表、LRU Cache、跳表、简易内存池。

---

## 二、常用库

两条主线：**手写原理** 与 **库 API 调用**。每期视频可以并排展示两边。

### PyTorch 核心


| 主题           | 原理拆解角度            | 库实现对照                                |
| ------------ | ----------------- | ------------------------------------ |
| Softmax + CE | 数值稳定              | `F.cross_entropy`, `F.softmax`       |
| Conv2d       | Im2Col 视角         | `nn.Conv2d`, `F.conv2d`              |
| BatchNorm    | train / eval 模式差异 | `nn.BatchNorm2d`, `.train()/.eval()` |
| Dropout      | 训练随机 / 推理缩放       | `nn.Dropout`, `F.dropout`            |
| Autograd     | 反向传播链式法则          | `loss.backward()`, `grad`, `no_grad` |
| AdamW        | 动量 + weight decay | `torch.optim.AdamW`, 参数组配置           |


### Transformer


| 主题            | 原理拆解角度         | 库实现对照                               |
| ------------- | -------------- | ----------------------------------- |
| Attention     | 单头 → 多头 → mask | `nn.MultiheadAttention`             |
| RoPE          | 旋转位置编码的几何意义    | transformers `LlamaRotaryEmbedding` |
| KV Cache      | 推理时的复用机制       | `use_cache=True`, `past_key_values` |
| GQA           | 多头到分组的折中       | Llama2/3 config                     |
| BPE Tokenizer | 子词切分 + 合并规则    | `AutoTokenizer.encode/decode`       |


### 训练 / 微调

- LoRA：低秩分解的直觉 + `peft` 用法
- 混合精度：`autocast`, `GradScaler` 是干嘛的
- 梯度累积：显存不够时的等效大 batch
- DeepSpeed / FSDP：一张图讲清三种并行

### 生成 / 采样

- Temperature / Top-K / Top-P：三个旋钮的几何含义
- Beam Search：为什么生成任务很少用它
- 投机解码 (Speculative Decoding)：用小模型抢跑

### Diffusion

- DDPM：前向加噪 + 逆向去噪的一对方程
- UNet 简化版：跟着画一遍
- Scheduler 三件套：DDPM, DDIM, Euler 区别
- diffusers Pipeline 拆解：`StableDiffusionPipeline` 内部是什么
- ControlNet / LoRA for SD：可控生成是怎么做到的

### 推理加速

- vLLM 的 PagedAttention：用一张图说清楚
- 量化对比：GPTQ vs AWQ vs bitsandbytes
- ONNX 导出：踩坑实录

### 强化学习

- Q-Learning → DQN：从表格到函数逼近
- Policy Gradient：从似然到 REINFORCE
- PPO：截断重要性采样的直观解释
- RLHF 流程（trl 库）：SFT → RM → PPO 三步走

### 几何（按需选）

- 四元数 SLERP：为什么不能直接线性插值
- SE(3) 变换：李群李代数最小够用版
- 卡尔曼滤波：先验 + 观测 = 后验
- ICP：点云配准是怎么收敛的

### 源码阅读系列

每期挑一个文件 / 模块「带读」：

- PyTorch：`autograd` / `nn.Module` / `optim`
- Transformers：`modeling_llama.py`
- vLLM：`attention` / `cache`
- diffusers：`UNet` / `scheduler`

---

## 三、数学

### 一期一公式

适合做成 3–5 分钟短视频，黑板板书风格。

- 反向传播链式法则
- Softmax 梯度
- Attention 梯度
- Adam 动量更新
- 重要性采样（PPO 用）
- SVD 分解的几何意义
- Box-Muller 高斯采样

### 「为什么是这样」系列

观点型视频，单期 5–8 分钟。

- 为什么 Attention 要除以 √d_k
- 为什么 AdamW 比 Adam 好
- 为什么 LayerNorm 而不是 BatchNorm
- KL 散度到底在度量什么

### 进阶数学（长线选题）

适合作为「番外篇」或多期连载。

- 李群李代数（SE3, SO3）：从机器人到 NeRF
- 黎曼度量与测地线：自然梯度的几何意义
- 凸优化与对偶
- 最优传输 / Wasserstein
- Neural ODE / SDE
- 随机过程：Diffusion 背后的数学
- 信息几何（Fisher 信息）
- 因果推断
- 博弈论

---

## 编辑约定

- 一项做成视频后，在条目后追加链接：`→ [episodes/2026-WXX-XXX](../episodes/2026-WXX-XXX/)`
- 想新增主题，直接在对应分节加一行即可
- 不再使用「必须 / 闭眼能写 / 修炼」这类自我要求口吻，统一成「可讲点 / 切入角度」的选题表达