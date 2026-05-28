# 2026-W22：NeRF / 3DGS / VGGT 与 逆渲染 —— 3D 正向渲染到逆向渲染的领域全景图

- 主题来源：Notion「20260525 NeRF 3DGS VGGT 与 逆渲染」
- 视频链接：TBD
- 录制时长目标：10–15 分钟（领域开篇 / 系列引子）
- 一句话定位：从一个公式（体渲染积分）讲起，串清楚「**正向渲染怎么算图**」与「**逆向渲染怎么从图反推世界**」两条主线，再点名后续要单独深讲的几篇代表作。

---

## 0. 这一期想做的事

观众类型：对 3D 视觉感兴趣、知道 NeRF 但没系统看过近两年进展的研究者 / 工程师。

观众看完应当带走的三件事：

1. **一个公式**：体渲染积分，能把 NeRF / 3DGS 都装进去
2. **一张地图**：正向渲染（NeRF / 3DGS / VGGT 前馈）↔ 逆向渲染（PBR-NeRF / GS-IR / Relightable3DGS）
3. **一份书单**：每条线 1–2 篇必读 + 1 篇最新代表作

不做的事：

- 不展开任何一篇论文的实现细节（留给后续期）
- 不讲驱动 / 重建 / 编辑等下游应用（另起一系列）

---

## 1. 从一个公式开始：体渲染积分

这一期的「钩子」。沿着相机射线 \(r(t) = o + td\) 把颜色累出来：

\[
C(r) = \int_{t_n}^{t_f} T(t)\,\sigma(r(t))\,c(r(t),d)\,dt,
\quad
T(t) = \exp\!\Big(-\!\int_{t_n}^{t} \sigma(r(s))\,ds\Big)
\]

口语版三句话：

- \(\sigma\) 是「每米吸收多少光」的密度，\(c\) 是「这一点朝相机的颜色」
- \(T(t)\) 是「光跑到这里还剩多少」（透射率）
- 最后把每段贡献一加总，就是这个像素的颜色

**它为什么是地图的中心？**

| 方法    | 体渲染积分被怎么实现                                      |
| ----- | ----------------------------------------------- |
| NeRF  | 用一个 MLP 表示 \(\sigma, c\)，沿射线**离散采样 + 数值积分**     |
| 3DGS  | 把场景表示成一堆 3D 高斯，**alpha-blending = 离散化的体渲染积分**  |
| VGGT  | 跳过这条积分：直接用 transformer 预测点云 / 深度 / 相机          |
| 逆渲染   | 把 \(c\) 拆成 BRDF × 光照，**回头反求**几何 + 材质 + 光        |

> 这就是这一期的总纲。后面四节就分别串这四条线。

---

## 2. NeRF 线 —— 隐式表示 + 体积渲染

> 必读 1 + 选读 2，目标是让观众明白「为什么 NeRF 之后还有这么多版本」。

### 2.1 必读

- **NeRF**（Mildenhall et al., ECCV 2020）—— 把场景写成一个 MLP，体渲染监督。

### 2.2 解决质量 / 速度问题的三件套

- **Mip-NeRF**（Barron et al., ICCV 2021）—— 用 cone 替代 ray，集成位置编码，干掉 aliasing
- **Instant-NGP**（Müller et al., SIGGRAPH 2022）—— 多分辨率 hash 网格，秒级训练
- **Mip-NeRF 360**（Barron et al., CVPR 2022）—— 无界场景 + proposal sampler
- **Zip-NeRF**（Barron et al., ICCV 2023）—— Mip-NeRF 抗锯齿 ⊕ Instant-NGP 速度
- **Ref-NeRF**（Verbin et al., CVPR 2022）—— 反射方向参数化，处理高光

### 2.3 一句话推荐

> 视频里挑 **NeRF + Instant-NGP + Zip-NeRF** 三篇画一条进化曲线就够。
> 想强调反射，就把 Ref-NeRF 也带上 —— 它也是 PBR-NeRF 的关键前作。

---

## 3. 3DGS 线 —— 显式高斯 + 光栅化

> 2023 年之后整个领域的重心从 NeRF 偏移到 3DGS。

### 3.1 原作

- **3D Gaussian Splatting**（Kerbl et al., SIGGRAPH 2023）—— 用 3D 各向异性高斯做显式表示，可微 splatting，1080p 实时渲染。

### 3.2 几何 / 抗锯齿方向

- **2D Gaussian Splatting (2DGS)**（Huang et al., SIGGRAPH 2024）—— 用 2D surfel 替代 3D 椭球，几何更准
- **Mip-Splatting**（Yu et al., CVPR 2024 Best Student Paper）—— 3D smoothing + 2D Mip filter，去掉缩放伪影
- **Scaffold-GS**（Lu et al., CVPR 2024）—— anchor + MLP 预测局部高斯，更结构化
- **Gaussian Opacity Fields (GOF)**（Yu et al., 2024）—— 高质量紧凑表面重建

### 3.3 一句话推荐

> 视频里画一条「2023 3DGS → 2024 Mip-Splatting / 2DGS」的进化线，
> 顺手丢一句「3DGS 的 alpha-blending 公式 = 第 1 节那条积分的离散版」。

---

## 4. VGGT 线 —— 前馈式 3D 重建

> 这是观众**最有可能没看过**的部分，也是这一期的「新意担当」。

### 4.1 必读

- **VGGT: Visual Geometry Grounded Transformer**（Wang et al., **CVPR 2025 Best Paper**）
  - 单/多/上百张图片 → 一次前向 → 直接吐出**相机内外参 + 深度图 + 点图 + 3D 点轨迹**
  - 在 RealEstate10K 上 0.2 秒达到 85.3 AUC@30；DUSt3R 要 7–10 秒；ETH3D 上比传统方法快 45 倍
  - 团队：Oxford VGG + Meta AI，开源在 `facebookresearch/vggt`

### 4.2 把 VGGT 放进语境的两篇前作

- **DUSt3R**（Wang et al., CVPR 2024）—— 两张图 → pointmap，开启「前馈几何」路线
- **MASt3R**（Leroy et al., ECCV 2024）—— DUSt3R + 匹配头，做大规模重建

### 4.3 一句话推荐

> 把 VGGT 讲成**「3D 重建的 GPT 时刻」**：
> 之前是 SfM / NeRF / 3DGS 的「逐场景优化」，VGGT 用一个大 transformer **预测**出 3D。
> 类比 LLM 直接预测下一个 token —— 不再每次都从零优化。

---

## 5. 逆渲染线 —— 从图反推几何 + 材质 + 光

> 把第 1 节体渲染里的 \(c(x,d)\) 拆成 \(\int f_r(x,\omega_i,\omega_o)\,L_i(x,\omega_i)\,(\omega_i\!\cdot\!n)\,d\omega_i\)（渲染方程），就到了逆渲染。

### 5.1 NeRF-based

- **NeRFactor**（Zhang et al., SIGGRAPH Asia 2021）—— 把 NeRF 拆成 几何 / 反照率 / 法向 / 可见度 / BRDF
- **NeRD / Neural-PIL** 系列（2021）—— 显式估计环境光 + BRDF
- **Ref-NeRF**（Verbin et al., 2022）—— 反射方向参数化（同时是 NeRF 改进 + 逆渲染前置）
- **NeRO**（Liu et al., SIGGRAPH 2023）—— 高反射物体的几何 + 材质恢复
- **PBR-NeRF**（2024, 2412.09680）—— 引入物理渲染先验做正则化的逆渲染 NeRF（→ 下一期 W23 深讲）

### 5.2 3DGS-based

- **GS-IR**（Liang et al., CVPR 2024）—— 把 3DGS 用到逆渲染：depth-derived 法向 + baked occlusion
- **Relightable 3D Gaussians (R3DG)**（Gao et al., ECCV 2024）—— 每个 Gaussian 带 BRDF + 入射光 + 点云 ray-tracing
- **GeoSplatting**（Ye et al., 2024, arXiv 2410.24204）—— 显式 mesh 引导的 PBR 逆渲染
- **RTR-GS**（2025, arXiv 2507.07733）—— Radiance Transfer + 反射，处理强反射物体

### 5.3 物理 / 光传输方向（CVPR 2025 风向）

- **Neural Inverse Rendering from Propagating Light**（**CVPR 2025 Best Student Paper**）—— 把光传播本身建进神经表示

### 5.4 一句话推荐

> 视频里说清三层：**几何 → 材质 / 光 → 重光照编辑**。
> 一句话总结：「把 NeRF 那个 \(c\) 换成渲染方程 = 逆渲染」。

---

## 6. 一张大表（视频里的关键 Figure）

> 录视频时这张表会做成竖屏可读的版本，存到 `figures/landscape.png`。

| 维度       | NeRF       | 3DGS         | VGGT         | 逆渲染         |
| -------- | ---------- | ------------ | ------------ | ----------- |
| 表示       | MLP 隐式     | 显式高斯         | Transformer 输出 | NeRF / 3DGS + BRDF |
| 渲染方式     | 体积积分（采样）   | 光栅化 splatting | 直接预测点云 / 深度  | 显式渲染方程       |
| 训练成本     | 数小时        | 数十分钟         | 一次前向（秒级）     | 比正向更慢       |
| 是否可重光照   | 否          | 否            | 否            | 是           |
| 代表年份     | 2020       | 2023         | 2025         | 2021–2025   |

---

## 7. 5 分钟讲稿（口播草稿）

> 此版只填了开场和过渡，细节录前再补。

**开场（30 秒）**

> 大家好，今天我们不讲某一篇论文，讲一条线。
> 你可能听过 NeRF，听过 3D Gaussian Splatting，最近又听到一个词叫 VGGT。
> 它们看起来在做不一样的事，但其实都在算同一个公式。
> 就是这个 —— 体渲染积分。看完这五分钟，你再看任何一篇 3D 论文，都能问出第一个问题：**这玩意儿是怎么把这条积分实现的？**

**过渡 1：从积分讲 NeRF（约 1 分钟）**

> NeRF 的做法最直接：把每一点的密度和颜色，扔进一个 MLP。
> 然后沿着射线采几十个点，把上面那条积分用加法近似。
> 训练就是「我画的这张图要和真实照片一样」。

**过渡 2：从 NeRF 跳 3DGS（约 1 分钟）**

> NeRF 慢在哪里？慢在每渲染一个像素都要查 MLP 几十次。
> 3DGS 把场景写成一堆有大小、有方向、有颜色的高斯。
> 然后**从前往后投影 + alpha 混合** —— 你仔细看，这个公式就是上面那条积分的离散版。
> 区别是：MLP 换成了一堆显式的高斯，迭代换成了 GPU 光栅化。所以快。

**过渡 3：跳出积分，进入 VGGT（约 1.5 分钟）**

> 那 VGGT 是什么？它直接说：我不要你逐场景优化了。
> 我训练一个大 transformer，喂它几张照片，它一次前向就告诉我相机在哪、深度多少、点在哪。
> 0.2 秒，对标过去几分钟的 NeRF / 3DGS 优化。
> 它是 CVPR 2025 的 Best Paper，Oxford × Meta。
> 我把它叫 **3D 重建的 GPT 时刻** —— 从「每个场景重新做题」变成「一个模型预测答案」。

**过渡 4：把渲染方程接回来，谈逆渲染（约 1 分钟）**

> 上面三类做的都是**正向渲染**：从 3D 算图。
> 反过来呢？我看到这张照片，能不能反推出**几何、材质、光**？
> 这就是逆渲染。把 NeRF 里那个简单颜色 \(c\) 换成完整的渲染方程，就是 NeRFactor、PBR-NeRF；
> 把 3DGS 里每个高斯加上 BRDF 和入射光，就是 GS-IR、Relightable 3DGS。
> 做完逆渲染，你就能换一盏灯重新打光。

**收尾（30 秒）**

> 总结一句话：
> NeRF 教会我们用神经网络做体渲染，3DGS 把这事做快，VGGT 把这事变成预测，逆渲染把这事做反。
> 下一期我们挑一篇 PBR-NeRF，把渲染方程怎么塞进 NeRF 这件事，从公式到代码讲一遍。

---

## 8. 配图与素材清单

> 录前补到 `figures/`：

- `landscape.png`：四列大表（同第 6 节）
- `volume_rendering.png`：体渲染积分示意（光线 + 采样点 + 透射率）
- `nerf_pipeline.png`：NeRF 原文 Figure 2
- `3dgs_splat.png`：3DGS 原文 splatting 示意
- `vggt_arch.png`：VGGT 原文架构图（alternating attention）
- `inverse_rendering.png`：渲染方程 + 拆解出 BRDF / 光 / 法向

---

## 9. 后续期次的伏笔

| 期次     | 主题                | 这一期里如何挖坑                        |
| ------ | ----------------- | ------------------------------- |
| W23    | PBR-NeRF 单篇精讲     | 第 5.1 节明确点名「下一期 PBR-NeRF」       |
| W24    | 3DGS 原作精讲（光栅化推导）  | 第 3.1 节末尾留一句「下一节我们手撕 3DGS 的微分」  |
| W25    | VGGT 架构精讲 + 跑通 demo | 第 4 节末尾留 GitHub 链接 + demo gif    |
| W26+   | 逆渲染深讲（GS-IR / R3DG / PBR-NeRF 三选一） | 第 5 节做成「下次细聊」预告                  |
