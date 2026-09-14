# Beyond H-Neurons：区分 LLM Hallucination 的诊断性神经元与因果机制

## 1. 暂定题目

### 首选标题

**Beyond H-Neurons: Disentangling Diagnostic Readouts from Causal Mechanisms of Hallucination in Large Language Models**

中文：

**超越 H-Neurons：区分大语言模型幻觉中的诊断性表征与因果机制**

### 如果后期 H-Circuit 结果非常强

**From H-Neurons to H-Circuits: Causal Tracing of Hallucination in Large Language Models**

### 如果最终更偏 ICML 方法论文

**Diagnostic–Causal Decomposition of Hallucination Representations in Large Language Models**

---

# 2. 一句话核心思想

现有 H-Neuron 工作主要回答：

> **哪些神经元能够预测模型即将/正在 hallucinate？**

但：

\[
\text{predictive association}
\neq
\text{causal responsibility}.
\]

我们希望进一步回答：

> **真正改变模型 factual outcome 的内部表示究竟在哪里？它与 H-Neurons 有多少重合？如果 H-Neurons 只是 downstream diagnostic readout，那么真正的 hallucination mechanism 是否存在于一个 distributed causal subspace / circuit 中？**

因此，本工作的核心不是再寻找一批“更好的 H-Neurons”，而是提出：

\[
\boxed{
\text{Diagnostic Hallucination Representation}
\neq
\text{Causal Hallucination Representation}
}
\]

并通过严格的 counterfactual intervention 对两者进行分解。

---

# 3. Research Motivation

## 3.1 H-Neurons 提出的重要发现

H-Neurons 从 FFN neurons 出发，通过 CETT 描述 neuron 对 residual stream 的贡献，再使用稀疏线性分类器识别极少数与 hallucination 高度相关的 neurons。

原工作发现：

\[
<0.1\%
\]

的 FFN neurons 即能够提供较强的 hallucination predictive signal，并进一步发现这些 neurons 与 over-compliance 行为存在因果联系；作者还发现这些 neurons 在对应 base model 中已经具有 hallucination predictive ability，因此推测它们在 pretraining 中形成。

官方代码已经提供 CETT extraction、sparse probing 和 neuron intervention，因此我们可以直接在其代码基础上进行开发，而不是重新构建完整 pipeline。

---

## 3.2 第一个 unresolved problem：predictive neuron 是否等于 causal neuron？

H-Neurons 的 identification 阶段实质是在学习：

\[
P(Y_{\mathrm{hall}}=1\mid\mathbf c),
\]

其中：

\[
\mathbf c=
[c_{1,1},c_{1,2},...,c_{L,d_{\mathrm{ff}}}]
\]

为 CETT features。

如果某个 neuron：

\[
c_j
\]

对 classifier 非常重要，这只能证明：

\[
I(c_j;Y_{\mathrm{hall}})>0,
\]

也就是它携带 hallucination information。

它并不能直接证明：

\[
do(c_j)\Longrightarrow Y_{\mathrm{hall}}
\]

发生变化。

换言之：

\[
\boxed{
\text{Readable information}
\neq
\text{Used information}.
}
\]

这是本项目最核心的理论出发点。

---

# 4. 为什么现在正好可以做这个问题？

H-Neurons 之后出现了几组互相并不完全一致的证据。

### Evidence A：H-Neurons 可以很好地预测 hallucination

原工作说明 sparse H-Neurons 具有明显 detection ability。

### Evidence B：H-Neurons 的跨域 generalization 很弱

2026 年的 *Do Hallucination Neurons Generalize?* 在 general QA、legal、financial、science、moral reasoning 和 code 六类任务及五个 3B–8B open-weight LLM 上发现：

\[
AUROC_{\mathrm{within}}=0.783
\]

而：

\[
AUROC_{\mathrm{cross}}=0.563.
\]

即 exact neuron-level signature 存在明显 domain specificity。

### Evidence C：某些特定 hallucination 中 neuron intervention 又确实有效

*Where Fake Citations Are Made* 在 citation hallucination 中发现 field-specific hallucination neurons，并报告 activation amplification 会增加错误、suppression 能改善结果。

因此现在存在一个比简单“有没有 H-Neurons”更重要的问题：

\[
\boxed{
\text{什么时候 predictive H-Neurons 是 causal 的，
什么时候只是 diagnostic 的？}
}
\]

我们不应该预设：

> H-Neurons 全部不是 causal。

更严谨的 hypothesis 应该是：

> **Predictivity 与 causality 是两个独立维度；不同 hallucination regime 中二者的 alignment 程度可能不同。**

这使得论文即使最后发现“部分 H-Neurons 确实 causal”，依然成立。

---

# 5. 与现有 subspace / causal hallucination 工作的区别

这是项目能否投稿 ACL/ICML 的关键。

## HARP

HARP 已在 ICLR 2026 使用 SVD 从 unembedding space 中分离 semantic/reasoning subspace，并利用 reasoning subspace 做 hallucination detection，TriviaQA 上报告了很高的 AUROC。

所以我们不能声称：

> “第一次发现 hallucination subspace。”

我们的区别是：

\[
\boxed{
\text{HARP: 哪个 subspace 更适合 detection?}
}
\]

而我们问：

\[
\boxed{
\text{哪个 internal representation 真正具有 causal effect?}
}
\]

---

## SPACE

NeurIPS 2025 的 SPACE 已经研究 factuality 与 faithfulness hallucination 的 shared activation subspace，并进行 activation-space editing。

因此：

> “发现一个 hallucination subspace，然后 steering”

本身也不能作为我们的核心 novelty。

我们的 subspace 必须通过：

\[
\textbf{interventional causal evidence}
\]

定义，而不是通过普通 activation covariance / classifier representation 得到。

---

## LANCET

LANCET 已经提出从 neuron 追踪 hallucination propagation pathway，并使用 structural entropy 建立 cross-layer path，再对 pathway 进行 intervention。

因此我们也不能简单声称：

> “第一次从 neuron 扩展到 pathway。”

我们的核心不是 pathway discovery，而是：

\[
\boxed{
\text{Predictive H-Neurons 与 causal mechanism 是否是同一组内部计算？}
}
\]

这是一个 measurement / causality question。

---

## CausalGaze

ACL 2026 Findings 的 CausalGaze 已经建立 dynamic causal graph，通过 counterfactual intervention 提高 hallucination detection。

因此我们与它最重要的区别应该明确写为：

> CausalGaze 的主要目标仍然是构建 causal graph 改进 hallucination detection；本工作则直接研究 **diagnostic representation 与 causal mechanism 的错位问题**，并以 H-Neurons 为具体 scientific case study。

---

# 6. 核心 Research Questions

整篇论文建议只围绕三个 RQ。

---

## RQ1：Hallucination-predictive neurons 是否也是 hallucination-causal neurons？

对于 neuron \(j\)，分别定义：

\[
D_j=\text{Diagnostic Importance}
\]

以及：

\[
C_j=\text{Causal Importance}.
\]

研究：

\[
\operatorname{Corr}(D,C)
\]

到底有多大。

我们的目标不是证明：

\[
D\perp C
\]

而是系统测量二者的 correspondence。

理想发现可能是：

\[
\rho(D,C)\ll1,
\]

并存在大量：

\[
D_j\gg0,\quad C_j\approx0.
\]

这类 neuron 即：

### Diagnostic H-Neurons

同时可能存在：

\[
D_j\approx0,\quad C_j\gg0,
\]

即：

### Hidden Causal Components

这类 component 对 classifier 不突出，但真正影响 factual output。

---

## RQ2：如果 causal information 不集中在 sparse H-Neurons，它以什么形式存在？

三个竞争假设：

### Hypothesis A：Axis-aligned neuron mechanism

\[
\text{Hallucination}
\approx
\{h_{j_1},h_{j_2},...\}
\]

即少数 individual neurons 就足够。

### Hypothesis B：Low-rank distributed representation

\[
z_H=U_H^\top h
\]

hallucination causal signal 存在于一个低维 subspace，而不是 individual axes。

### Hypothesis C：Distributed cross-layer circuit

\[
\text{Attention}
\rightarrow
\text{MLP}
\rightarrow
\text{Residual}
\rightarrow
\text{Output}
\]

hallucination 是多个 layer/components 联合形成的 computation。

我们通过 intervention 比较三种解释。

---

## RQ3：什么可以跨 domain generalize？

cross-domain H-Neurons 已经显示：

\[
\text{Neuron}_{d_1}
\not\approx
\text{Neuron}_{d_2}.
\]

但这仍然不能说明：

\[
\text{CausalRepresentation}_{d_1}
\not\approx
\text{CausalRepresentation}_{d_2}.
\]

因此我们分别测：

\[
\text{Neuron overlap},
\]

\[
\text{Diagnostic subspace similarity},
\]

\[
\text{Causal subspace similarity},
\]

以及最重要的：

\[
\text{Cross-domain intervention transfer}.
\]

可能出现一个非常有价值的结果：

\[
J(H_{d_1},H_{d_2})\approx0,
\]

但：

\[
\operatorname{Sim}(U^C_{d_1},U^C_{d_2})\gg0.
\]

即：

\[
\boxed{
\text{Domain-specific neurons,
shared causal geometry.}
}
\]

如果这个结果成立，论文会明显更接近 ICML。

---

# 7. Method Overview

整个方法分成四层：

\[
\boxed{
\text{H-Neuron Identification}
}
\]

↓

\[
\boxed{
\text{Diagnostic–Causal Decomposition}
}
\]

↓

\[
\boxed{
\text{Causal H-Subspace}
}
\]

↓

\[
\boxed{
\text{Minimal H-Circuit}
}
\]

其中前三层是 **must-have**。

H-Circuit 是增强版，不建议一开始投入大量工程。

---

# 8. Stage I：复现 Diagnostic H-Neurons

对于 Transformer 第 \(l\) 层 FFN：

\[
a_l=\phi(W_{\mathrm{up}}h_l,\;W_{\mathrm{gate}}h_l),
\]

其输出为：

\[
o_l=W_{\mathrm{down}}a_l.
\]

对于 FFN neuron \(j\)，按照 H-Neurons pipeline 使用 CETT：

\[
c_{l,j}
=
a_{l,j}
\left\|
W_{\mathrm{down}}[:,j]
\right\|_2.
\]

这一 quantity 同时考虑 neuron activation 和其向 residual stream 投影时的 magnitude。后续 cross-domain 工作也按照该定义重新实现 H-Neuron extraction。

对于全部 neurons：

\[
\mathbf c(x)
=
[c_{1,1},...,c_{L,d_{\mathrm{ff}}}].
\]

训练：

\[
P(Y_H=1|\mathbf c)
=
\sigma
(\mathbf w^\top\mathbf c+b)
\]

并加入：

\[
\lambda\|\mathbf w\|_1.
\]

定义：

\[
\boxed{
D_{l,j}=|\hat w_{l,j}|
}
\]

为 Diagnostic Importance。

为了防止不同层 coefficient scale 不一致，可进一步采用：

\[
\tilde D_{l,j}
=
|\hat w_{l,j}|
\cdot
\operatorname{Std}(c_{l,j}).
\]

最终得到：

\[
\mathcal H_D
=
\operatorname{TopK}
(\tilde D).
\]

这一阶段严格复现原 H-Neurons，而不是修改它。

这样 reviewer 无法质疑：

> “你所谓 diagnostic neuron 与 H-Neurons 根本不是一个东西。”

---

# 9. Stage II：Diagnostic–Causal Decomposition

这是本文真正的方法核心。

## 9.1 Factual Margin

对每个 factual QA example：

\[
x
\]

设 gold answer：

\[
y^+
\]

以及模型产生的主要 hallucinated answer：

\[
y^-.
\]

定义 length-normalized factual margin：

\[
M(x)
=
\frac{1}{|y^+|}
\log P(y^+|x)
-
\frac{1}{|y^-|}
\log P(y^-|x).
\]

如果：

\[
M>0,
\]

模型内部更偏向事实答案；

如果：

\[
M<0,
\]

更偏向 hallucinated answer。

这比只使用：

\[
\text{Accuracy}\in\{0,1\}
\]

更适合作为 causal intervention metric，因为它可以测量干预后 output distribution 的连续变化。

---

# 10. 一个重要的方法修正：不能简单把 same-prompt stochastic pair 当成主要 causal evidence

一个看似自然的方法是：

同一个 prompt：

\[
x
\]

采样两次：

\[
x\rightarrow y^F
\]

和：

\[
x\rightarrow y^H.
\]

这类 pair 可以很好地分析 hallucination trajectory。

但是它存在一个重要 causal limitation：

在两个 generation 第一次 sampling 分叉以前，模型获得的：

\[
\text{prompt prefix}
\]

完全相同。

对于 deterministic forward pass：

\[
h_t^F=h_t^H.
\]

两个 trajectory 的真正差异来自第一次 sampled token。

所以这类 pair 特别适合研究：

> hallucination 如何被维持和传播，

却不能单独证明：

> hallucination 在分叉之前由哪个内部 representation 产生。

因此本文将采用两个 complementary pair protocols。

---

# 11. Counterfactual Pair Protocol

## Protocol A：Natural Hallucination Pairs

用于主要 observational analysis。

选取语义、answer type、difficulty 接近的 factual/hallucinated samples：

\[
(x_i^F,x_i^H).
\]

通过 propensity matching 控制：

- answer length；
- question length；
- token entropy；
- entity frequency；
- prompt template；
- domain。

主要用于估计：

\[
\Delta h_l
=
h_l^H-h_l^F.
\]

---

## Protocol B：Controlled Counterfactual Pairs

这是主要 causal identification protocol。

对于同一问题：

\[
q
\]

构造：

\[
x^F=(q,c^F)
\]

和：

\[
x^H=(q,c^H),
\]

其中：

\[
c^F
\]

提供真实且足够 evidence，

而：

\[
c^H
\]

只进行最小扰动，例如：

- 替换关键 entity；
- 替换数字；
- 删除关键 factual evidence；
- 加入 plausible but false evidence。

要求：

\[
\operatorname{EditDistance}(c^F,c^H)
\]

尽可能小。

并筛选模型行为：

\[
f(x^F)=y^+,
\]

\[
f(x^H)=y^-.
\]

因此得到：

\[
(x^F,x^H,y^+,y^-).
\]

这样 factual 与 hallucinated states 之间只存在受控变量差异。

---

# 12. Bidirectional Causal Patching

对于 layer \(l\) component \(j\)，设：

\[
z_{l,j}^F
\]

来自 factual run，

\[
z_{l,j}^H
\]

来自 hallucinated run。

## Denoising intervention

在 hallucinated trajectory 中：

\[
z_{l,j}^H
\leftarrow
z_{l,j}^F.
\]

定义：

\[
R_{l,j}^{H\rightarrow F}
=
M(x^H;
do(z_{l,j}=z_{l,j}^F))
-
M(x^H).
\]

如果：

\[
R^{H\rightarrow F}>0,
\]

则 factual activation 可以部分恢复事实倾向。

---

## Noising intervention

反过来：

\[
z_{l,j}^F
\leftarrow
z_{l,j}^H.
\]

定义：

\[
R_{l,j}^{F\rightarrow H}
=
M(x^F)
-
M(x^F;
do(z_{l,j}=z_{l,j}^H)).
\]

如果：

\[
R^{F\rightarrow H}>0,
\]

说明 hallucinated activation 能够破坏 factual behavior。

---

## Symmetric Causal Score

最终：

\[
\boxed{
C_{l,j}
=
\frac{1}{2}
\left(
R_{l,j}^{H\rightarrow F}
+
R_{l,j}^{F\rightarrow H}
\right)
}
\]

只在两个方向都成立时给予较高 causal score。

这比只做：

\[
a_j\leftarrow0
\]

强很多。

因为 neuron ablation 可能只是破坏网络。

而：

\[
F\leftrightarrow H
\]

的双向干预更接近：

- necessity；
- sufficiency；

的联合证据。

---

# 13. Normalized Causal Recovery

为避免不同样本 factual margin scale 不同，我们定义：

\[
\operatorname{NCR}_{l,j}^{H\rightarrow F}
=
\frac{
M_{\mathrm{patched}}-M_H
}{
M_F-M_H+\epsilon
}.
\]

解释：

\[
NCR=0
\]

表示 patch 没效果；

\[
NCR=1
\]

表示该 component 单独恢复了完整 factual–hallucination gap。

也可以出现：

\[
NCR>1
\]

代表 over-recovery。

主论文中使用：

\[
\boxed{\text{Normalized Causal Recovery}}
\]

作为核心 causal metric。

---

# 14. 论文最重要的二维空间

每一个 neuron 都有：

\[
(D_j,C_j).
\]

于是产生四种 neuron：

| 类型 | Diagnostic \(D\) | Causal \(C\) | 含义 |
|---|---:|---:|---|
| Diagnostic Readout | 高 | 低 | 可以检测 hallucination，但不是 causal driver |
| Causal Driver | 低 | 高 | probe 看不到，但真正改变 factual output |
| Diagnostic-Causal | 高 | 高 | 真正意义上的核心 H-Neuron |
| Irrelevant | 低 | 低 | 无明显作用 |

论文最关键的 Figure 2：

横轴：

\[
D_j
\]

纵轴：

\[
C_j.
\]

同时报告：

\[
\rho_S
=
\operatorname{Spearman}(D,C)
\]

以及：

\[
J_K
=
\frac{
|\operatorname{TopK}(D)\cap\operatorname{TopK}(C)|
}{
|\operatorname{TopK}(D)\cup\operatorname{TopK}(C)|
}.
\]

如果：

\[
\rho_S
\]

和：

\[
J_K
\]

都不高，就能非常直接地说明：

\[
\boxed{
\text{The neurons that best predict hallucination
are not necessarily those that cause it.}
}
\]

---

# 15. Stage III：从 neuron 到 Causal H-Subspace

如果 causal effect 并不高度 axis-aligned，我们进一步研究：

\[
\boxed{
\text{Hallucination causal signal 是否分布在一个低维 subspace？}
}
\]

这里不能简单：

\[
\operatorname{PCA}(h_H-h_F)
\]

因为这仍然只是 correlational representation。

因此我们提出：

## Intervention-Weighted Causal Subspace

首先进行 layer-level patching：

\[
h_l^H\leftarrow h_l^F.
\]

获得每个 example：

\[
r_{i,l}
=
NCR_i(l).
\]

然后定义 pair difference：

\[
\Delta h_{i,l}
=
h_{i,l}^H-h_{i,l}^F.
\]

使用 causal recovery 作为权重：

\[
w_{i,l}
=
\max(r_{i,l},0).
\]

构造：

\[
\Sigma_l^C
=
\frac{
\sum_i
w_{i,l}
\Delta h_{i,l}
\Delta h_{i,l}^{\top}
}{
\sum_iw_{i,l}
}.
\]

进行 eigendecomposition：

\[
\Sigma_l^C
=
U_l\Lambda_lU_l^\top.
\]

取前 \(k\) 个方向：

\[
U_l^C
=
[u_1,...,u_k].
\]

定义：

\[
\boxed{
\mathcal S_l^C
=
\operatorname{span}(U_l^C)
}
\]

为 candidate Causal Hallucination Subspace。

---

# 16. 为什么这里比普通 PCA/HARP 更强？

普通 subspace：

\[
U_{\mathrm{corr}}
=
\operatorname{SVD}
(h_H-h_F)
\]

回答：

> 哪些 activation directions 区分 hallucination 与 factual response？

我们的方法加入：

\[
w_i=NCR_i.
\]

因此优先保留：

> **被实际 intervention 验证过能够改变 factual outcome 的 representation differences。**

也就是说：

\[
\boxed{
\text{Correlation-weighted geometry}
\rightarrow
\text{Intervention-weighted geometry}.
}
\]

但为了避免过度宣称，我们不会仅凭：

\[
U^C
\]

的构造就称其为 causal。

最终必须重新做真正 intervention 验证。

---

# 17. Subspace-level Intervention

对于 hallucinated hidden state：

\[
h_l^H,
\]

计算 factual 与 hallucinated centroid：

\[
\mu_l^F,\qquad
\mu_l^H.
\]

定义 causal factual direction：

\[
v_l^C
=
U_l^CU_l^{C\top}
(\mu_l^F-\mu_l^H).
\]

然后：

\[
h_l'
=
h_l+\eta v_l^C.
\]

或者更保守地仅替换 causal projection：

\[
h_l'
=
(I-U_l^CU_l^{C\top})h_l^H
+
U_l^CU_l^{C\top}h_l^F.
\]

比较：

\[
\Delta M_{\mathrm{CausalSubspace}}
\]

与：

\[
\Delta M_{\mathrm{HNeuron}}
\]

和：

\[
\Delta M_{\mathrm{CorrelationSubspace}}.
\]

核心假设：

\[
\Delta M_{\mathrm{CausalSubspace}}
>
\Delta M_{\mathrm{HNeuron}}
\]

同时：

\[
\Delta\text{Utility}
\]

更小。

---

# 18. 必须加入 Subspace Patching 的安全控制

已有 mechanistic interpretability 研究已经指出：

> subspace activation patching 有可能人为激活 dormant pathway，产生“看起来有意义”的虚假 subspace interpretation。

因此本文必须包含以下 controls：

### Random subspace

与：

\[
\dim(U_C)
\]

完全相同。

### PCA difference subspace

使用相同 factual/hallucination activations，但没有 causal weighting。

### Probe subspace

通过 linear classifier direction 得到。

### Orthogonal complement

测试：

\[
U_C^\perp
\]

是否缺乏相同干预效果。

### Projection magnitude control

保证不同方法：

\[
\|\Delta h\|_2
\]

一致。

否则 intervention 强度不同，会造成不公平比较。

---

# 19. Stage IV：Cross-Domain Causal Geometry

对 domain：

\[
d\in
\{\text{General},
\text{Biomedical},
\text{Science},
...\},
\]

分别得到：

\[
H_d,
\quad
U_d^D,
\quad
U_d^C.
\]

其中：

\[
H_d
\]

是 H-Neuron set；

\[
U_d^D
\]

是 diagnostic subspace；

\[
U_d^C
\]

是 causal subspace。

---

## 19.1 Neuron overlap

\[
J(d_i,d_j)
=
\frac{|H_i\cap H_j|}
{|H_i\cup H_j|}.
\]

---

## 19.2 Subspace similarity

使用 principal angles：

\[
\sigma_k
=
\cos\theta_k
\]

来自：

\[
U_i^\top U_j.
\]

定义：

\[
Sim(U_i,U_j)
=
\frac1K
\sum_{k=1}^K
\cos^2\theta_k.
\]

分别计算：

\[
Sim(U_i^D,U_j^D)
\]

和：

\[
Sim(U_i^C,U_j^C).
\]

关键比较：

\[
Sim_C
\quad vs\quad
Sim_D.
\]

---

# 20. 真正关键的是 Cross-Domain Intervention Transfer

几何相似并不能自动证明 functional generalization。

因此我们做：

在 source domain：

\[
d_s
\]

学习：

\[
U_{d_s}^C.
\]

把它 intervention 到：

\[
d_t.
\]

定义：

\[
T_{s\rightarrow t}^C
=
\Delta M
(U_{d_s}^C
\rightarrow d_t).
\]

最终构建：

\[
D\times D
\]

causal transfer matrix。

然后和 H-Neuron transfer matrix 对比。

理想结果：

\[
T^C_{\mathrm{cross}}
>
T^H_{\mathrm{cross}}.
\]

这比仅仅比较：

\[
\text{subspace cosine similarity}
\]

要强得多。

---

# 21. Shared Causal H-Subspace

如果发现：

\[
U_d^C
\]

之间具有明显共同方向，可以进一步构建：

\[
\Sigma_{\mathrm{shared}}^C
=
\sum_d
\pi_d\Sigma_d^C.
\]

取：

\[
U_{\mathrm{shared}}^C
=
\operatorname{TopEig}
(\Sigma_{\mathrm{shared}}^C).
\]

模型：

\[
h_d
=
U_{\mathrm{shared}}^Cz_s
+
U_d^Cz_d
+
\epsilon.
\]

其中：

\[
z_s
\]

表示 domain-general causal hallucination component，

\[
z_d
\]

表示 domain-specific component。

这部分如果实验成立，会成为 ICML 版论文的重要提升。

---

# 22. Stage V：Minimal H-Circuit（增强版）

这一部分不是论文成立的必要条件。

如果前三阶段结果很强，再增加。

我们不重新设计类似 CausalGaze/LANCET 的全局 causal graph，而是问：

> **哪些 attention/MLP components 支持已经找到的 causal subspace？**

因此从：

\[
U_l^C
\]

向上游执行 path patching。

节点包括：

\[
\{\text{Attention Heads},
\text{MLP Outputs},
\text{Residual States}\}.
\]

对于 edge：

\[
e:i\rightarrow j,
\]

定义 path-specific recovery：

\[
C_e
=
NCR(
do(e^H\leftarrow e^F)
).
\]

然后寻找最小 edge set：

\[
E^*
=
\arg\min_E|E|
\]

subject to：

\[
R(E)
\ge
\tau
R(E_{\mathrm{all}}),
\]

例如：

\[
\tau=0.9.
\]

最终得到：

\[
\boxed{
\text{Minimal Causal Support Circuit}.
}
\]

这里不把“circuit discovery”作为核心 novelty，而作为：

> 对 causal subspace 的 mechanistic explanation。

这样能够避免与 LANCET 正面对撞。

---

# 23. Dataset Design

论文不要一开始铺六七个领域。

建议首先聚焦 **factual hallucination**。

## Main factual datasets

### TriviaQA

必须有。

因为：

- H-Neurons 原始 setting；
- HARP；
- cross-domain H-Neuron；

都使用/涉及 TriviaQA。

是最重要 anchor benchmark。

### Natural Questions

用于验证不是 TriviaQA-specific。

### PopQA

特别适合研究 long-tail factual knowledge。

它能检查：

\[
\text{knowledge rarity}
\]

是否改变：

\[
D-C
\]

alignment。

### BioASQ

作为真正 domain shift：

\[
\text{General QA}
\rightarrow
\text{Biomedical}.
\]

---

# 24. Optional hallucination regime

后期建议额外加入一种：

### Citation Hallucination

原因不是增加 benchmark 数量，而是已有研究报告这里的 field-specific neurons 具有明显 intervention effect。

这给我们一个极其有价值的 comparison：

\[
\text{General factual hallucination}
\]

vs

\[
\text{Citation hallucination}.
\]

可能最后得到：

> Predictive–causal alignment depends strongly on hallucination type.

这比简单说：

> H-Neurons 全都不是 causal

更加科学。

---

# 25. Model Setup

开发阶段：

\[
\boxed{\text{Qwen2.5-3B-Instruct}}
\]

理由：

- cross-domain H-Neuron 工作使用；
- activation extraction 成本较低；
- 方便快速验证 hypothesis。

正式论文：

### Qwen2.5-3B-Instruct

主开发模型。

### Mistral-7B-Instruct-v0.3

验证 architecture transfer。

### Llama-3.1-8B-Instruct

增加主流 architecture。

cross-domain H-Neuron 工作本身已经在这些 3B–8B 规模模型上建立了较完整的 reference results。

这个规模也比 32B/70B 更适合大量 activation patching。

---

# 26. Baselines

需要分成三个层级。

## Diagnostic baselines

- H-Neurons
- Full CETT probe
- Hidden-state linear probe
- HARP
- Random neurons

---

## Neuron-level intervention

- Random neurons
- Top H-Neurons
- Gradient-ranked neurons
- Activation-magnitude neurons
- Our Causal Neurons

---

## Representation/pathway intervention

- PCA hallucination direction
- Probe direction
- Random subspace
- SPACE
- LANCET
- Our Causal H-Subspace

如果某个 baseline 没有合适公开代码，可以只在与其 setting 可公平复现时比较，不建议硬搬。

---

# 27. Main Evaluation Metrics

## Detection

\[
AUROC,
\quad
AUPRC.
\]

Detection 不是主要 contribution，但用于确认 diagnostic signal。

---

## Diagnostic–Causal Alignment

### Spearman correlation

\[
\rho(D,C).
\]

### Kendall's \(\tau\)

\[
\tau(D,C).
\]

### Top-K overlap

\[
J_K(D,C).
\]

### Causal Precision@K

\[
CP@K
=
\frac{
|\operatorname{TopK}(D)
\cap
\{C_j>\delta\}|
}{
K
}.
\]

这个指标特别直观：

> H-Neuron ranking 里面到底多少 neuron 真正具有 causal effect？

---

# 28. Causal Metrics

## Normalized Causal Recovery

\[
NCR.
\]

## Necessity

抑制/替换 causal representation 后：

\[
\Delta M<0
\]

是否显著。

## Sufficiency

把 hallucinated representation 注入 factual run：

\[
F\rightarrow H
\]

是否能够诱发 hallucination preference。

## Bidirectional Consistency

定义：

\[
BC_j
=
\min
(
R_j^{H\rightarrow F},
R_j^{F\rightarrow H}
).
\]

高：

\[
BC
\]

表示两个方向都有稳定 causal effect。

---

# 29. Utility Preservation

任何 intervention 都必须避免：

> simply breaking the model。

因此报告：

### Correct-answer retention

原来正确的样本：

\[
Acc_{\mathrm{before}}
\]

vs：

\[
Acc_{\mathrm{after}}.
\]

### General capability

选择少量通用任务检查：

\[
\Delta Utility.
\]

### KL drift

\[
D_{KL}
(
p_{\mathrm{base}}
\|
p_{\mathrm{intervened}}
).
\]

我们真正希望最大化：

\[
\boxed{
\text{Hallucination Reduction}
/
\text{Utility Damage}.
}
\]

---

# 30. Main Experimental Tables

## Table 1：Reproduce H-Neurons

| Model | Dataset | AUROC | #H-Neurons | %Neurons |
|---|---|---:|---:|---:|

证明 baseline 正确。

---

## Table 2：Predictive ≠ Causal

| Model | Dataset | Spearman \(D,C\) | Top-50 overlap | Top-100 overlap | CP@100 |
|---|---|---:|---:|---:|---:|

这是第一张核心表。

---

## Table 3：Intervention Comparison

| Method | Hallucination ↓ | NCR ↑ | Utility Δ | KL ↓ |
|---|---:|---:|---:|---:|
| Random | | | | |
| H-Neurons | | | | |
| Gradient Neurons | | | | |
| PCA Subspace | | | | |
| Causal Neurons | | | | |
| **Causal H-Subspace** | | | | |

---

## Table 4：Cross-Domain Transfer

| Source → Target | H-Neuron | Diagnostic Subspace | Causal Subspace |
|---|---:|---:|---:|

重点不是 AUROC，而是：

\[
NCR_{\mathrm{cross}}.
\]

---

# 31. 最关键的 Figures

## Figure 1：Conceptual Motivation

左边：

\[
\text{Prompt}
\rightarrow
\boxed{\text{H-Neuron}}
\rightarrow
\text{Hallucination}
\]

标：

> Existing implicit assumption

中间打：

\[
?
\]

右边：

\[
\text{Prompt}
\rightarrow
\boxed{\text{Causal Representation}}
\rightarrow
\text{Hallucinated Output}
\]

同时：

\[
\text{Causal/Latent State}
\rightarrow
\boxed{\text{Diagnostic H-Neurons}}.
\]

Caption：

> **A neuron can reveal that a model is hallucinating without being responsible for the hallucination.**

---

## Figure 2：Diagnostic vs Causal Scatter

\[
x=D_j,
\qquad
y=C_j.
\]

这是最关键的 evidence figure。

---

## Figure 3：Layer × Token Causal Map

横轴：

\[
\text{token position}
\]

纵轴：

\[
\text{layer}.
\]

颜色：

\[
NCR.
\]

展示 hallucination causal information 在哪里出现、传播。

---

## Figure 4：Neuron vs Subspace Generalization

展示：

\[
\text{Neuron Jaccard}
\]

很低，

但：

\[
\text{Causal Subspace Similarity}
\]

以及 cross-domain intervention transfer 更高。

如果能得到这张图，论文会非常漂亮。

---

# 32. Ablation Study

必须做：

### Causal subspace rank

\[
k\in
\{1,2,4,8,16,32,64\}.
\]

验证 hallucination mechanism 是否 low-dimensional。

### Intervention strength

\[
\eta
\in
\{0.25,0.5,1.0,1.5,2.0\}.
\]

### Top-K H-Neurons

\[
K
\in
\{10,50,100,500\}.
\]

### One-way vs Bidirectional causal score

只做：

\[
H\rightarrow F
\]

与双向：

\[
H\leftrightarrow F
\]

比较。

### Controlled pair vs unmatched pair

验证 causal localization 是否依赖 pair quality。

### Without intervention weighting

\[
\Sigma_{\mathrm{PCA}}
\]

vs：

\[
\Sigma_C.
\]

这是证明 Causal H-Subspace 设计必要性的关键 ablation。

---

# 33. 最小“生死实验”

在任何复杂方法开发之前，只做：

\[
\boxed{
\text{Qwen2.5-3B}
+
\text{TriviaQA}
}
\]

。

### Step 1

直接复现 H-Neurons：

\[
D_j.
\]

确认 AUROC 明显高于 random。

### Step 2

构造约：

\[
200\sim500
\]

个高质量 controlled factual/hallucination pairs。

### Step 3

选择：

\[
\text{Top-100 H-Neurons}
\]

和：

\[
100
\]

个 matched random neurons。

### Step 4

进行 bidirectional intervention，获得：

\[
C_j.
\]

### Step 5

计算：

\[
\rho(D,C)
\]

和：

\[
J_{100}.
\]

---

# 34. Go / No-Go Criteria

## 强烈继续

如果：

\[
\rho(D,C)<0.3
\]

同时：

\[
J_{100}
\]

明显较低，

且存在不少：

\[
D\downarrow,\ C\uparrow
\]

components。

说明核心 hypothesis 成立。

---

## 仍然值得继续

如果：

\[
0.3<\rho<0.6,
\]

但发现：

\[
\text{Causal Subspace}
\gg
\text{H-Neuron intervention},
\]

也依然很有价值。

故事变成：

> neuron-level predictivity captures part, but not all, of the causal computation.

---

## 风险较高

如果：

\[
\rho>0.8
\]

且：

\[
TopK(D)\approx TopK(C),
\]

那么：

> predictive H-Neurons 与 causal neurons 基本一致。

此时原始“diagnostic–causal mismatch”故事会弱。

但仍可转向：

\[
\boxed{
\text{When are H-Neurons causal?}
}
\]

比较：

- General factual QA；
- Long-tail factual QA；
- Biomedical；
- Citation hallucination；

形成 hallucination-type dependent causal taxonomy。

---

# 35. 本项目最大的 scientific risk

不是算力。

而是：

\[
D
\]

和：

\[
C
\]

最后太一致。

所以千万不要先花大量时间构建 circuit。

研究顺序必须是：

\[
\boxed{
D-C\ Matching Test
}
\]

↓

如果 mismatch 存在：

\[
\boxed{
Causal Subspace
}
\]

↓

如果 subspace 很强：

\[
\boxed{
Cross-domain
}
\]

↓

最后再：

\[
\boxed{
H-Circuit.
}
\]

---

# 36. 第二个风险：activation patching 本身造成 artefact

因此需要：

- bidirectional intervention；
- random patch；
- matched patch；
- projection-norm matching；
- multiple donors；
- clean → corrupted；
- corrupted → clean；
- zero-ablation 作为辅助而非主要 evidence。

尤其不能：

> patch 一次 output 变了，就说找到了 causal mechanism。

真正有说服力的是：

\[
\text{same component}
\]

在：

\[
H\rightarrow F
\]

和：

\[
F\rightarrow H
\]

两个方向都产生符合预期的改变。

---

# 37. 第三个风险：把 classification errors 全部叫 hallucination

cross-domain H-Neuron 工作中的部分领域使用 legal classification、financial sentiment、moral judgment、code vulnerability 等任务；这对于测试 domain specificity 很有价值，但严格来说：

\[
\text{wrong prediction}
\]

并不总等于：

\[
\text{hallucination}.
\]

因此我们的主论文最好坚持：

\[
\boxed{
\text{verifiable factual generation}.
}
\]

也就是：

> 模型生成了一个明确、可验证、与 gold fact 冲突的 factual claim。

这样定义更干净。

---

# 38. 与 H-Neurons 的核心区别

H-Neurons：

\[
\text{Activation}
\rightarrow
\text{Hallucination Label}.
\]

我们：

\[
\text{Activation}
\rightarrow
\begin{cases}
\text{Diagnostic Information}\\
\text{Causal Effect}
\end{cases}
\]

并显式研究：

\[
\boxed{
D\neq C?
}
\]

进一步：

\[
\text{Neuron}
\rightarrow
\text{Causal Geometry}
\rightarrow
\text{Causal Computation}.
\]

---

# 39. 预期 Contributions

如果结果理想，论文可以写成四项贡献。

### Contribution 1

首次系统区分 hallucination-associated internal units 的：

\[
\textbf{diagnostic importance}
\]

与：

\[
\textbf{causal responsibility}.
\]

---

### Contribution 2

提出 bidirectional counterfactual intervention framework：

\[
F\leftrightarrow H,
\]

以同时测量 necessity 与 sufficiency，并揭示 H-Neuron predictivity 与 causal effect 之间的关系。

---

### Contribution 3

发现 hallucination 的 causal information 相比 sparse neuron coordinates 更可能具有 distributed / low-rank representation，并提出：

\[
\textbf{Intervention-Weighted Causal H-Subspace}.
\]

---

### Contribution 4

揭示：

\[
\text{neuron-level domain specificity}
\]

与：

\[
\text{causal representation-level generality}
\]

是否能够共存，从而重新解释已有 H-Neuron cross-domain failure。

---

# 40. 如果实验最理想，摘要里的核心结果应该长这样

未来论文可以得到类似：

> Although a sparse set of neurons accurately predicts hallucinations, their diagnostic importance is only weakly correlated with their interventional causal effect. We find that hallucination causality is distributed across low-dimensional representations that are substantially more stable across domains than individual H-Neuron identities. Intervening on these causal representations reverses hallucinated predictions while preserving unrelated model capabilities.

最有记忆点的一句话：

\[
\boxed{
\text{H-Neurons tell us when a model hallucinates,
but not necessarily why.}
}
\]

---

# 41. ACL 与 ICML 的两种版本

## ACL Main 版本

重点：

- hallucination；
- H-Neurons；
- mechanistic interpretation；
- counterfactual intervention；
- factual QA；
- domain generalization。

核心 claim：

\[
\boxed{
\text{Predictive H-Neurons are not equivalent to causal hallucination mechanisms.}
}
\]

这是最稳妥版本。

---

## ICML 版本

必须进一步把方法抽象成：

\[
\boxed{
\text{Diagnostic–Causal Representation Decomposition}
}
\]

不再只服务 hallucination。

形式化研究：

对于 representation：

\[
Z,
\]

区分：

\[
Z_D=
\text{information predictive of }Y
\]

和：

\[
Z_C=
\text{information causally used for }Y.
\]

然后 hallucination 只是一个重要 testbed。

如果同时获得：

\[
\text{cross-domain shared causal geometry}
\]

以及理论/算法上的 decomposition objective，那么更接近 ICML。

---

# 42. 当前推荐路线

我建议第一阶段**不要把项目直接命名成 H-Circuit**。

因为：

- CausalGaze 已有 causal graph；
- LANCET 已有 pathway；
- SPACE 已有 activation subspace；
- HARP 已有 hallucination subspace。

当前最安全、最有辨识度的定位是：

\[
\boxed{
\textbf{Beyond H-Neurons:
Diagnostic–Causal Decomposition}
}
\]

论文核心：

\[
\textbf{Predictivity vs Causality}.
\]

然后：

\[
\text{Causal H-Subspace}
\]

是核心方法发现。

而：

\[
\text{H-Circuit}
\]

作为后期 mechanism analysis。

这样整篇论文不会落入：

> “别人已经做过 subspace/circuit，你只是换一种算法。”

的问题。

---

# 43. 最终项目结构

整个项目可以概括为：

\[
\boxed{
\text{H-Neurons}
}
\]

### What predicts hallucination?

↓

\[
\boxed{
\text{Diagnostic–Causal Decomposition}
}
\]

### Does prediction imply causation?

↓

\[
\boxed{
\text{Causal H-Subspace}
}
\]

### Where is the actual causal information?

↓

\[
\boxed{
\text{Cross-Domain Causal Geometry}
}
\]

### What generalizes?

↓

\[
\boxed{
\text{Minimal H-Circuit}
}
\]

### How is the causal representation computed?

---

# 44. 最重要的判断

这篇工作的价值不应该建立在：

> “我们的 hallucination rate 比 H-Neurons 低 3%。”

而应该建立在一个新的 scientific finding 上：

\[
\boxed{
\text{Hallucination-associated neurons should not automatically
be interpreted as hallucination-generating neurons.}
}
\]

然后我们给出一个正式 framework，把：

\[
\text{association},
\quad
\text{prediction},
\quad
\text{causal intervention}
\]

区分开。

如果最终还能发现：

\[
\boxed{
\text{Neuron identities are domain-specific,
while causal hallucination representations exhibit
greater cross-domain stability,}
}
\]

那么论文就会从：

> “H-Neurons follow-up”

升级成：

> **关于 LLM hallucination internal mechanism 的更一般结论。**

这是我认为真正具备 ACL Main，甚至进一步冲击 ICML 潜力的版本。