# 问题一模型公式

本文给出问题一的可复现数学定义。附件中的小字方法建议、预置参数和预置结论不作为模型依据。下标 ($i$) 表示文本，($j=1,\ldots,22$) 表示质量指标，($d$) 表示质量信号领域，($k=1,\ldots,17$) 表示配方领域，($r=1,\ldots,13$) 表示验证 Loss 领域，($s$) 表示模型规模。

## 1 质量指标标量化

对于**标量**字段直接记为 ($x_{ij}$)。**列表型**指标按其输出含义压缩：

1. 二分类 logits（`fluency_en`、`ad_en`）转成正类概率：

$$
x_{ij}=\frac{\exp(a_1)}{\exp(a_0)+\exp(a_1)}.
$$

2. 六级有序 logits（四个 `modernbert_*` 字段）转为归一化期望等级：

$$
x_{ij}=\sum_{c=0}^{5}\frac{c}{5}
\frac{\exp(a_c)}{\sum_{h=0}^{5}\exp(a_h)}.
$$

3. `qurater` 的四个维度视为并列质量侧面，分别经逻辑函数后取均值：

$$
x_{ij}=\frac{1}{4}\sum_{c=1}^{4}\frac{1}{1+\exp(-a_c)}.
$$

4. `fineweb_edu` 的单元素列表直接取唯一元素。

## 2 指标方向统一

以 A1 为**参考总体**，计算每个指标的 1%、50%、99% 分位数 ($q_{.01,j},q_{.50,j},q_{.99,j}$)，先将极端值缩尾到该区间。
效益型指标为

$$
S_{ij}=\frac{\operatorname{clip}(x_{ij})-q_{.01,j}}
{q_{.99,j}-q_{.01,j}}.
$$

成本型指标为

$$
S_{ij}=1-\frac{\operatorname{clip}(x_{ij})-q_{.01,j}}
{q_{.99,j}-q_{.01,j}}.
$$

长度、句数、数字比例、平均词长不宜预设为单调优劣，定义为目标型指标：

$$
S_{ij}=\begin{cases}
1-\dfrac{q_{.50,j}-x_{ij}}{q_{.50,j}-q_{.01,j}},&x_{ij}\le q_{.50,j},\\[6pt]
1-\dfrac{x_{ij}-q_{.50,j}}{q_{.99,j}-q_{.50,j}},&x_{ij}>q_{.50,j}.
\end{cases}
$$

结果截断至 ([0,1])，从而全部指标均为“**越高越好**”。**缺失值**先以同领域中位数填补，再以 A1 全局中位数兜底；同时保留覆盖率，避免把填补后的记录误认为信息完整。

## 3 受限 CRITIC 权重

在 A1 上计算标准差 ($\sigma_j$) 与相关系数 ($\rho_{jh}$)，CRITIC 信息量为

$$
C_j=\sigma_j\sum_{h=1}^{22}(1-|\rho_{jh}|),\qquad
w_j^{(c)}=\frac{C_j}{\sum_{h=1}^{22} C_h}.
$$

为避免单个指标因噪声获得过大权重，将其与**等权重混合**并**限制范围**：

$$
\widetilde w_j=\frac12\frac1{22}+\frac12w_j^{(c)},\qquad
w_j=\frac{\operatorname{clip}(\widetilde w_j,0.5/22,2/22)}
{\sum_{h=1}^{22}\operatorname{clip}(\widetilde w_h,0.5/22,2/22)}.
$$

## 4 冲突鲁棒综合质量分(Not fully understood)

**综合质量** ($Q_i$) 采用加权 Huber 位置估计：

$$
Q_i=\arg\min_{q\in[0,1]}\sum_{j=1}^{22}w_j
\rho_{1.345}\!\left(\frac{S_{ij}-q}{\widehat\sigma_i}\right),
$$

其中 ($\widehat\sigma_i=1.4826\operatorname{median}_j|S_{ij}-q|$)，Huber 损失为

$$
\rho_c(u)=\begin{cases}\tfrac12u^2,&|u|\le c,\\c|u|-\tfrac12c^2,&|u|>c.\end{cases}
$$

该估计在指标一致时接近加权均值，在少数指标异常时自动降低离群指标影响。

定义**加权分歧度**与**稳健极差**：

$$
D_i=\sqrt{\sum_{j=1}^{22}w_j(S_{ij}-Q_i)^2},\qquad
R_i=P_{90}(S_{i\cdot})-P_{10}(S_{i\cdot}).
$$

以 A1 的分歧度 95% 分位数 ($\tau_D$) 为数据驱动阈值。显著冲突定义为

$$
I_i=\mathbf 1(D_i>\tau_D\ \land\ R_i>0.60).
$$

质量置信度为

$$
Conf_i=Coverage_i\left(1-\min\left\{\frac{D_i}{0.50},1\right\}\right).
$$

指标 ($j$) **对冲突的总体贡献**可用

$$
A_j=\frac{\mathbb E_{i:I_i=1}[w_j|S_{ij}-Q_i|]}
{\sum_{h=1}^{22}\mathbb E_{i:I_i=1}[w_h|S_{ih}-Q_i|]}
$$

排序，用于解释冲突主要由哪些指标产生。

## 5 领域质量及扩展集检验

**领域**质量为领域内 Huber 得分的均值：

$$
Q_d=\frac1{n_d}\sum_{i:g(i)=d}Q_i,
$$

并报告均值标准误区间、冲突率和平均置信度。对 arxiv、github 分别比较 A1 抽样与 A2/A3 全量扩展：

- 均值差 ($\Delta_d=\bar Q_{d,extend}-\bar Q_{d,sample}$)；
- 标准化差异 Cohen's (d)；
- 两样本 Kolmogorov-Smirnov 统计量；
- 冲突率之差。

最终领域评分中，**arxiv、github 使用扩展集估计，其余领域使用 A1**；这样既满足全量要求，又避免把 A1 中可能与扩展集重合的记录重复计数。

## 6 单纯形上的领域配比模型

配比向量满足

$$
\mathbf p_i\in\mathcal S^{16}=\left\{\mathbf p:p_k\ge0,\ \sum_{k=1}^{17}p_k=1\right\}.
$$

**对零分量加入小伪计数** ($\epsilon=10^{-4}$) 后闭合：

$$
\widetilde p_{ik}=\frac{p_{ik}+\epsilon}{\sum_{k=1}^{17}(p_{ih}+\epsilon)}.
$$

令 ($H\in\mathbb R^{16\times17}$) 为 Helmert 正交对比矩阵，ILR 坐标为

$$
\mathbf z_i=H\left(\log\widetilde{\mathbf p}_i-
\frac1{17}\mathbf 1\mathbf 1^\top\log\widetilde{\mathbf p}_i\right).
$$

**对每个验证域** ($r$) 建立共享正则强度的二次岭回归：

$$
L_{ir}=\beta_{0r}+\boldsymbol\beta_r^\top\mathbf z_i
+\mathbf z_i^\top\Gamma_r\mathbf z_i+\varepsilon_{ir}.
$$

参数由

$$
\min_{\beta,\Gamma}\sum_{i,r}(L_{ir}-\widehat L_{ir})^2
+\lambda\left(\|\beta\|_2^2+\|\Gamma\|_F^2\right)
$$

估计，($\lambda$) 在对数网格上交叉选择。ILR 消除了“删掉一个配比分量”的任意性，二次项刻画领域组合效应。

## 7 验证、跨规模迁移与外推(Not fully understood)

在 A6/A7 的同规模检验上报告 RMSE、MAE、($R^2$) 和各 Loss 领域 Spearman 相关的均值。

A8/A9、A10/A11 与训练表的参数规模不同，绝对 Loss 水平必然变化。因此跨规模的主指标是排序保持：

$$
\bar\rho_s=\frac1{13}\sum_{r=1}^{13}
\operatorname{Spearman}(L_{sr},\widehat L_{1M,r}).
$$

另可拟合 ($L_{sr}=a_{sr}+b_{sr}\widehat L_{1M,r}+e$) 给出描述性校准 RMSE，但必须明确该校准使用了检验标签，不属于独立预测成绩。

A12--A15 的 Loss 是外推值，且配比是训练配比子集，只用于比较排序、效应方向和结论稳定性，不作为新的真实实验验证。

## 8 边际效应和组合效应(Not fully understood)

将领域 (k) 的份额增加 ($\delta=0.01$)，其余份额按比例缩减，记该单纯形内扰动为 ($T_k(\mathbf p,\delta)$)。平均边际效应为

$$
ME_k=\mathbb E_{\mathbf p}\left[
\frac{\bar L(T_k(\mathbf p,\delta))-\bar L(\mathbf p)}{\delta}
\right],\qquad \bar L=\frac1{13}\sum_rL_r.
$$

($ME_k<0$) 表示在当前配方分布附近增加该领域通常降低平均 Loss。

领域 ($j,k$) 的二阶组合效应为

$$
INT_{jk}=\frac{
\bar L(T_{jk})-\bar L(T_j)-\bar L(T_k)+\bar L(\mathbf p)
}{\delta^2}.
$$

($INT_{jk}<0$) 表示协同降低 Loss，($INT_{jk}>0$) 表示组合效果弱于可加预期。

## 9 质量信息与配比模型的衔接

A16 只给出 6 个配方域到质量域的直接或近直接映射，其余 11 域没有可观测质量分。对配方 ($\mathbf p$)，**已映射质量**和与**覆盖质量**为

$$
K(\mathbf p)=\sum_{k\in M}p_kQ_{m(k)},\qquad
c(\mathbf p)=\sum_{k\in M}p_k.
$$

**未映射部分**用已观测领域质量范围 ($[Q_{min},Q_{max}]$) 做部分识别：

$$
Q_{mix}\in
\left[K+(1-c)Q_{min},\ K+(1-c)Q_{max}\right].
$$

**中点估计**用全局加权均值 ($\bar Q$)：

$$
Q_{mix}^{mid}=K+(1-c)\bar Q.
$$

由于 ($Q_{mix}^{mid}$) 本身是 ($\mathbf p$) 的线性函数，在仅有 A4--A15 时把它与 17 域配比同时放入回归会产生**不可辨识或强共线性**。因此本方案将其作为解释和敏感性指标，暂不声称存在独立“质量效应”。
