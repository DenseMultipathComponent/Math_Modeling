# 问题二模型公式与推导

本文件与 `problem2_solution.py` 一致。问题二读取附件 B，并通过问题一导出的模型、质量评分和配比表引入附件 A 的信息。这里提出的是具有明确跨源假设的广义标度律；不把互不相同的 Loss 口径直接拼成一个回归样本。

## 1 符号与单位

| 符号 | 含义 | 代码中的对应 |
| --- | --- | --- |
| $N$ | 实际参数个数 | $10^9\times$ `N_params_B` |
| $D$ | 实际训练 token 数 | $10^9\times$ `D_tokens_B` |
| $n=N/10^9$ | 以十亿参数为单位的规模 | `N_params_B` |
| $d=D/10^{11}$ | 以一千亿 tokens 为单位的数据量 | `D_tokens_B / 100` |
| $Q\in[0,1]$ | 当前训练语料的质量 | 问题一评分或 B6 的 `Q_score` |
| $\mathbf p\in\mathcal S^{16}$ | 17 个训练领域的配比 | 问题一导出的配比列 |
| $\bar Q(\mathbf p)$ | 该配方在问题一语料条件下的常规质量 | `p @ bridge.qmid` |
| $f_A(\mathbf p)$ | 问题一二次 ILR 岭模型预测的 13 域平均 Loss | `bridge.predict(p)` |
| $\mathbf p_0$ | 512 个训练配方闭合后的算术均值 | `bridge.p0` |
| $Q_0=\bar Q(\mathbf p_0)$ | 参考质量 | 约 0.610914 |
| $\gamma$ | B6 识别的质量响应系数 | `gamma` |
| $\kappa$ | A 与 B 质量分数的相对尺度 | 主情景为 1；敏感性取 0.5、1、2 |
| $\lambda$ | 配比效应的跨源迁移强度 | 主情景为 1；敏感性取 0、0.5、1、2 |

约束为 $p_j\ge0$、$\sum_{j=1}^{17}p_j=1$。CSV 的配比存在舍入误差，问题二先闭合，再调用问题一的 ILR 变换。$A,B$ 的数值依赖于上述单位；不能直接把原始参数个数代入本文件估计的公式。

## 2 数据角色与可检验假设

| 数据 | 用途 | 证据性质 |
| --- | --- | --- |
| B1，1176 行 | 估计 $E,A,B,\alpha,\beta$；按规模和时间留出验证 | 按附件标注为真实训练日志 |
| B2，1029 行 | 原参数直接迁移检验 | 半合成，不能视为真实族外验证 |
| B3，4000 行 | 检验轨迹形状的一致性 | 插值，自 B1 派生 |
| B4，57 行；B5，44 行 | 跨族、文献数据的直接迁移检验 | 来源间 Loss 口径未完全统一 |
| B6，360 行 | 估计质量响应 $\gamma$ | 半合成 |
| B7，450 行 | 去除与 B6 重合的 360 行后验证 | 90 个新增质量水平，仍为半合成 |
| B8，1704 行 | 质量方向冲突、校准区与外推区检验 | 半合成，其中 720 行标为外推 |
| B9，132 行 | 大模型 $N,D$ 情景范围 | 元数据；4 行不满足正数与完整性要求 |
| B10，128 行 | 大规模估算值的一致性比较 | 估算 Loss，不能作为独立验证 |
| B11、B12 | 数据来源辅助记录 | 元数据，不进入 Loss 回归 |

采用以下假设，同时明确它们的可识别程度：

1. **规模形状假设**：同一参考评估口径下，规模影响可用双幂律刻画。B1 的按规模留出和后期检查点留出用于检验。
2. **质量迁移假设**：B6 中“质量提高使超额 Loss 按比例减少”的规律可用于参考模型。B6 允许独立的 $E_6,A_6,B_6$；不要求其绝对 Loss 与 B1 相同。共享指数与自由指数模型以同一分组验证比较。
3. **配比迁移假设**：问题一的相对配比响应可在目标模型中保留部分形状，用 $\lambda$ 表示强度。只验证排序和敏感性，不声称已经识别精确迁移倍数。
4. **质量尺度假设**：问题一的综合评分与 B6 的 `Q_score` 均为越大越好，但相同的 0.1 增量并不自动等价。$\kappa=1$ 是情景假设，代码另做尺度敏感性分析。
5. **单独改善质量的含义**：固定配比 $\mathbf p$ 时，可以通过清洗、筛选等方式使 $Q$ 偏离其常规值 $\bar Q(\mathbf p)$。改变配比时产生的原有质量差异已包含在 $f_A$ 中，额外质量项仅刻画这种偏离。

第五条避免把问题一配比模型中已经包含的质量差异重复加一次。由于没有同时改变 $N,D,Q,\mathbf p$ 的真实联合实验，$\kappa,\lambda$ 不能从现有附件唯一识别；本方案因此输出一个带敏感性参数的模型族。

## 3 从问题一构造质量与配比接口

### 3.1 部分映射的配方质量

令 $M$ 为 6 个有直接或近直接映射的配方域，其余 11 域没有直接质量观测。问题一输出领域质量 $q_j$，未知域的中点使用问题一全局加权质量均值 $q_G$：

$$
\bar q_j=\begin{cases}q_j,&j\in M,\\q_G,&j\notin M.\end{cases}
\qquad
\bar Q(\mathbf p)=\sum_{j=1}^{17}p_j\bar q_j.
$$

令 $c(\mathbf p)=\sum_{j\in M}p_j$，$K(\mathbf p)=\sum_{j\in M}p_jq_j$。假设未知域质量落在已观测领域质量范围内，则

$$
Q^-(\mathbf p)=K(\mathbf p)+(1-c(\mathbf p))q_{\min},
\qquad
Q^+(\mathbf p)=K(\mathbf p)+(1-c(\mathbf p))q_{\max}.
$$

这是**带范围假设的映射区间**，不是严格无假设界，也不是 95% 置信区间。若不接受该范围假设，应把未知域范围改成 $[0,1]$，区间会明显扩大。

### 3.2 配比响应

直接复用问题一的预测器，不重新读取 A 原始数据，也不在问题二重训配比模型：

$$
f_A(\mathbf p)=\frac1{13}\sum_{r=1}^{13}\widehat L_{A,r}(\mathbf p),
\qquad
h(\mathbf p)=\log\frac{f_A(\mathbf p)}{f_A(\mathbf p_0)}.
$$

因此 $h(\mathbf p_0)=0$。$h$ 是相对变化量，避免直接把 1M RegMix 模型的绝对 Loss 加到 Pythia 的 Loss 上。这里使用总 Loss 比值是建模选择，不等于识别了 RegMix 的不可约损失；迁移幅度的不确定性由 $\lambda$ 表达。

问题一的 ILR 二次模型为

$$
\mathbf z=H\log\widetilde{\mathbf p},\qquad
\widehat L_{A,r}=b_{0r}+\mathbf b_r^\top\mathbf z+\mathbf z^\top\Gamma_r\mathbf z.
$$

其中 $H\mathbf1=0$，$\widetilde{\mathbf p}$ 为加入 $10^{-4}$ 伪计数后闭合的配比。具体系数与特征标准化完全从问题一导出的 NPZ 文件恢复。

## 4 广义标度律

定义参数受限项、数据受限项和资源可改善部分：

$$
X=A n^{-\alpha},\qquad Y=B d^{-\beta},\qquad R=X+Y.
$$

构造修正因子

$$
G(Q,\mathbf p)=\exp\left\{
\gamma\kappa[\bar Q(\mathbf p)-Q]+\lambda h(\mathbf p)
\right\}.
$$

主模型为

$$
\boxed{
L(n,d,Q,\mathbf p)=E+
\left(A n^{-\alpha}+B d^{-\beta}\right)
\exp\left\{\gamma\kappa[\bar Q(\mathbf p)-Q]+\lambda h(\mathbf p)\right\}.
}
$$

其中 $E\ge0$，$A,B,\alpha,\beta>0$，$\gamma\ge0$，主情景 $\kappa=\lambda=1$。代码中的 `r(p)` 就是本文件的 $h(\mathbf p)$。

**参考条件退化**：当 $\mathbf p=\mathbf p_0$ 且 $Q=Q_0$ 时，$G=1$，退化为经典形式。

**常规配比变化**：沿 $Q=\bar Q(\mathbf p)$，

$$
L=E+R\left[\frac{f_A(\mathbf p)}{f_A(\mathbf p_0)}\right]^\lambda.
$$

因而质量差异不被重复计算；当 $\lambda>0$ 时，配方排序与问题一的宏平均预测一致。

**独立质量改善**：固定 $n,d,\mathbf p$，质量提高 $\Delta Q$ 时，

$$
\frac{L(Q+\Delta Q)-E}{L(Q)-E}=e^{-\gamma\kappa\Delta Q}.
$$

本模型不强制 $Q=1$ 时对所有配方退化为经典式。原因是 B1 并未观测到“完美质量”，且即使质量较高，领域分布仍会影响目标域 Loss。将基准锚定在可计算的参考条件，比把未知 B1 质量设为 1 更易解释。$Q_0$ 是归一化锚点，不能声称是实测的 Pythia 语料质量。

## 5 参数估计

### 5.1 B1 估计规模参数

设 $\theta=(E,A,B,\alpha,\beta)$，残差为

$$
e_i(\theta)=E+A n_i^{-\alpha}+B d_i^{-\beta}-L_i.
$$

为避免同一模型的后期密集检查点主导结果，将 $\log d$ 的全局范围分为 8 个等宽区间。若模型 $g$ 占有 $K_g$ 个非空区间，区间 $(g,b)$ 内有 $m_{gb}$ 个样本，原始权重为

$$
w_i^{raw}=\frac1{K_gm_{gb}},\qquad w_i=\frac{w_i^{raw}}{\overline{w^{raw}}}.
$$

使用与 `scipy.optimize.least_squares(loss="soft_l1")` 一致的目标：

$$
\widehat\theta=\arg\min_\theta
\sum_i\delta^2\left[\sqrt{1+\frac{w_i e_i(\theta)^2}{\delta^2}}-1\right],
\qquad\delta=0.1.
$$

使用多个初值，限制 $0\le E<\min_iL_i$、$10^{-8}\le A,B\le30$、$0.005\le\alpha,\beta\le2$。这些是数值求解边界，不是文献给定参数；应检查估计是否贴边。当前估计均不贴边。

### 5.2 B6 估计质量响应

固定 B1 的指数 $\widehat\alpha,\widehat\beta$，引入来源特定系数：

$$
L_{6,i}=E_6+
\left(A_6n_i^{-\widehat\alpha}+B_6d_i^{-\widehat\beta}\right)
\exp\{\gamma(Q_0-Q_i)\}+\varepsilon_i.
$$

用 soft-L1 目标估计 $E_6,A_6,B_6,\gamma$，取 $\delta=0.05$，约束 $E_6\in[0,20]$、$A_6,B_6\in[10^{-8},30]$、$\gamma\in[0,8]$。只有响应系数 $\gamma$ 迁移到主模型，不能把 $E_6,A_6,B_6$ 覆盖 B1 参数。

在 B6 上按完整 $(N,D)$ 组划分 5 折，比较：

- $M_0$：$\gamma=0$，没有质量项；
- $M_1$：共享 B1 指数的质量模型，作为主模型；
- $M_2$：$\alpha,\beta$ 也在 B6 上重新估计，作为跨源形状假设的检验。

同组不同质量水平全部进入同一折，避免只记住某一 $(N,D)$ 组合造成乐观评价。

### 5.3 本次参数

主情景下，估计为

$$
\widehat E=1.689482,\quad
\widehat A=0.354064,\quad
\widehat B=0.342016,\quad
\widehat\alpha=0.339921,\quad
\widehat\beta=0.279796,\quad
\widehat\gamma=0.410160.
$$

具体 95% 区间见 `problem2_outputs/final_run/parameter_estimates.csv`。这里 $A$ 对应 1B 参数，$B$ 对应 100B tokens；精确复现使用 JSON 中的未舍入数值。

## 6 边际效用与弹性

令 $\eta=\gamma\kappa$，以下偏导均固定其余变量。边际效用定义为“减少 Loss 的正收益”而非 Loss 本身的偏导：

$$
U_n=-\frac{\partial L}{\partial n}=\frac{\alpha XG}{n},\qquad
U_d=-\frac{\partial L}{\partial d}=\frac{\beta YG}{d},\qquad
U_Q=-\frac{\partial L}{\partial Q}=\eta RG.
$$

若数据量使用 CSV 的 $D_B=D/10^9=100d$，则 $U_{D_B}=\beta YG/D_B$。代码输出的参数收益单位是“每增加 1B 参数减少的 Loss”，数据收益单位是“每增加 1B tokens 减少的 Loss”。

以超额损失 $F=L-E=RG$ 为对象，定义正向改进弹性：

$$
\epsilon_n^F=-\frac{\partial\log F}{\partial\log n}
=\frac{\alpha X}{X+Y},\quad
\epsilon_d^F=\frac{\beta Y}{X+Y},\quad
\epsilon_Q^F=\eta Q.
$$

对总 Loss 的弹性则为

$$
\epsilon_n^L=\frac{\alpha XG}{L},\qquad
\epsilon_d^L=\frac{\beta YG}{L},\qquad
\epsilon_Q^L=\frac{\eta QRG}{L}.
$$

$Q$ 是有界综合评分，质量半弹性 $-\partial\log F/\partial Q=\eta$ 和“提高 0.1”的比较通常比百分比质量变化更直观。不能把对总 Loss 的弹性和对超额 Loss 的弹性混报。

二阶偏导为

$$
\frac{\partial^2 L}{\partial n^2}=\frac{\alpha(\alpha+1)XG}{n^2}>0,\quad
\frac{\partial^2 L}{\partial d^2}=\frac{\beta(\beta+1)YG}{d^2}>0,\quad
\frac{\partial^2 L}{\partial Q^2}=\eta^2RG>0.
$$

因此边际降损收益递减。又有

$$
\frac{\partial^2 L}{\partial n\partial Q}=\frac{\eta\alpha XG}{n}>0,
\qquad
\frac{\partial^2 L}{\partial d\partial Q}=\frac{\eta\beta YG}{d}>0.
$$

在该模型的“降损边际收益”定义下，质量改善会降低继续扩大 $N$ 或 $D$ 的边际收益，表现为局部替代。固定质量与配比时 $L_{nd}=0$ 是本模型的可加结构假设，不能写成真实训练中两种资源永远没有交互。

## 7 单位成本下的比较

令 $c_n=\partial C/\partial n$ 表示每增加 1B 参数的边际成本，$c_Q=\partial C/\partial Q$ 表示提高一个质量单位的边际成本。单位成本降损收益为

$$
V_n=\frac{\alpha XG}{nc_n},\qquad
V_Q=\frac{\eta RG}{c_Q}.
$$

因此优先改善质量的条件是

$$
\boxed{\frac{c_Q}{c_n}<\frac{\eta n(X+Y)}{\alpha X}.}
$$

附件没有给出本问的真实货币成本，所以只能计算临界成本比，不能凭空宣布哪种方案“总是更省钱”。在 7B、300B tokens 参考点，右端约为 20.0718；其单位由“每一质量单位”和“每 1B 参数”的成本口径决定。

若固定总训练算力，扩容会同时减少可用数据量。由 $C=6ND=6\times10^{20}nd$，有 $d\propto1/n$，于是

$$
-\frac{dL}{dn}\bigg|_{C,Q,\mathbf p}=\frac{G}{n}(\alpha X-\beta Y).
$$

此时只有 $\alpha X>\beta Y$ 才适合继续增大参数；固定 $D$ 的结论不能直接套用到固定算力的情景。这为问题三留下接口，本问不另行设定训练、质量和注意力总成本。

## 8 质量提升与模型扩容的精确替代条件

### 8.1 相同起点下的两种改进

比较两种方案：

- 方案一：参数仍为 $n$，质量从 $Q$ 提高到 $Q+\Delta Q$；
- 方案二：质量仍为 $Q$，参数从 $n$ 提高到 $n_{eq}$。

两者固定同一 $d$ 与 $\mathbf p$。令 $t=e^{-\eta\Delta Q}$，等 Loss 条件为

$$
A n_{eq}^{-\alpha}+Y=t(X+Y).
$$

解得

$$
\boxed{
n_{eq}=\left[\frac{A}{t(X+Y)-Y}\right]^{1/\alpha},\qquad
\frac{n_{eq}}n=
\left[\frac{t(X+Y)-Y}{X}\right]^{-1/\alpha}.
}
$$

有限解存在的必要充分条件为

$$
\boxed{t(X+Y)>Y.}
$$

当 $\eta>0$ 时也可写为

$$
0<\Delta Q<\frac1\eta\log\left(1+\frac XY\right),
\qquad Q+\Delta Q\le1.
$$

等号对应 $n_{eq}\to\infty$；若分母为负，提高质量已使目标 Loss 低于“固定数据量、只增大参数”的极限。这种情况必须输出“无有限等效扩容解”，不能对负数开幂或硬设一个巨大参数量。

局部一阶等效关系为

$$
\boxed{\frac{d\log n_{eq}}{dQ}=\frac{\eta(X+Y)}{\alpha X}.}
$$

有限增量应使用精确式，而不是把一阶比例直接当作 0.1 增量的最终结果。

### 8.2 维持原有性能时允许缩小多少参数

若质量提高后只要求保持原 Loss，新的参数量 $n_{keep}$ 满足

$$
\frac{n_{keep}}n=
\left[\frac{e^{\eta\Delta Q}(X+Y)-Y}{X}\right]^{-1/\alpha}.
$$

这与上一节的问题不同；一个是“达到同样改进需要扩容多少”，另一个是“保持原水平可以缩容多少”，不能直接互换倍数。

### 8.3 当前附件的数值例子

取 $n=7$、$d=3$、$\mathbf p=\mathbf p_0$、$Q=0.610914$、$\Delta Q=0.1$、$\kappa=1$：

$$
L_{before}\approx2.123722,\qquad
L_{after}\approx2.106271,
$$

$$
n_{eq}\approx9.404464,\qquad
\frac{n_{eq}}n\approx1.343495.
$$

即在当前假设下约等价于增加 **34.35%** 参数。200 次分组 Bootstrap 给出的倍数区间约为 **[1.322870, 1.366724]**，对应参数量约 **[9.260090B, 9.567067B]**。该例的 300B tokens 略超出 B1 最大值 299.893B，属于边界附近的轻微外推。

该区间只覆盖给定模型与映射假设下的样本重采样变化，没有覆盖“B6 半合成机制是否真实”“A/B 质量尺度是否一致”等结构不确定性。尺度 $\kappa$ 的敏感性单独见 `quality_scale_sensitivity.csv`。

## 9 领域替代与互补

### 9.1 必须在单纯形内定义变化

不能在不补偿其他份额的情况下单独增加 $p_j$。单域边际分析采用

$$
v_{j,j}=1,\qquad v_{j,l}=-\frac{p_l}{1-p_j}\quad(l\ne j),
\qquad\sum_l v_{j,l}=0.
$$

将其他域按比例减少，定义方向导数 $D_{v_j}L$。代码输出的是每单位份额变化的导数；增加一个百分点的局部变化约为 $0.01D_{v_j}L$，前提是扰动仍可行且局部近似有效。

固定 $Q$ 时令

$$
a_v=\eta\bar{\mathbf q}^{\top}v+\lambda D_vh,
\qquad D_vL=(L-E)a_v.
$$

沿常规质量路径 $Q=\bar Q(\mathbf p)$ 时质量项抵消：

$$
D_vL=(L-E)\lambda D_vh.
$$

代码分别输出 `fixed_Q` 和 `with_Q` 两种口径，后者代表随配方使用其常规质量，不能混用。

### 9.2 混合导数与局部关系

对领域 $j,k$，先从其他 15 域构造共同供给池：

$$
u_l=\begin{cases}0,&l\in\{j,k\},\\
\dfrac{p_l}{1-p_j-p_k},&l\notin\{j,k\}.
\end{cases}
\qquad v_j=e_j-u,\quad v_k=e_k-u.
$$

在参考配比 $\mathbf p_0$ 固定这两个方向，定义

$$
I_{jk}=D_{v_j}D_{v_k}L.
$$

对固定质量或线性的常规质量路径，都有

$$
I_{jk}=(L-E)\left(a_{v_j}a_{v_k}+\lambda D_{v_j}D_{v_k}h\right),
$$

其中 $a_v$ 按对应路径定义。代码使用相同方向的四角中心差分：

$$
I_{jk}\approx\frac{
L(\mathbf p_0+\delta v_j+\delta v_k)-L(\mathbf p_0+\delta v_j-\delta v_k)
-L(\mathbf p_0-\delta v_j+\delta v_k)+L(\mathbf p_0-\delta v_j-\delta v_k)
}{4\delta^2},\qquad\delta=10^{-4}.
$$

- $I_{jk}<0$：增加一个域使另一个域的边际降损作用增强，称为该路径下的局部互补；
- $I_{jk}>0$：边际降损作用减弱，称为该路径下的局部替代。

这些关系依赖参考配方、质量路径、供给池和迁移强度。它们不是领域间的普适因果关系，也不等价于“两域占比越多越好”。尤其当两个方向本身都使 Loss 上升时，要结合一阶效应解释。

在局部等 Loss 曲线上，若两个方向都能降损，则

$$
\frac{ds_k}{ds_j}=-\frac{D_{v_j}L}{D_{v_k}L}
$$

给出份额之间的边际替代率。若只是从域 $k$ 转移到域 $j$，应直接使用方向 $e_j-e_k$，不能把两个不同供给池的单域边际效应随意相减。

## 10 检验、不确定性与不可识别部分

1. **B1 按模型规模留出**：8 次训练，每次完整留出一个参数规模的所有检查点，合并预测后计算 RMSE、MAE、$R^2$ 和 Spearman。
2. **B1 时间验证**：每个规模按 $D$ 排序，前 70% 拟合、后 30% 验证。B3 的 4000 个插值点只用于轨迹一致性比较。
3. **B4/B5**：使用 B1 原参数直接预测，整体与各模型族分别报告；不使用检验标签做隐含仿射校准。
4. **B6/B7**：按 $(N,D)$ 分组验证；B7 先按 $(N,D,Q)$ 与 B6 去重。新增 90 点属于新质量水平验证，不是新的模型规模实验。
5. **B8**：固定 $(N,D)$ 后计算 $Q$ 与 Loss 的斜率。150 组全部为正，与“质量越高越好”和 B6 的趋势冲突。保留原值，分别检验 `calibrated`、`extrapolated`，不擅自做 $Q\leftarrow1-Q$。允许负 $\gamma$ 的拟合仅是冲突诊断，不能进入主模型的质量收益结论。
6. **Bootstrap**：对 B1 重采样完整规模轨迹，对 B6 重采样完整 $(N,D)$ 组；每次重新拟合两阶段模型。B1 同一个规模若被抽中多次，以不同抽样簇计数，保留其重采样权重。
7. **映射敏感性**：取 $Q\in[Q^-,Q^+]$，固定中点定义的配方基准，计算条件预测区间。由于 $L_Q<0$，最低预测 Loss 对应 $Q^+$。
8. **迁移敏感性**：分别改变 $\kappa$、$\lambda$。沿 $Q=\bar Q(\mathbf p)$，任意正 $\lambda$ 的排序相同，因此良好的排序也不能识别迁移幅度。
9. **可用范围**：B1 的参数范围约为 0.070542B–11.965825B，数据量范围为 0.134B–299.893B tokens。B9/B10 所覆盖的更大模型属于外推情景；缺失的实际质量与配比不能被当作已观测。

此外，宏平均 Loss 与逐验证域排序是不同指标。问题一报告的 13 域平均 Spearman，不能替换本文件对 13 域平均 Loss 算出的单个 Spearman。

## 11 文献来源与本文新增部分

- Hoffmann et al. (2022). [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)：经典规模与数据量标度关系的研究背景。
- Liu et al. (2024; ICLR 2025). [RegMix: Data Mixture as Regression for Language Model Pre-training](https://arxiv.org/abs/2407.01492)：通过小模型配方回归研究配比迁移的背景。
- 本文件的质量偏离修正、配比响应桥接、具体求解边界和本地附件参数估计属于本方案的建模选择。上述文献不构成本文所有跨源假设均成立的证明。
