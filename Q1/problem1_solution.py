#!/usr/bin/env python3
"""2026 中国研究生数学建模竞赛 F 题问题一分析脚本。

本脚本只依据正式题面要求和实际附件字段建模，不采用附件中以小字出现的
预置方法、预置参数或预置结论。默认读取 A1--A16，其中 A1--A3 使用全部记录。

主要输出：
  quality_record_scores.csv.gz        全量文本质量、冲突与置信度
  quality_domain_summary.csv          分来源、分领域质量汇总
  quality_extension_comparison.csv    A1 与 A2/A3 的领域对照
  quality_metric_diagnostics.csv      指标方向、权重、缺失率与参考分位数
  conflict_metric_contributions.csv   冲突来源贡献
  mixture_validation_metrics.csv      A6--A11 验证结果
  mixture_domain_effects.csv           17 域边际效应
  mixture_interactions.csv             领域组合效应
  extrapolation_stability.csv          A12--A15 外推稳定性
  mixture_quality_projection.csv       映射质量的部分识别结果
  analysis_summary.json                关键参数和摘要
  figures/*.png                        论文可用图形
"""

from __future__ import annotations

import argparse
import json
import lzma
import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


QUALITY_FIELDS = [
    "fineweb_edu",
    "fluency_en",
    "modernbert_cleanliness",
    "modernbert_readability",
    "modernbert_reasoning",
    "modernbert_professionalism",
    "dsir_books",
    "dsir_wiki",
    "dsir_math",
    "qurater",
    "ad_en",
    "rps_doc_word_count",
    "rps_doc_num_sentences",
    "rps_doc_unigram_entropy",
    "rps_doc_frac_unique_words",
    "rps_doc_frac_no_alph_words",
    "rps_doc_frac_chars_top_2gram",
    "rps_doc_frac_chars_top_3gram",
    "rps_lines_uppercase_letter_fraction",
    "rps_lines_ending_with_terminal_punctution_mark",
    "rps_lines_numerical_chars_fraction",
    "rps_doc_mean_word_length",
]


def expit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    ex = np.exp(x[~positive])
    out[~positive] = ex / (1.0 + ex)
    return out


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    shifted = x - np.max(x)
    ex = np.exp(shifted)
    return ex / ex.sum()


def helmert_basis(dimension: int) -> np.ndarray:
    """返回 (D-1)×D 的正交 Helmert 对比矩阵。"""
    basis = np.zeros((dimension - 1, dimension), dtype=float)
    for k in range(1, dimension):
        scale = math.sqrt(k * (k + 1))
        basis[k - 1, :k] = 1.0 / scale
        basis[k - 1, k] = -k / scale
    return basis


def ks_2sample(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """两样本 KS 统计量及渐近 p 值；避免依赖 SciPy。"""
    a = np.sort(np.asarray(a, float))
    b = np.sort(np.asarray(b, float))
    combined = np.concatenate([a, b])
    d = float(np.max(np.abs(
        np.searchsorted(a, combined, side="right") / len(a)
        - np.searchsorted(b, combined, side="right") / len(b)
    )))
    en = math.sqrt(len(a) * len(b) / (len(a) + len(b)))
    lam = (en + 0.12 + 0.11 / max(en, 1e-12)) * d
    terms = [2.0 * ((-1) ** (k - 1)) * math.exp(-2.0 * k * k * lam * lam) for k in range(1, 101)]
    return d, float(np.clip(sum(terms), 0.0, 1.0))


def spearman_statistic(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank(method="average").to_numpy(float)
    rb = pd.Series(b).rank(method="average").to_numpy(float)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def polynomial_features_degree2(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, float)
    parts = [z]
    for j in range(z.shape[1]):
        parts.append(z[:, j:j + 1] * z[:, j:])
    return np.concatenate(parts, axis=1)


class QuadraticRidgeModel:
    """带二次特征的多输出岭回归，完全基于 NumPy。"""

    def __init__(self, alpha: float, x_mean: np.ndarray, x_std: np.ndarray,
                 y_mean: np.ndarray, coef: np.ndarray):
        self.alpha = float(alpha)
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.coef = coef

    def predict(self, z: np.ndarray) -> np.ndarray:
        x = polynomial_features_degree2(z)
        xs = (x - self.x_mean) / self.x_std
        return self.y_mean + xs @ self.coef


def _ridge_solution(xs: np.ndarray, yc: np.ndarray, alpha: float) -> np.ndarray:
    gram = xs.T @ xs
    values, vectors = np.linalg.eigh(gram)
    projected = vectors.T @ (xs.T @ yc)
    return vectors @ (projected / (values[:, None] + alpha))


def fit_quadratic_ridge_cv(z: np.ndarray, y: np.ndarray, seed: int = 20260924) -> QuadraticRidgeModel:
    x = polynomial_features_degree2(z)
    alphas = np.logspace(-4, 4, 25)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(x))
    folds = np.array_split(shuffled, 5)
    errors = np.zeros(len(alphas), dtype=float)
    for valid_idx in folds:
        train_idx = np.setdiff1d(shuffled, valid_idx, assume_unique=True)
        x_train, x_valid = x[train_idx], x[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        mean = x_train.mean(axis=0)
        std = x_train.std(axis=0, ddof=0)
        std[std < 1e-12] = 1.0
        xs = (x_train - mean) / std
        xv = (x_valid - mean) / std
        y_mean = y_train.mean(axis=0)
        yc = y_train - y_mean
        gram = xs.T @ xs
        values, vectors = np.linalg.eigh(gram)
        projected = vectors.T @ (xs.T @ yc)
        for ai, alpha in enumerate(alphas):
            coef = vectors @ (projected / (values[:, None] + alpha))
            pred = y_mean + xv @ coef
            errors[ai] += float(np.mean((y_valid - pred) ** 2))
    best_alpha = float(alphas[int(np.argmin(errors))])
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0, ddof=0)
    x_std[x_std < 1e-12] = 1.0
    xs = (x - x_mean) / x_std
    y_mean = y.mean(axis=0)
    coef = _ridge_solution(xs, y - y_mean, best_alpha)
    return QuadraticRidgeModel(best_alpha, x_mean, x_std, y_mean, coef)

# benefit：越大越好；cost：越小越好；target：处于经验主体区间更好。
# 长度、句数、数字比例与平均词长不宜武断设为单调关系，因此设为目标型。
DIRECTION = {
    "fineweb_edu": "benefit",
    "fluency_en": "benefit",
    "modernbert_cleanliness": "benefit",
    "modernbert_readability": "benefit",
    "modernbert_reasoning": "benefit",
    "modernbert_professionalism": "benefit",
    "dsir_books": "benefit",
    "dsir_wiki": "benefit",
    "dsir_math": "benefit",
    "qurater": "benefit",
    "ad_en": "cost",
    "rps_doc_word_count": "target",
    "rps_doc_num_sentences": "target",
    "rps_doc_unigram_entropy": "benefit",
    "rps_doc_frac_unique_words": "benefit",
    "rps_doc_frac_no_alph_words": "cost",
    "rps_doc_frac_chars_top_2gram": "cost",
    "rps_doc_frac_chars_top_3gram": "cost",
    "rps_lines_uppercase_letter_fraction": "cost",
    "rps_lines_ending_with_terminal_punctution_mark": "benefit",
    "rps_lines_numerical_chars_fraction": "target",
    "rps_doc_mean_word_length": "target",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F 题问题一：质量评价、冲突消解和配比建模")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="A_data_value 目录，例如 D:/.../real_attachments/A_data_value",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("problem1_outputs"))
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--critic-blend", type=float, default=0.5,
                        help="CRITIC 权重占比，0 为等权，1 为纯 CRITIC")
    parser.add_argument("--conflict-quantile", type=float, default=0.95,
                        help="A1 分歧度阈值分位数")
    parser.add_argument("--conflict-spread", type=float, default=0.60,
                        help="冲突判定的 P90-P10 最小极差")
    parser.add_argument("--pseudocount", type=float, default=1e-4,
                        help="配比零值替换伪计数")
    parser.add_argument("--effect-delta", type=float, default=0.01,
                        help="领域边际效应的份额扰动")
    parser.add_argument(
        "--max-records-per-file",
        type=int,
        default=None,
        help="仅用于快速调试；正式运行请省略，确保使用 A1--A3 全量记录",
    )
    parser.add_argument("--no-plots", action="store_true", help="不生成 PNG 图")
    return parser.parse_args()


def finite_number(value: object) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return np.nan
    return x if np.isfinite(x) else np.nan


def scalarize(field: str, value: object) -> float:
    """把 8 个列表型信号压缩为有明确含义的标量。"""
    if not isinstance(value, (list, tuple, np.ndarray)):
        return finite_number(value)
    arr = np.asarray([finite_number(v) for v in value], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    if arr.size == 1:
        return float(arr[0])
    if field in {"fluency_en", "ad_en"}:
        # 二分类 logits 的正类概率；ad_en 稍后按成本型反向。
        return float(softmax(arr)[-1])
    if field.startswith("modernbert_"):
        # 有序等级 logits 转为 [0,1] 上的期望等级。
        ranks = np.linspace(0.0, 1.0, arr.size)
        return float(np.dot(softmax(arr), ranks))
    if field == "qurater":
        # 四个独立质量维度先转为概率，再取平均，避免直接平均 logits。
        return float(np.mean(expit(arr)))
    return float(np.mean(arr))


def iter_jsonl_xz(path: Path, limit: int | None = None) -> Iterable[dict]:
    with lzma.open(path, "rt", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if limit is not None and i >= limit:
                break
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                warnings.warn(f"跳过无法解析的 JSON 行：{path.name}:{i + 1}")


def load_quality_data(root: Path, limit: int | None) -> pd.DataFrame:
    specs = [
        (root / "slimpajama_quality_signal_sample.jsonl.xz", "sample", None),
    ]
    ext_dir = root / "slimpajama_quality_extended"
    for path in sorted(ext_dir.glob("arxiv_*.jsonl.xz")):
        specs.append((path, "extended", "arxiv"))
    for path in sorted(ext_dir.glob("github_*.jsonl.xz")):
        specs.append((path, "extended", "github"))
    missing = [str(p) for p, _, _ in specs if not p.exists()]
    if missing:
        raise FileNotFoundError("缺少质量数据：" + ", ".join(missing))

    columns: dict[str, list] = defaultdict(list)
    for path, dataset, domain_override in specs:
        for obj in iter_jsonl_xz(path, limit):
            columns["id"].append(str(obj.get("id", "")))
            columns["dataset"].append(dataset)
            columns["source_file"].append(path.name)
            domain = domain_override or str(obj.get("_source_domain", "unknown")).lower()
            columns["domain"].append(domain)
            for field in QUALITY_FIELDS:
                columns[field].append(scalarize(field, obj.get(field)))
    frame = pd.DataFrame(columns)
    if frame.empty:
        raise ValueError("质量数据为空")
    return frame


def fit_reference_transform(sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for field in QUALITY_FIELDS:
        values = pd.to_numeric(sample[field], errors="coerce").to_numpy(float)
        finite = values[np.isfinite(values)]
        if finite.size < 10:
            raise ValueError(f"指标 {field} 的有效样本不足")
        q01, q50, q99 = np.quantile(finite, [0.01, 0.50, 0.99])
        if not q99 > q01:
            q99 = q01 + 1.0
        rows.append(
            {
                "metric": field,
                "direction": DIRECTION[field],
                "q01": q01,
                "q50": q50,
                "q99": q99,
                "sample_missing_rate": float(np.mean(~np.isfinite(values))),
            }
        )
    return pd.DataFrame(rows).set_index("metric")


def transform_desirability(raw: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)
    for field in QUALITY_FIELDS:
        x = pd.to_numeric(raw[field], errors="coerce").to_numpy(float)
        row = reference.loc[field]
        lo, med, hi = float(row.q01), float(row.q50), float(row.q99)
        clipped = np.clip(x, lo, hi)
        direction = row.direction
        if direction == "benefit":
            score = (clipped - lo) / (hi - lo)
        elif direction == "cost":
            score = (hi - clipped) / (hi - lo)
        else:
            left = np.maximum(med - lo, 1e-12)
            right = np.maximum(hi - med, 1e-12)
            score = np.where(clipped <= med, 1.0 - (med - clipped) / left,
                             1.0 - (clipped - med) / right)
        out[field] = np.clip(score, 0.0, 1.0)
    # 先按领域中位数填补，再用 A1 全局中位数兜底；同时另行保留覆盖率。
    domains = raw["domain"]
    for field in QUALITY_FIELDS:
        med_by_domain = out[field].groupby(domains).transform("median")
        out[field] = out[field].fillna(med_by_domain).fillna(out[field].median())
    return out


def critic_weights(sample_scores: pd.DataFrame, blend: float = 0.5) -> pd.Series:
    x = sample_scores[QUALITY_FIELDS].to_numpy(float)
    sd = np.nanstd(x, axis=0, ddof=1)
    corr = np.corrcoef(x, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    information = sd * np.sum(1.0 - np.abs(corr), axis=1)
    if not np.any(information > 0):
        information = np.ones(len(QUALITY_FIELDS))
    critic = information / information.sum()
    equal = np.full(len(QUALITY_FIELDS), 1.0 / len(QUALITY_FIELDS))
    blend = float(np.clip(blend, 0.0, 1.0))
    blended = (1.0 - blend) * equal + blend * critic
    lower, upper = 0.5 / len(QUALITY_FIELDS), 2.0 / len(QUALITY_FIELDS)
    blended = np.clip(blended, lower, upper)
    blended /= blended.sum()
    return pd.Series(blended, index=QUALITY_FIELDS, name="weight")


def huber_location(x: np.ndarray, weights: np.ndarray, iterations: int = 6) -> np.ndarray:
    base = np.broadcast_to(weights, x.shape)
    mu = np.sum(base * x, axis=1) / np.sum(base, axis=1)
    for _ in range(iterations):
        residual = x - mu[:, None]
        scale = 1.4826 * np.median(np.abs(residual), axis=1) + 1e-6
        ratio = np.abs(residual) / (1.345 * scale[:, None])
        robust = np.where(ratio <= 1.0, 1.0, 1.0 / np.maximum(ratio, 1e-12))
        effective = base * robust
        mu = np.sum(effective * x, axis=1) / np.sum(effective, axis=1)
    return np.clip(mu, 0.0, 1.0)


def score_quality(
    raw: pd.DataFrame,
    critic_blend: float = 0.5,
    conflict_quantile: float = 0.95,
    conflict_spread: float = 0.60,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_mask = raw["dataset"].eq("sample")
    reference = fit_reference_transform(raw.loc[sample_mask])
    scores = transform_desirability(raw, reference)
    weights = critic_weights(scores.loc[sample_mask], critic_blend)
    x = scores[QUALITY_FIELDS].to_numpy(float)
    w = weights.to_numpy(float)
    q = huber_location(x, w)
    disagreement = np.sqrt(np.sum(w[None, :] * (x - q[:, None]) ** 2, axis=1))
    spread = np.quantile(x, 0.90, axis=1) - np.quantile(x, 0.10, axis=1)
    sample_threshold = float(np.quantile(disagreement[sample_mask.to_numpy()], conflict_quantile))
    conflict = (disagreement > sample_threshold) & (spread > conflict_spread)
    coverage = raw[QUALITY_FIELDS].notna().mean(axis=1).to_numpy(float)
    confidence = coverage * (1.0 - np.clip(disagreement / 0.50, 0.0, 1.0))

    record = raw[["id", "dataset", "domain", "source_file"]].copy()
    record["Q"] = q
    record["disagreement"] = disagreement
    record["spread_p90_p10"] = spread
    record["conflict"] = conflict.astype(int)
    record["confidence"] = confidence
    record["coverage"] = coverage

    diagnostics = reference.copy()
    diagnostics["weight"] = weights
    diagnostics["all_missing_rate"] = raw[QUALITY_FIELDS].isna().mean()
    diagnostics["conflict_threshold_disagreement"] = sample_threshold
    diagnostics["critic_blend"] = critic_blend
    diagnostics["conflict_quantile"] = conflict_quantile
    diagnostics["conflict_spread_threshold"] = conflict_spread
    diagnostics = diagnostics.reset_index()

    conflicting_x = x[conflict]
    if conflicting_x.size:
        conflicting_q = q[conflict]
        contrib = np.mean(w[None, :] * np.abs(conflicting_x - conflicting_q[:, None]), axis=0)
        contrib = contrib / max(contrib.sum(), 1e-12)
    else:
        contrib = np.zeros(len(QUALITY_FIELDS))
    contributions = pd.DataFrame(
        {"metric": QUALITY_FIELDS, "weight": w, "conflict_contribution": contrib}
    ).sort_values("conflict_contribution", ascending=False)

    merged = pd.concat([record, scores.add_prefix("S_")], axis=1)
    return merged, diagnostics, contributions, scores


def summarize_quality(record: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grouped = record.groupby(["dataset", "domain"], observed=True)
    summary = grouped.agg(
        n=("Q", "size"),
        Q_mean=("Q", "mean"),
        Q_median=("Q", "median"),
        Q_std=("Q", "std"),
        conflict_rate=("conflict", "mean"),
        mean_confidence=("confidence", "mean"),
    ).reset_index()
    se = summary["Q_std"].fillna(0.0) / np.sqrt(summary["n"].clip(lower=1))
    summary["Q_ci95_low"] = (summary["Q_mean"] - 1.96 * se).clip(0, 1)
    summary["Q_ci95_high"] = (summary["Q_mean"] + 1.96 * se).clip(0, 1)

    rows = []
    for domain in ["arxiv", "github"]:
        a = record.loc[(record.dataset == "sample") & (record.domain == domain), "Q"].to_numpy()
        b = record.loc[(record.dataset == "extended") & (record.domain == domain), "Q"].to_numpy()
        if not len(a) or not len(b):
            continue
        pooled = math.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) /
                           max(len(a) + len(b) - 2, 1))
        ks_stat, ks_pvalue = ks_2sample(a, b)
        rows.append(
            {
                "domain": domain,
                "n_sample": len(a),
                "n_extended": len(b),
                "Q_sample": np.mean(a),
                "Q_extended": np.mean(b),
                "delta_extended_minus_sample": np.mean(b) - np.mean(a),
                "cohen_d": (np.mean(b) - np.mean(a)) / max(pooled, 1e-12),
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_pvalue,
                "conflict_rate_sample": record.loc[(record.dataset == "sample") & (record.domain == domain), "conflict"].mean(),
                "conflict_rate_extended": record.loc[(record.dataset == "extended") & (record.domain == domain), "conflict"].mean(),
            }
        )
    comparison = pd.DataFrame(rows)

    # A2/A3 是相应域的更完整文件；用于最终域评分时替代 A1 中的同域抽样，避免重复计数。
    selected_rows = []
    for domain in sorted(record.domain.unique()):
        ext = record[(record.dataset == "extended") & (record.domain == domain)]
        chosen = ext if len(ext) else record[(record.dataset == "sample") & (record.domain == domain)]
        selected_rows.append(
            {
                "domain": domain,
                "selected_source": "extended" if len(ext) else "sample",
                "n": len(chosen),
                "Q": chosen.Q.mean(),
                "Q_std": chosen.Q.std(ddof=1),
                "conflict_rate": chosen.conflict.mean(),
            }
        )
    selected = pd.DataFrame(selected_rows)
    return summary, comparison, selected


def close_composition(p: np.ndarray, pseudocount: float = 1e-4) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.where(np.isfinite(p) & (p >= 0), p, 0.0)
    p = p + pseudocount
    return p / p.sum(axis=1, keepdims=True)


def ilr_transform(p: np.ndarray, pseudocount: float = 1e-4) -> np.ndarray:
    closed = close_composition(p, pseudocount)
    clr = np.log(closed) - np.log(closed).mean(axis=1, keepdims=True)
    basis = helmert_basis(closed.shape[1])
    return clr @ basis.T


def load_pair(table_dir: Path, mixture_name: str, loss_name: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str], list[str]]:
    mix = pd.read_csv(table_dir / mixture_name)
    loss = pd.read_csv(table_dir / loss_name)
    joined = mix.merge(loss, on="index", how="inner", validate="one_to_one")
    mix_cols = [c for c in mix.columns if c != "index"]
    loss_cols = [c for c in loss.columns if c != "index"]
    p = joined[mix_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y = joined[loss_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if np.isnan(p).any() or np.isnan(y).any():
        raise ValueError(f"{mixture_name}/{loss_name} 含非数值或缺失字段")
    return joined, p, y, mix_cols, loss_cols


def macro_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    rho = []
    for j in range(y.shape[1]):
        value = spearman_statistic(y[:, j], pred[:, j])
        if np.isfinite(value):
            rho.append(value)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean(axis=0, keepdims=True)) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "mae": float(np.mean(np.abs(y - pred))),
        "r2": 1.0 - ss_res / max(ss_tot, 1e-12),
        "mean_target_spearman": float(np.mean(rho)) if rho else np.nan,
    }


def affine_calibrated_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    calibrated = np.zeros_like(pred)
    for j in range(y.shape[1]):
        design = np.column_stack([np.ones(len(pred)), pred[:, j]])
        coef, *_ = np.linalg.lstsq(design, y[:, j], rcond=None)
        calibrated[:, j] = design @ coef
    result = macro_metrics(y, calibrated)
    return {f"affine_{k}": v for k, v in result.items()}


def perturb_share(p: np.ndarray, indices: Sequence[int], delta: float) -> np.ndarray:
    """各指定分量增加 delta，其余分量按比例缩减，始终留在单纯形。"""
    q = np.asarray(p, dtype=float).copy()
    selected = np.zeros(q.size, dtype=bool)
    selected[list(indices)] = True
    total_add = delta * selected.sum()
    available = q[~selected].sum()
    if available <= total_add or np.any(q[selected] + delta >= 1.0):
        return q
    q[~selected] *= (available - total_add) / available
    q[selected] += delta
    q = np.clip(q, 0.0, None)
    return q / q.sum()


def quality_projection(
    label: str,
    indices: np.ndarray,
    p: np.ndarray,
    mix_cols: list[str],
    selected_quality: pd.DataFrame,
    mapping_path: Path,
) -> pd.DataFrame:
    mapping = pd.read_csv(mapping_path)
    q_map = selected_quality.set_index("domain")["Q"].to_dict()
    domain_names = [c.replace("train_the_pile_", "") for c in mix_cols]
    q_vector = []
    mapped = []
    for domain in domain_names:
        row = mapping.loc[mapping.mixture_domain == domain]
        quality_domain = None if row.empty else str(row.iloc[0].quality_domain)
        value = q_map.get(quality_domain, np.nan)
        q_vector.append(value)
        mapped.append(np.isfinite(value))
    q_vector = np.asarray(q_vector, float)
    mapped = np.asarray(mapped, bool)
    q_min, q_max = float(selected_quality.Q.min()), float(selected_quality.Q.max())
    q_global = float(np.average(selected_quality.Q, weights=selected_quality.n))
    known_sum = p[:, mapped] @ q_vector[mapped]
    coverage = p[:, mapped].sum(axis=1)
    missing = 1.0 - coverage
    return pd.DataFrame(
        {
            "dataset": label,
            "index": indices,
            "mapped_mass": coverage,
            "Q_lower": known_sum + missing * q_min,
            "Q_mid": known_sum + missing * q_global,
            "Q_upper": known_sum + missing * q_max,
        }
    )


def fit_mixture_model(
    root: Path,
    selected_quality: pd.DataFrame,
    pseudocount: float = 1e-4,
    delta: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, QuadraticRidgeModel]:
    table_dir = root / "regmix_tables"
    train, p_train, y_train, mix_cols, loss_cols = load_pair(
        table_dir, "train_mixture_1m.csv", "train_pile_loss_1m.csv"
    )
    z_train = ilr_transform(p_train, pseudocount)
    model = fit_quadratic_ridge_cv(z_train, y_train)

    validation_rows = []
    projections = [quality_projection("train_1m", train["index"].to_numpy(), p_train, mix_cols,
                                      selected_quality, root / "domain_mapping_guide.csv")]
    test_specs = [
        ("test_1m", "test_mixture_1m.csv", "test_pile_loss_1m.csv", True),
        ("test_60m", "test_mixture_60m.csv", "test_pile_loss_60m.csv", False),
        ("test_1B", "test_mixture_1B.csv", "test_pile_loss_1B.csv", False),
    ]
    for label, mix_name, loss_name, same_scale in test_specs:
        joined, p, y, mcols, lcols = load_pair(table_dir, mix_name, loss_name)
        if mcols != mix_cols or lcols != loss_cols:
            raise ValueError(f"{label} 的字段与训练表不一致")
        pred = model.predict(ilr_transform(p, pseudocount))
        row = {"dataset": label, "same_scale_as_training": int(same_scale), **macro_metrics(y, pred)}
        # 跨规模仅作描述性仿射校准；核心迁移证据是未校准排序相关。
        row.update(affine_calibrated_metrics(y, pred))
        validation_rows.append(row)
        projections.append(quality_projection(label, joined["index"].to_numpy(), p, mix_cols,
                                              selected_quality, root / "domain_mapping_guide.csv"))
    validation = pd.DataFrame(validation_rows)

    # 在训练配方分布上平均局部边际效应，避免只依赖一个参考点。
    rng = np.random.default_rng(20260924)
    take = rng.choice(len(p_train), size=min(200, len(p_train)), replace=False)
    effects = []
    for j, col in enumerate(mix_cols):
        diffs = []
        target_diffs = []
        for p0 in p_train[take]:
            p1 = perturb_share(p0, [j], delta)
            if np.allclose(p0, p1):
                continue
            f0 = model.predict(ilr_transform(p0[None, :], pseudocount))[0]
            f1 = model.predict(ilr_transform(p1[None, :], pseudocount))[0]
            target_diffs.append((f1 - f0) / delta)
            diffs.append(np.mean(f1 - f0) / delta)
        arr = np.asarray(target_diffs)
        effects.append(
            {
                "domain": col.replace("train_the_pile_", ""),
                "mean_macro_loss_derivative": float(np.mean(diffs)),
                "std_macro_loss_derivative": float(np.std(diffs, ddof=1)),
                "beneficial_if_negative": int(np.mean(diffs) < 0),
                **{f"effect_{loss_cols[t].replace('metric/the_pile_', '').replace('_val_loss', '')}": float(np.mean(arr[:, t]))
                   for t in range(len(loss_cols))},
            }
        )
    effects_df = pd.DataFrame(effects).sort_values("mean_macro_loss_derivative")

    # 二阶有限差分给出领域组合的协同/拮抗；负值表示组合降低 Loss 超过可加预期。
    p_ref = close_composition(np.mean(p_train, axis=0, keepdims=True), pseudocount)[0]
    f0 = float(model.predict(ilr_transform(p_ref[None, :], pseudocount)).mean())
    interaction_rows = []
    for j in range(len(mix_cols)):
        pj = perturb_share(p_ref, [j], delta)
        fj = float(model.predict(ilr_transform(pj[None, :], pseudocount)).mean())
        for k in range(j + 1, len(mix_cols)):
            pk = perturb_share(p_ref, [k], delta)
            pjk = perturb_share(p_ref, [j, k], delta)
            fk = float(model.predict(ilr_transform(pk[None, :], pseudocount)).mean())
            fjk = float(model.predict(ilr_transform(pjk[None, :], pseudocount)).mean())
            interaction_rows.append(
                {
                    "domain_1": mix_cols[j].replace("train_the_pile_", ""),
                    "domain_2": mix_cols[k].replace("train_the_pile_", ""),
                    "second_difference_per_share2": (fjk - fj - fk + f0) / (delta ** 2),
                    "synergy_if_negative": int((fjk - fj - fk + f0) < 0),
                }
            )
    interactions = pd.DataFrame(interaction_rows)
    interactions["abs_interaction"] = interactions.second_difference_per_share2.abs()
    interactions = interactions.sort_values("abs_interaction", ascending=False)

    extrapolation_rows = []
    for scale in ["10b", "70b"]:
        joined, p, y, mcols, lcols = load_pair(
            table_dir, f"est_mixture_{scale}.csv", f"est_pile_loss_{scale}.csv"
        )
        pred = model.predict(ilr_transform(p, pseudocount))
        row = {"dataset": f"estimated_{scale}", **macro_metrics(y, pred)}
        row.update(affine_calibrated_metrics(y, pred))
        row["independent_experiment"] = 0
        extrapolation_rows.append(row)
        projections.append(quality_projection(f"estimated_{scale}", joined["index"].to_numpy(), p,
                                              mix_cols, selected_quality, root / "domain_mapping_guide.csv"))
    extrapolation = pd.DataFrame(extrapolation_rows)
    projection = pd.concat(projections, ignore_index=True)
    return validation, effects_df, interactions, extrapolation, projection, model


def _chinese_font_candidates(bold: bool = False) -> list[Path]:
    """按优先级返回常见中文字体；Windows/VSCode 环境优先使用微软雅黑。"""
    windows_fonts = Path(r"C:\Windows\Fonts")
    names = (
        ["msyhbd.ttc", "simhei.ttf", "dengb.ttf", "simsun.ttc"]
        if bold
        else ["msyh.ttc", "simhei.ttf", "deng.ttf", "simsun.ttc"]
    )
    candidates = [windows_fonts / name for name in names]
    # 兼容 Linux、Overleaf 容器和部分 Conda 环境。
    candidates.extend([
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
    ])
    return candidates


def _find_chinese_font_path(bold: bool = False) -> Path | None:
    return next((path for path in _chinese_font_candidates(bold) if path.exists()), None)


def _configure_matplotlib_chinese(matplotlib_module) -> None:
    """为 Matplotlib 显式注册中文字体，避免标题显示为方框。"""
    from matplotlib import font_manager

    font_path = _find_chinese_font_path(False)
    if font_path is not None:
        font_manager.fontManager.addfont(str(font_path))
        font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
        matplotlib_module.rcParams["font.family"] = "sans-serif"
        matplotlib_module.rcParams["font.sans-serif"] = [
            font_name, "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
            "WenQuanYi Micro Hei", "DejaVu Sans",
        ]
    else:
        warnings.warn(
            "未找到中文字体；请安装微软雅黑、黑体或 Noto Sans CJK SC，"
            "或者在 _chinese_font_candidates() 中加入本机字体路径。"
        )
    # 使用 ASCII 减号，避免部分中文字体缺少 Unicode 负号 U+2212。
    matplotlib_module.rcParams["axes.unicode_minus"] = False


def make_plots(
    output_dir: Path,
    selected: pd.DataFrame,
    summary: pd.DataFrame,
    effects: pd.DataFrame,
    comparison: pd.DataFrame,
    contributions: pd.DataFrame,
    validation: pd.DataFrame,
    interactions: pd.DataFrame,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        _configure_matplotlib_chinese(matplotlib)
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        make_plots_pillow(
            output_dir, selected, summary, effects, comparison,
            contributions, validation, interactions,
        )
        return
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    q = selected.sort_values("Q")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(q.domain, q.Q, color="#3B6EA8")
    ax.set(xlabel="领域综合质量得分 Q", ylabel="", xlim=(0, 1), title="选定领域的综合质量得分")
    fig.tight_layout()
    fig.savefig(fig_dir / "quality_domain_scores.png", dpi=220)
    plt.close(fig)

    c = summary.pivot(index="domain", columns="dataset", values="conflict_rate").fillna(0)
    fig, ax = plt.subplots(figsize=(9, 5))
    c.plot(kind="bar", ax=ax, color=["#6B8E23", "#C16E70"][: len(c.columns)])
    ax.set(ylabel="质量冲突率", xlabel="", title="各领域及数据源的质量冲突率")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(fig_dir / "quality_conflict_rates.png", dpi=220)
    plt.close(fig)

    e = effects.sort_values("mean_macro_loss_derivative")
    colors = np.where(e.mean_macro_loss_derivative < 0, "#2E8B57", "#B55A4A")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(e.domain, e.mean_macro_loss_derivative, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(xlabel="领域配比增加 1 单位时的平均 Loss 变化", ylabel="",
           title="各预训练领域对平均 Loss 的局部边际效应")
    fig.tight_layout()
    fig.savefig(fig_dir / "mixture_domain_effects.png", dpi=220)
    plt.close(fig)

    c = contributions.head(10).sort_values("conflict_contribution")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(c.metric, c.conflict_contribution, color="#9C6ADE")
    ax.set(xlabel="标准化冲突贡献度", ylabel="", title="质量冲突的主要指标来源")
    fig.tight_layout()
    fig.savefig(fig_dir / "conflict_metric_contributions.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(validation.dataset, validation.mean_target_spearman, color="#4178BE")
    ax.set(ylabel="目标平均 Spearman 相关系数", xlabel="", ylim=(0, 1),
           title="跨模型规模的配比排序迁移能力")
    fig.tight_layout()
    fig.savefig(fig_dir / "mixture_validation_spearman.png", dpi=220)
    plt.close(fig)

    # 另外三幅图由 Pillow 绘制；设为 False 可避免覆盖上面的 Matplotlib 图。
    make_plots_pillow(
        output_dir, selected, summary, effects, comparison,
        contributions, validation, interactions, include_standard=False,
    )


def _pil_font(size: int, bold: bool = False):
    from PIL import ImageFont
    # 中文字体必须排在 Arial 之前，否则中文会显示为方框。
    candidates = _chinese_font_candidates(bold)
    candidates.append(Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"))
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _hbar_pillow(
    path: Path,
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    xlabel: str,
    x_min: float | None = None,
    x_max: float | None = None,
    color_negative: tuple[int, int, int] = (46, 139, 87),
    color_positive: tuple[int, int, int] = (181, 90, 74),
) -> None:
    from PIL import Image, ImageDraw
    labels = list(labels)
    values = np.asarray(values, float)
    width, height = 1800, max(850, 125 + 62 * len(labels))
    left, right, top, bottom = 430, 80, 110, 100
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _pil_font(38, True)
    label_font = _pil_font(25)
    value_font = _pil_font(22)
    axis_font = _pil_font(25)
    draw.text((width / 2, 38), title, fill=(20, 20, 20), font=title_font, anchor="ma")
    lo = float(np.nanmin(values)) if x_min is None else float(x_min)
    hi = float(np.nanmax(values)) if x_max is None else float(x_max)
    if lo >= 0 and x_min is None:
        lo = 0.0
    if hi <= 0 and x_max is None:
        hi = 0.0
    padding = max((hi - lo) * 0.08, 1e-9)
    lo = lo - padding if x_min is None else lo
    hi = hi + padding if x_max is None else hi
    plot_w, plot_h = width - left - right, height - top - bottom
    x_of = lambda v: left + (float(v) - lo) / max(hi - lo, 1e-12) * plot_w
    zero_x = x_of(np.clip(0.0, lo, hi))
    for tick in np.linspace(lo, hi, 6):
        x = x_of(tick)
        draw.line((x, top, x, top + plot_h), fill=(225, 225, 225), width=2)
        draw.text((x, top + plot_h + 12), f"{tick:.2f}", fill=(50, 50, 50), font=value_font, anchor="ma")
    draw.line((zero_x, top, zero_x, top + plot_h), fill=(60, 60, 60), width=3)
    row_h = plot_h / max(len(labels), 1)
    for i, (label, value) in enumerate(zip(labels, values)):
        y = top + (i + 0.5) * row_h
        draw.text((left - 18, y), str(label), fill=(30, 30, 30), font=label_font, anchor="rm")
        x = x_of(value)
        color = color_negative if value < 0 else color_positive
        draw.rectangle((min(zero_x, x), y - row_h * 0.28, max(zero_x, x), y + row_h * 0.28), fill=color)
        anchor = "lm" if value >= 0 else "rm"
        offset = 10 if value >= 0 else -10
        draw.text((x + offset, y), f"{value:.3f}", fill=(25, 25, 25), font=value_font, anchor=anchor)
    draw.text((left + plot_w / 2, height - 34), xlabel, fill=(30, 30, 30), font=axis_font, anchor="ma")
    image.save(path, dpi=(220, 220))


def make_plots_pillow(
    output_dir: Path,
    selected: pd.DataFrame,
    summary: pd.DataFrame,
    effects: pd.DataFrame,
    comparison: pd.DataFrame,
    contributions: pd.DataFrame,
    validation: pd.DataFrame,
    interactions: pd.DataFrame,
    include_standard: bool = True,
) -> None:
    """无 matplotlib 时用 Pillow 生成论文所需的静态图。"""
    from PIL import Image, ImageDraw
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if include_standard:
        q = selected.sort_values("Q")
        _hbar_pillow(
            fig_dir / "quality_domain_scores.png", q.domain, q.Q,
            "选定领域的综合质量得分", "领域综合质量得分 Q", 0.0, 1.0,
            color_positive=(59, 110, 168),
        )
        c = selected.sort_values("conflict_rate")
        _hbar_pillow(
            fig_dir / "quality_conflict_rates.png", c.domain, c.conflict_rate,
            "选定数据源的各领域质量冲突率", "质量冲突率", 0.0,
            max(0.65, float(c.conflict_rate.max()) * 1.1), color_positive=(193, 110, 112),
        )
        e = effects.sort_values("mean_macro_loss_derivative")
        _hbar_pillow(
            fig_dir / "mixture_domain_effects.png", e.domain, e.mean_macro_loss_derivative,
            "各预训练领域对平均 Loss 的局部边际效应",
            "领域配比增加 1 单位时的平均 Loss 变化",
        )
        top = contributions.head(10).sort_values("conflict_contribution")
        _hbar_pillow(
            fig_dir / "conflict_metric_contributions.png", top.metric, top.conflict_contribution,
            "质量冲突的主要指标来源", "标准化冲突贡献度", 0.0,
            float(top.conflict_contribution.max()) * 1.15, color_positive=(128, 86, 173),
        )
        v = validation.sort_values("mean_target_spearman")
        _hbar_pillow(
            fig_dir / "mixture_validation_spearman.png", v.dataset, v.mean_target_spearman,
            "跨模型规模的配比排序迁移能力", "目标平均 Spearman 相关系数", 0.0, 1.0,
            color_positive=(65, 120, 190),
        )

    # A1 与扩展集的领域均值对照。
    width, height = 1500, 850
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((width / 2, 45), "A1 抽样集与扩展集的质量得分对比",
              fill=(20, 20, 20), font=_pil_font(38, True), anchor="ma")
    left, right, top_y, bottom = 170, 90, 130, 120
    plot_w, plot_h = width - left - right, height - top_y - bottom
    draw.line((left, top_y + plot_h, left + plot_w, top_y + plot_h), fill=(60, 60, 60), width=3)
    colors = [(59, 110, 168), (193, 110, 112)]
    domains = comparison.domain.tolist()
    group_w = plot_w / max(len(domains), 1)
    for tick in np.linspace(0, 0.8, 5):
        y = top_y + plot_h * (1 - tick / 0.8)
        draw.line((left, y, left + plot_w, y), fill=(225, 225, 225), width=2)
        draw.text((left - 15, y), f"{tick:.1f}", fill=(40, 40, 40), font=_pil_font(22), anchor="rm")
    for i, row in comparison.reset_index(drop=True).iterrows():
        center = left + (i + 0.5) * group_w
        for j, key in enumerate(["Q_sample", "Q_extended"]):
            value = float(row[key])
            x0 = center + (j - 1) * 105
            x1 = x0 + 90
            y = top_y + plot_h * (1 - value / 0.8)
            draw.rectangle((x0, y, x1, top_y + plot_h), fill=colors[j])
            draw.text(((x0 + x1) / 2, y - 12), f"{value:.3f}", fill=(25, 25, 25), font=_pil_font(22), anchor="ms")
        draw.text((center, top_y + plot_h + 35), str(row.domain), fill=(30, 30, 30), font=_pil_font(27), anchor="ma")
    draw.rectangle((width - 430, 95, width - 405, 120), fill=colors[0])
    draw.text((width - 395, 108), "A1 抽样集", fill=(30, 30, 30), font=_pil_font(22), anchor="lm")
    draw.rectangle((width - 250, 95, width - 225, 120), fill=colors[1])
    draw.text((width - 215, 108), "扩展集", fill=(30, 30, 30), font=_pil_font(22), anchor="lm")
    image.save(fig_dir / "quality_extension_comparison.png", dpi=(220, 220))

    # 全部 17 域的二阶组合效应矩阵。
    domains = sorted(set(interactions.domain_1) | set(interactions.domain_2))
    n = len(domains)
    matrix = np.zeros((n, n), float)
    index = {d: i for i, d in enumerate(domains)}
    for row in interactions.itertuples(index=False):
        j, k = index[row.domain_1], index[row.domain_2]
        matrix[j, k] = matrix[k, j] = float(row.second_difference_per_share2)
    bound = float(np.quantile(np.abs(matrix[np.triu_indices(n, 1)]), 0.95)) or 1.0
    cell, left_m, top_m = 56, 330, 230
    width, height = left_m + n * cell + 120, top_m + n * cell + 100
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((width / 2, 42), "领域配比的两两交互效应",
              fill=(20, 20, 20), font=_pil_font(36, True), anchor="ma")
    label_font = _pil_font(17)
    for i, domain in enumerate(domains):
        draw.text((left_m - 12, top_m + (i + 0.5) * cell), domain,
                  fill=(30, 30, 30), font=label_font, anchor="rm")
        draw.text((left_m + (i + 0.5) * cell, top_m - 12), str(i + 1),
                  fill=(30, 30, 30), font=label_font, anchor="ms")
    for i in range(n):
        for j in range(n):
            value = float(np.clip(matrix[i, j] / bound, -1, 1))
            if value < 0:
                color = (int(245 + 145 * value), int(245 + 55 * value), int(245 + 95 * value))
            else:
                color = (int(245 - 65 * value), int(245 - 115 * value), int(245 - 35 * value))
            x0, y0 = left_m + j * cell, top_m + i * cell
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=color, outline=(235, 235, 235))
    draw.text((left_m + n * cell / 2, height - 35),
              "列 1—17 与行顺序一致；绿色为协同，红色为拮抗",
              fill=(30, 30, 30), font=_pil_font(23), anchor="ma")
    image.save(fig_dir / "mixture_interaction_heatmap.png", dpi=(220, 220))

    # 论文中的建模流程图。
    width, height = 1800, 1000
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((width / 2, 45), "问题一建模流程", fill=(20, 20, 20),
              font=_pil_font(40, True), anchor="ma")
    boxes = [
        ("A1—A3 质量信号", "22 项指标数值化\n指标方向统一\n参考分布归一化"),
        ("稳健质量模型", "有界 CRITIC 权重\nHuber 稳健聚合\n冲突度与置信度"),
        ("领域质量 Q", "A1 与 A2/A3 对照\n领域聚合\n映射不确定性"),
        ("A4—A15 配比", "单纯形闭合\nILR 坐标变换\n13 个 Loss 目标"),
        ("二次岭回归", "主效应\n两两交互效应\n正则化"),
        ("验证与输出", "1M 绝对验证\n60M/1B 排序迁移\n10B/70B 稳定性"),
    ]
    positions = [(90, 190), (650, 190), (1210, 190), (90, 600), (650, 600), (1210, 600)]
    box_w, box_h = 500, 245
    for (head, body), (x, y) in zip(boxes, positions):
        draw.rounded_rectangle((x, y, x + box_w, y + box_h), radius=28,
                               fill=(239, 245, 251), outline=(59, 110, 168), width=4)
        draw.text((x + box_w / 2, y + 45), head, fill=(30, 60, 95),
                  font=_pil_font(29, True), anchor="ma")
        for li, line in enumerate(body.split("\n")):
            draw.text((x + box_w / 2, y + 105 + li * 42), line, fill=(35, 35, 35),
                      font=_pil_font(24), anchor="ma")
    for (x1, y1), (x2, y2) in [(positions[0], positions[1]), (positions[1], positions[2]),
                               (positions[3], positions[4]), (positions[4], positions[5])]:
        start = (x1 + box_w, y1 + box_h / 2)
        end = (x2, y2 + box_h / 2)
        draw.line((*start, *end), fill=(80, 80, 80), width=5)
        draw.polygon([(end[0], end[1]), (end[0] - 18, end[1] - 12), (end[0] - 18, end[1] + 12)], fill=(80, 80, 80))
    draw.line((1460, 435, 340, 600), fill=(80, 80, 80), width=5)
    draw.polygon([(340, 600), (360, 578), (365, 610)], fill=(80, 80, 80))
    image.save(fig_dir / "problem1_workflow.png", dpi=(220, 220))


def write_outputs(args: argparse.Namespace) -> None:
    root = args.data_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    raw = load_quality_data(root, args.max_records_per_file)
    record, diagnostics, contributions, _ = score_quality(
        raw,
        critic_blend=args.critic_blend,
        conflict_quantile=args.conflict_quantile,
        conflict_spread=args.conflict_spread,
    )
    summary, comparison, selected = summarize_quality(record)
    validation, effects, interactions, extrapolation, projection, model = fit_mixture_model(
        root, selected, pseudocount=args.pseudocount, delta=args.effect_delta
    )

    record.to_csv(out / "quality_record_scores.csv.gz", index=False, compression="gzip")
    summary.to_csv(out / "quality_domain_summary.csv", index=False)
    comparison.to_csv(out / "quality_extension_comparison.csv", index=False)
    selected.to_csv(out / "quality_domain_selected.csv", index=False)
    diagnostics.to_csv(out / "quality_metric_diagnostics.csv", index=False)
    contributions.to_csv(out / "conflict_metric_contributions.csv", index=False)
    validation.to_csv(out / "mixture_validation_metrics.csv", index=False)
    effects.to_csv(out / "mixture_domain_effects.csv", index=False)
    interactions.to_csv(out / "mixture_interactions.csv", index=False)
    extrapolation.to_csv(out / "extrapolation_stability.csv", index=False)
    projection.to_csv(out / "mixture_quality_projection.csv", index=False)

    summary_json = {
        "formal_run": args.max_records_per_file is None,
        "quality_records": int(len(record)),
        "sample_records": int((record.dataset == "sample").sum()),
        "extended_records": int((record.dataset == "extended").sum()),
        "quality_domains": int(record.domain.nunique()),
        "conflict_threshold": float(diagnostics.conflict_threshold_disagreement.iloc[0]),
        "overall_conflict_rate": float(record.conflict.mean()),
        "critic_blend": args.critic_blend,
        "conflict_quantile": args.conflict_quantile,
        "conflict_spread": args.conflict_spread,
        "pseudocount": args.pseudocount,
        "effect_delta": args.effect_delta,
        "mixture_selected_ridge_alpha": model.alpha,
        "primary_validation": validation.to_dict(orient="records"),
        "extrapolation_note": "A12--A15 are estimated/subset data; use for stability discussion, not independent proof.",
        "quality_mapping_note": "Q is partially identified for unmapped mixture domains and is not added as a separately identifiable regressor.",
    }
    (out / "analysis_summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not args.no_plots:
        make_plots(
            out, selected, summary, effects, comparison,
            contributions, validation, interactions,
        )

    print(json.dumps(summary_json, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    write_outputs(parse_args())
