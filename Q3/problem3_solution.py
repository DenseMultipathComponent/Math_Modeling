#!/usr/bin/env python3
"""2026 中国研究生数学建模竞赛 F 题问题三求解脚本。

本脚本读取问题二输出的广义标度律与问题一传递的配比响应，并读取 C7
架构元数据中的上下文长度。对给定算力预算 C 和外生上下文长度 L_ctx，
求解参数量 N、训练数据量 D、数据质量 Q，并比较三类质量成本函数。

默认同时给出两种配比口径：
1. reference：固定问题二参考配比 p0，作为论文主结果；
2. candidates：在问题一/二已评估的经验配方库中离散选择，作为联合优化敏感性。

脚本不把题面或数据说明中的建议性文字当作既定算法或结论；成本函数参数来自
题面附录 B，标度律参数来自用户已运行的问题二结果。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize


BASE = Path(__file__).resolve().parent
ETA_ATTENTION = 2.0e-4
L_CONTEXT_CRITICAL = 6.0 / ETA_ATTENTION
DOMAIN_PREFIX = "train_the_pile_"

COST_MODELS = {
    "exponential": {"label": "指数型", "gamma": 1.0e7, "lambda": 6.0},
    "power": {"label": "幂函数型", "gamma": 5.0e9, "lambda": 4.0},
    "logarithmic": {"label": "对数渐进型", "gamma": 2.0e9, "lambda": 10.0},
}


@dataclass(frozen=True)
class ScalingParameters:
    E: float
    A: float
    B: float
    alpha: float
    beta: float
    gamma_quality: float
    kappa: float = 1.0
    lambda_mixture: float = 1.0


@dataclass(frozen=True)
class EvidenceBounds:
    n_min: float
    n_max: float
    d_min: float
    d_max: float
    b1_n_min: float
    b1_n_max: float
    b1_d_min: float
    b1_d_max: float


def configure_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def save_csv(frame, path: Path) -> None:
    pd.DataFrame(frame).to_csv(path, index=False, encoding="utf-8-sig")


def find_latest_problem2_output(explicit: Path | None) -> Path:
    if explicit is not None:
        candidate = explicit.resolve()
        required = [candidate / "generalized_model.json", candidate / "mixture_scenarios.csv"]
        if not all(path.is_file() for path in required):
            raise FileNotFoundError(f"指定的问题二目录缺少接口文件：{candidate}")
        return candidate

    root = BASE.parent / "Q2" / "problem2_outputs"
    if not root.is_dir():
        raise FileNotFoundError(f"找不到问题二输出根目录：{root}")
    candidates = []
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        required = [directory / "generalized_model.json", directory / "mixture_scenarios.csv"]
        if all(path.is_file() for path in required):
            candidates.append(directory)
    if not candidates:
        raise FileNotFoundError("没有找到同时含 generalized_model.json 与 mixture_scenarios.csv 的问题二输出。")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def quality_cost_value(q, name: str):
    spec = COST_MODELS[name]
    q = np.asarray(q, dtype=float)
    if name == "exponential":
        return spec["gamma"] * np.exp(spec["lambda"] * q)
    if name == "power":
        return spec["gamma"] * np.power(q, spec["lambda"])
    if name == "logarithmic":
        return spec["gamma"] * np.log1p(spec["lambda"] * q)
    raise KeyError(name)


def quality_cost_derivative(q, name: str):
    spec = COST_MODELS[name]
    q = np.asarray(q, dtype=float)
    if name == "exponential":
        return spec["gamma"] * spec["lambda"] * np.exp(spec["lambda"] * q)
    if name == "power":
        return spec["gamma"] * spec["lambda"] * np.power(q, spec["lambda"] - 1.0)
    if name == "logarithmic":
        return spec["gamma"] * spec["lambda"] / (1.0 + spec["lambda"] * q)
    raise KeyError(name)


def load_context_lengths(c7_path: Path) -> tuple[pd.DataFrame, list[int]]:
    frame = pd.read_csv(c7_path)
    required = {"model_name", "max_position_embeddings"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"C7 缺少字段：{sorted(missing)}")
    frame["max_position_embeddings"] = pd.to_numeric(
        frame["max_position_embeddings"], errors="coerce"
    )
    valid = frame["max_position_embeddings"].dropna()
    valid = valid[valid > 0]
    if valid.empty:
        raise ValueError("C7 中没有有效的正上下文长度。")
    contexts = sorted({int(value) for value in valid})
    return frame, contexts


def load_model(problem2_output: Path) -> tuple[dict, ScalingParameters, EvidenceBounds]:
    model = json.loads((problem2_output / "generalized_model.json").read_text(encoding="utf-8"))
    base = model["baseline_parameters"]
    params = ScalingParameters(
        E=float(base["E"]), A=float(base["A"]), B=float(base["B"]),
        alpha=float(base["alpha"]), beta=float(base["beta"]),
        gamma_quality=float(model["gamma"]),
        kappa=1.0,
        lambda_mixture=float(model["mixture_bridge"].get("mixture_transfer_lambda", 1.0)),
    )
    summary_path = problem2_output / "analysis_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    b1_n = summary.get("B1_N_range", [0.070542, 11.965825])
    b1_d = summary.get("B1_D_range", [0.134, 299.893])

    n_max_b = 10000.0
    d_max_b = 36000.0
    b9_path = problem2_output / "B9_large_model_scenarios.csv"
    if b9_path.is_file():
        b9 = pd.read_csv(b9_path)
        if "N_params_B" in b9:
            n_max_b = max(n_max_b, float(pd.to_numeric(b9.N_params_B, errors="coerce").max()))
        if "D_tokens_B" in b9:
            d_max_b = max(d_max_b, float(pd.to_numeric(b9.D_tokens_B, errors="coerce").max()))

    bounds = EvidenceBounds(
        n_min=float(b1_n[0]) * 1.0e9,
        n_max=n_max_b * 1.0e9,
        d_min=float(b1_d[0]) * 1.0e9,
        d_max=d_max_b * 1.0e9,
        b1_n_min=float(b1_n[0]) * 1.0e9,
        b1_n_max=float(b1_n[1]) * 1.0e9,
        b1_d_min=float(b1_d[0]) * 1.0e9,
        b1_d_max=float(b1_d[1]) * 1.0e9,
    )
    return model, params, bounds


def close_composition(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, float), 0.0, None)
    total = values.sum(axis=-1, keepdims=True)
    if np.any(total <= 0):
        raise ValueError("领域配比存在非正总和。")
    return values / total


def load_mixture_library(problem2_output: Path, model: dict, max_mixtures: int) -> tuple[pd.DataFrame, list[str]]:
    samples = pd.read_csv(problem2_output / "mixture_scenarios.csv")
    domain_columns = [column for column in samples.columns if column.startswith(DOMAIN_PREFIX)]
    if len(domain_columns) != 17:
        raise ValueError(f"问题二配方接口应有 17 个领域列，实际为 {len(domain_columns)}。")
    required = {"dataset", "index", "Q_closed_mid", "mixture_log_relative_loss"}
    missing = required - set(samples.columns)
    if missing:
        raise ValueError(f"mixture_scenarios.csv 缺少字段：{sorted(missing)}")

    composition = close_composition(samples[domain_columns].to_numpy(float))
    samples.loc[:, domain_columns] = composition
    samples["q_base"] = pd.to_numeric(samples["Q_closed_mid"], errors="coerce")
    samples["mixture_r"] = pd.to_numeric(samples["mixture_log_relative_loss"], errors="coerce")
    samples = samples.dropna(subset=["q_base", "mixture_r"]).copy()
    samples = samples[(samples.q_base > 0) & (samples.q_base <= 1)].copy()

    # 在 r 与 q_base 的不同权衡下取代表性经验配方，避免 1214 个配方全部进入细网格。
    r = samples.mixture_r.to_numpy(float)
    q = samples.q_base.to_numpy(float)
    r_norm = (r - r.min()) / max(r.max() - r.min(), 1e-12)
    q_norm = (q - q.min()) / max(q.max() - q.min(), 1e-12)
    chosen: set[int] = set()
    weight_count = max(8, min(max_mixtures, 64))
    for weight in np.linspace(0.0, 1.0, weight_count):
        score = weight * r_norm + (1.0 - weight) * q_norm
        chosen.add(int(np.argmin(score)))
    chosen.update(np.argsort(r)[: min(max_mixtures // 2, len(samples))].tolist())
    chosen.update(np.argsort(q)[: min(max_mixtures // 4, len(samples))].tolist())
    chosen.update(np.argsort(q)[-min(max_mixtures // 4, len(samples)):].tolist())

    selected = samples.iloc[sorted(chosen)].copy()
    if len(selected) > max_mixtures:
        selected = selected.sort_values(["mixture_r", "q_base"]).head(max_mixtures).copy()
    selected["mix_id"] = [f"candidate_{i:03d}" for i in range(len(selected))]
    selected["allocation_policy"] = "candidates"

    reference_p = model["mixture_bridge"]["reference_p"]
    domains = [column.removeprefix(DOMAIN_PREFIX) for column in domain_columns]
    reference_row = {
        "dataset": "reference",
        "index": -1,
        "q_base": float(model["mixture_bridge"]["Q_reference"]),
        "mixture_r": 0.0,
        "mix_id": "reference_p0",
        "allocation_policy": "reference",
    }
    for column, domain in zip(domain_columns, domains):
        reference_row[column] = float(reference_p[domain])
    reference = pd.DataFrame([reference_row])
    library = pd.concat([reference, selected], ignore_index=True, sort=False)
    library.loc[:, domain_columns] = close_composition(library[domain_columns].to_numpy(float))
    return library, domain_columns


def data_for_budget(
    budget: float,
    context_length: float,
    cost_name: str,
    n_raw,
    q,
    q_base: float,
):
    n_raw = np.asarray(n_raw, float)
    q = np.asarray(q, float)
    delta_g = np.maximum(quality_cost_value(q, cost_name) - quality_cost_value(q_base, cost_name), 0.0)
    coefficient = 6.0 + ETA_ATTENTION * context_length
    return budget / (coefficient * n_raw + delta_g)


def generalized_loss(
    params: ScalingParameters,
    n_raw,
    d_raw,
    q,
    q_base: float,
    mixture_r: float,
):
    n = np.asarray(n_raw, float) / 1.0e9
    d = np.asarray(d_raw, float) / 1.0e11
    excess = params.A * np.power(n, -params.alpha) + params.B * np.power(d, -params.beta)
    factor = np.exp(
        params.gamma_quality * params.kappa * (q_base - np.asarray(q, float))
        + params.lambda_mixture * mixture_r
    )
    return params.E + excess * factor


def coarse_candidate(
    candidate: pd.Series,
    budget: float,
    context_length: float,
    cost_name: str,
    params: ScalingParameters,
    bounds: EvidenceBounds,
    n_grid_points: int = 61,
    q_grid_points: int = 25,
) -> dict | None:
    q_base = float(candidate.q_base)
    q_grid = q_base + (1.0 - q_base) * np.linspace(0.0, 1.0, q_grid_points)
    log_n = np.linspace(math.log10(bounds.n_min), math.log10(bounds.n_max), n_grid_points)
    n_grid = np.power(10.0, log_n)
    q_matrix = q_grid[:, None]
    n_matrix = n_grid[None, :]
    d_matrix = data_for_budget(budget, context_length, cost_name, n_matrix, q_matrix, q_base)
    feasible = (d_matrix >= bounds.d_min) & (d_matrix <= bounds.d_max)
    if not np.any(feasible):
        return None
    loss = generalized_loss(
        params, n_matrix, d_matrix, q_matrix, q_base, float(candidate.mixture_r)
    )
    loss = np.where(feasible, loss, np.inf)
    flat = int(np.argmin(loss))
    qi, ni = np.unravel_index(flat, loss.shape)
    return {
        "loss": float(loss[qi, ni]),
        "log_n": float(log_n[ni]),
        "q": float(q_grid[qi]),
        "d": float(d_matrix[qi, ni]),
    }


def refine_candidate(
    candidate: pd.Series,
    coarse: dict,
    budget: float,
    context_length: float,
    cost_name: str,
    params: ScalingParameters,
    bounds: EvidenceBounds,
) -> dict:
    q_base = float(candidate.q_base)
    log_d_min, log_d_max = math.log10(bounds.d_min), math.log10(bounds.d_max)

    def evaluate(vector):
        n_raw = 10.0 ** float(vector[0])
        q = float(vector[1])
        d_raw = float(data_for_budget(budget, context_length, cost_name, n_raw, q, q_base))
        loss = float(generalized_loss(params, n_raw, d_raw, q, q_base, float(candidate.mixture_r)))
        return loss, n_raw, d_raw, q

    def objective(vector):
        return evaluate(vector)[0]

    def lower_data_constraint(vector):
        d_raw = evaluate(vector)[2]
        return math.log10(max(d_raw, 1e-300)) - log_d_min

    def upper_data_constraint(vector):
        d_raw = evaluate(vector)[2]
        return log_d_max - math.log10(max(d_raw, 1e-300))

    result = minimize(
        objective,
        x0=np.array([coarse["log_n"], coarse["q"]]),
        method="SLSQP",
        bounds=[(math.log10(bounds.n_min), math.log10(bounds.n_max)), (q_base, 1.0)],
        constraints=[
            {"type": "ineq", "fun": lower_data_constraint},
            {"type": "ineq", "fun": upper_data_constraint},
        ],
        options={"maxiter": 300, "ftol": 1e-11, "disp": False},
    )
    vector = result.x if result.success and np.all(np.isfinite(result.x)) else np.array([coarse["log_n"], coarse["q"]])
    loss, n_raw, d_raw, q = evaluate(vector)
    if not (bounds.d_min * (1 - 1e-7) <= d_raw <= bounds.d_max * (1 + 1e-7)):
        loss, n_raw, d_raw, q = (
            coarse["loss"], 10.0 ** coarse["log_n"], coarse["d"], coarse["q"]
        )
        success = False
        message = "使用可行粗网格结果"
    else:
        success = bool(result.success)
        message = str(result.message)
    return {
        "loss": float(loss), "N": float(n_raw), "D": float(d_raw), "Q": float(q),
        "solver_success": success, "solver_message": message,
    }


def optimize_scenario(
    library: pd.DataFrame,
    allocation_policy: str,
    budget: float,
    context_length: float,
    cost_name: str,
    params: ScalingParameters,
    bounds: EvidenceBounds,
    top_refine: int = 3,
) -> dict:
    candidates = library[library.allocation_policy == allocation_policy]
    coarse_rows = []
    for row_index, candidate in candidates.iterrows():
        coarse = coarse_candidate(candidate, budget, context_length, cost_name, params, bounds)
        if coarse is not None:
            coarse_rows.append((coarse["loss"], row_index, coarse))
    if not coarse_rows:
        raise RuntimeError(
            f"预算 {budget:g}、上下文 {context_length:g}、成本 {cost_name} 下没有可行解。"
        )
    coarse_rows.sort(key=lambda item: item[0])
    refined = []
    for _, row_index, coarse in coarse_rows[: max(1, top_refine)]:
        candidate = library.loc[row_index]
        solution = refine_candidate(
            candidate, coarse, budget, context_length, cost_name, params, bounds
        )
        refined.append((solution["loss"], row_index, solution))
    refined.sort(key=lambda item: item[0])
    _, row_index, solution = refined[0]
    candidate = library.loc[row_index]

    n_raw, d_raw, q = solution["N"], solution["D"], solution["Q"]
    q_base = float(candidate.q_base)
    delta_g = float(max(quality_cost_value(q, cost_name) - quality_cost_value(q_base, cost_name), 0.0))
    c_train = 6.0 * n_raw * d_raw
    c_attention = ETA_ATTENTION * n_raw * d_raw * context_length
    c_quality = d_raw * delta_g
    c_total = c_train + c_attention + c_quality
    if not np.isclose(c_total, budget, rtol=3e-7, atol=1.0):
        raise RuntimeError("预算恒等式未通过数值检查。")

    q_tolerance = 2e-5
    if q - q_base <= q_tolerance:
        quality_status = "lower_bound"
    elif 1.0 - q <= q_tolerance:
        quality_status = "upper_bound"
    else:
        quality_status = "interior"
    evidence = []
    if not (bounds.b1_n_min <= n_raw <= bounds.b1_n_max):
        evidence.append("N_outside_B1")
    if not (bounds.b1_d_min <= d_raw <= bounds.b1_d_max):
        evidence.append("D_outside_B1")
    if allocation_policy == "candidates":
        evidence.append("empirical_mixture_candidate")

    row = {
        "allocation_policy": allocation_policy,
        "cost_model": cost_name,
        "cost_model_cn": COST_MODELS[cost_name]["label"],
        "budget_FLOPs": float(budget),
        "context_length": float(context_length),
        "attention_to_training_ratio": float(ETA_ATTENTION * context_length / 6.0),
        "attention_dominant": bool(context_length >= L_CONTEXT_CRITICAL),
        "predicted_loss": solution["loss"],
        "N_parameters": n_raw,
        "N_B": n_raw / 1.0e9,
        "D_tokens": d_raw,
        "D_B": d_raw / 1.0e9,
        "Q_base": q_base,
        "Q_opt": q,
        "quality_uplift": q - q_base,
        "quality_status": quality_status,
        "quality_marginal_cost_per_token": float(quality_cost_derivative(q, cost_name)),
        "C_train": c_train,
        "C_quality": c_quality,
        "C_attention": c_attention,
        "share_train": c_train / budget,
        "share_quality": c_quality / budget,
        "share_attention": c_attention / budget,
        "mix_id": str(candidate.mix_id),
        "mix_dataset": str(candidate.dataset),
        "mix_index": int(candidate["index"]),
        "mixture_r": float(candidate.mixture_r),
        "evidence_flag": ";".join(evidence) if evidence else "within_B1",
        "solver_success": bool(solution["solver_success"]),
        "solver_message": str(solution["solver_message"]),
    }
    return row


def parse_policies(name: str) -> list[str]:
    if name == "both":
        return ["reference", "candidates"]
    return [name]


def detect_budget_transitions(
    path: pd.DataFrame,
    share_threshold: float,
    elasticity_threshold: float,
) -> pd.DataFrame:
    rows = []
    group_columns = ["allocation_policy", "cost_model", "context_length"]
    for key, group in path.groupby(group_columns, sort=False):
        group = group.sort_values("budget_FLOPs").reset_index(drop=True)
        previous_elasticity = None
        for i in range(1, len(group)):
            before, after = group.iloc[i - 1], group.iloc[i]
            dlogc = math.log(after.budget_FLOPs / before.budget_FLOPs)
            elasticity_n = math.log(after.N_parameters / before.N_parameters) / dlogc
            elasticity_d = math.log(after.D_tokens / before.D_tokens) / dlogc
            share_tv = 0.5 * sum(
                abs(float(after[column]) - float(before[column]))
                for column in ("share_train", "share_quality", "share_attention")
            )
            reasons = []
            if before.quality_status != after.quality_status:
                reasons.append("quality_active_set_changed")
            if before.mix_id != after.mix_id:
                reasons.append("mixture_changed")
            if share_tv >= share_threshold:
                reasons.append("allocation_share_jump")
            elasticity_jump = np.nan
            if previous_elasticity is not None:
                elasticity_jump = max(
                    abs(elasticity_n - previous_elasticity[0]),
                    abs(elasticity_d - previous_elasticity[1]),
                )
                if elasticity_jump >= elasticity_threshold:
                    reasons.append("resource_elasticity_changed")
            previous_elasticity = (elasticity_n, elasticity_d)
            if reasons:
                rows.append({
                    "allocation_policy": key[0], "cost_model": key[1],
                    "context_length": key[2],
                    "budget_from": before.budget_FLOPs,
                    "budget_to": after.budget_FLOPs,
                    "geometric_transition_budget": math.sqrt(before.budget_FLOPs * after.budget_FLOPs),
                    "share_total_variation": share_tv,
                    "elasticity_N": elasticity_n,
                    "elasticity_D": elasticity_d,
                    "elasticity_jump": elasticity_jump,
                    "quality_status_from": before.quality_status,
                    "quality_status_to": after.quality_status,
                    "mix_from": before.mix_id,
                    "mix_to": after.mix_id,
                    "reason": ";".join(reasons),
                })
    return pd.DataFrame(rows)


def summarize_context_sensitivity(allocations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["allocation_policy", "cost_model", "budget_FLOPs"]
    for key, group in allocations.groupby(keys, sort=False):
        group = group.sort_values("context_length")
        reference = group.iloc[0]
        for row in group.itertuples(index=False):
            rows.append({
                "allocation_policy": key[0], "cost_model": key[1], "budget_FLOPs": key[2],
                "context_length": row.context_length,
                "context_relative_to_critical": row.context_length / L_CONTEXT_CRITICAL,
                "attention_to_training_ratio": row.attention_to_training_ratio,
                "attention_dominant": row.attention_dominant,
                "predicted_loss": row.predicted_loss,
                "loss_increase_vs_shortest_context": row.predicted_loss - reference.predicted_loss,
                "N_ratio_vs_shortest_context": row.N_parameters / reference.N_parameters,
                "D_ratio_vs_shortest_context": row.D_tokens / reference.D_tokens,
                "Q_change_vs_shortest_context": row.Q_opt - reference.Q_opt,
            })
    return pd.DataFrame(rows)


def chinese_font_setup() -> None:
    import matplotlib
    from matplotlib import font_manager

    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    font_path = next((path for path in candidates if path.exists()), None)
    if font_path is not None:
        font_manager.fontManager.addfont(str(font_path))
        name = font_manager.FontProperties(fname=str(font_path)).get_name()
        matplotlib.rcParams["font.sans-serif"] = [name, "Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


def make_plots(out: Path, budget_path: pd.DataFrame, context_sensitivity: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        chinese_font_setup()
        import matplotlib.pyplot as plt
    except Exception as error:
        print(f"[提示] 绘图库不可用，跳过图片：{error}", file=sys.stderr)
        return
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    contexts = sorted(budget_path.context_length.unique())
    chosen_context = min(contexts, key=lambda value: abs(value - 8192))
    subset = budget_path[
        (budget_path.allocation_policy == "reference")
        & (budget_path.context_length == chosen_context)
    ]
    colors = {"share_train": "#3B6EA8", "share_quality": "#C16E70", "share_attention": "#6B8E23"}
    labels = {"share_train": "基础训练", "share_quality": "质量提升", "share_attention": "长文本注意力"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, cost_name in zip(axes, COST_MODELS):
        group = subset[subset.cost_model == cost_name].sort_values("budget_FLOPs")
        for column in colors:
            ax.plot(group.budget_FLOPs, group[column], marker="o", label=labels[column], color=colors[column])
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_title(COST_MODELS[cost_name]["label"])
        ax.set_xlabel("总算力预算 C（FLOPs）")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("预算份额")
    axes[-1].legend(loc="best")
    fig.suptitle(f"固定参考配比下的资源份额路径（上下文长度 {chosen_context:g}）")
    fig.tight_layout()
    fig.savefig(fig_dir / "q3_fig01_resource_allocation_shares.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for cost_name in COST_MODELS:
        group = subset[subset.cost_model == cost_name].sort_values("budget_FLOPs")
        label = COST_MODELS[cost_name]["label"]
        axes[0].plot(group.budget_FLOPs, group.N_B, marker="o", label=label)
        axes[1].plot(group.budget_FLOPs, group.D_B, marker="o", label=label)
        axes[2].plot(group.budget_FLOPs, group.Q_opt, marker="o", label=label)
    for ax in axes:
        ax.set_xscale("log")
        ax.grid(alpha=0.25)
        ax.set_xlabel("总算力预算 C（FLOPs）")
    axes[0].set_yscale("log"); axes[1].set_yscale("log")
    axes[0].set_ylabel("参数量 N（B）"); axes[1].set_ylabel("训练数据 D（B tokens）")
    axes[2].set_ylabel("最优质量 Q")
    axes[2].legend(loc="best")
    fig.suptitle(f"最优参数量、数据量与质量路径（上下文长度 {chosen_context:g}）")
    fig.tight_layout()
    fig.savefig(fig_dir / "q3_fig02_optimal_N_D_Q_paths.pdf", bbox_inches="tight")
    plt.close(fig)

    if not context_sensitivity.empty:
        budget = sorted(context_sensitivity.budget_FLOPs.unique())[len(context_sensitivity.budget_FLOPs.unique()) // 2]
        group = context_sensitivity[
            (context_sensitivity.allocation_policy == "reference")
            & (context_sensitivity.cost_model == "exponential")
            & (context_sensitivity.budget_FLOPs == budget)
        ].sort_values("context_length")
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(group.context_length, group.predicted_loss, marker="o", color="#3B6EA8")
        ax1.axvline(L_CONTEXT_CRITICAL, color="#B55A4A", linestyle="--", label="解析临界长度 30000")
        ax1.set_xscale("log")
        ax1.set_xlabel("上下文长度 Lctx")
        ax1.set_ylabel("预测 Loss")
        ax1.grid(alpha=0.25)
        ax1.legend(loc="best")
        ax1.set_title(f"上下文长度对最优预测 Loss 的影响（C={budget:.0e}）")
        fig.tight_layout()
        fig.savefig(fig_dir / "q3_fig03_context_length_sensitivity.pdf", bbox_inches="tight")
        plt.close(fig)


def write_results_summary(
    out: Path,
    allocations: pd.DataFrame,
    transitions: pd.DataFrame,
    contexts: Sequence[int],
    problem2_output: Path,
) -> None:
    reference = allocations[allocations.allocation_policy == "reference"].copy()
    lines = [
        "# 问题三运行结果摘要",
        "",
        f"- 问题二接口：`{problem2_output}`",
        f"- C7 上下文长度：{', '.join(map(str, contexts))}",
        f"- 注意力与训练开销相当的临界长度：{L_CONTEXT_CRITICAL:.0f}",
        f"- 正式预算：{', '.join(f'{x:.0e}' for x in sorted(allocations.budget_FLOPs.unique()))}",
        "",
        "## 固定参考配比的代表性结果",
        "",
        "| 成本函数 | 预算 | 上下文 | N(B) | D(B tokens) | Q | Loss | 训练/质量/注意力份额 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in reference.sort_values(["cost_model", "budget_FLOPs", "context_length"]).itertuples(index=False):
        if row.context_length not in (min(contexts), max(contexts)):
            continue
        lines.append(
            f"| {row.cost_model_cn} | {row.budget_FLOPs:.0e} | {row.context_length:.0f} | "
            f"{row.N_B:.4g} | {row.D_B:.4g} | {row.Q_opt:.4f} | {row.predicted_loss:.5f} | "
            f"{row.share_train:.1%}/{row.share_quality:.1%}/{row.share_attention:.1%} |"
        )
    lines.extend([
        "",
        "## 结构性转移",
        "",
        f"代码按活跃约束变化、配比切换、预算份额总变差和资源弹性变化识别，共记录 {len(transitions)} 个候选转移区间。",
        "具体区间以 `structural_transitions.csv` 为准；网格识别给出的是区间，不应写成精确断点。",
        "",
        "## 解释边界",
        "",
        "- reference 是主结果：领域配比固定为前两问的参考配比。",
        "- candidates 只在前两问已经评价过的经验配方中选择，不等价于识别连续 17 维全局最优配比。",
        "- 超出 B1 的 N 或 D 会在 `evidence_flag` 中标记；高预算结果通常属于条件外推。",
        "- 三类质量成本参数来自题面附录 B，不是由附件观测反推得到。",
    ])
    (out / "results_summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="问题三：算力约束下 N-D-Q-p 联合优化与结构转移")
    parser.add_argument("--problem2-output", type=Path, default=None, help="问题二输出目录；默认自动选取最新有效目录")
    parser.add_argument(
        "--c7-path", type=Path,
        default=BASE.parent / "real_attachments" / "C_efficiency_evolution" / "model_architecture_metadata.csv",
        help="C7 架构元数据文件",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=BASE / "problem3_outputs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
    )
    parser.add_argument("--budgets", nargs="+", type=float, default=[1e19, 1e22, 1e24])
    parser.add_argument("--contexts", nargs="+", type=int, default=None, help="默认使用 C7 的全部不同上下文长度")
    parser.add_argument("--cost-models", nargs="+", choices=list(COST_MODELS), default=list(COST_MODELS))
    parser.add_argument("--mixture-mode", choices=["reference", "candidates", "both"], default="both")
    parser.add_argument("--max-mixtures", type=int, default=48)
    parser.add_argument("--budget-grid-points", type=int, default=15)
    parser.add_argument("--share-transition-threshold", type=float, default=0.10)
    parser.add_argument("--elasticity-transition-threshold", type=float, default=0.35)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if len(args.budgets) < 3:
        raise ValueError("题面要求至少考察三个不同量级的预算。")
    if any(value <= 0 for value in args.budgets):
        raise ValueError("预算必须为正数。")
    if args.max_mixtures < 1 or args.budget_grid_points < 5:
        raise ValueError("max-mixtures 至少为 1，budget-grid-points 至少为 5。")

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    problem2_output = find_latest_problem2_output(args.problem2_output)
    c7_path = args.c7_path.resolve()
    if not c7_path.is_file():
        raise FileNotFoundError(f"找不到 C7：{c7_path}")
    c7, c7_contexts = load_context_lengths(c7_path)
    contexts = sorted({int(value) for value in (args.contexts or c7_contexts)})
    unsupported = sorted(set(contexts) - set(c7_contexts))
    if unsupported:
        raise ValueError(f"以下上下文长度不在 C7 中：{unsupported}；C7 支持 {c7_contexts}")

    model, params, bounds = load_model(problem2_output)
    library, domain_columns = load_mixture_library(problem2_output, model, args.max_mixtures)
    policies = parse_policies(args.mixture_mode)
    print(f"[1/6] 问题二接口：{problem2_output}")
    print(f"[2/6] C7 上下文长度：{contexts}；解析临界值：{L_CONTEXT_CRITICAL:.0f}")
    print(f"[3/6] 配方库：参考配方 1 个，经验候选 {(library.allocation_policy == 'candidates').sum()} 个")

    cache: dict[tuple, dict] = {}

    def solve(policy, cost_name, context, budget):
        key = (policy, cost_name, int(context), float(budget))
        if key not in cache:
            cache[key] = optimize_scenario(
                library, policy, float(budget), float(context), cost_name, params, bounds
            )
        return cache[key]

    allocations = []
    for policy in policies:
        for cost_name in args.cost_models:
            for context in contexts:
                for budget in sorted(set(args.budgets)):
                    allocations.append(solve(policy, cost_name, context, budget))
    allocations = pd.DataFrame(allocations)
    save_csv(allocations, out / "optimal_allocations.csv")
    print(f"[4/6] 正式预算情景完成：{len(allocations)} 行")

    budget_grid = np.geomspace(min(args.budgets), max(args.budgets), args.budget_grid_points)
    path_rows = []
    for policy in policies:
        for cost_name in args.cost_models:
            for context in contexts:
                for budget in budget_grid:
                    path_rows.append(solve(policy, cost_name, context, float(budget)))
    budget_path = pd.DataFrame(path_rows)
    transitions = detect_budget_transitions(
        budget_path, args.share_transition_threshold, args.elasticity_transition_threshold
    )
    context_sensitivity = summarize_context_sensitivity(allocations)
    save_csv(budget_path, out / "budget_paths.csv")
    save_csv(transitions, out / "structural_transitions.csv")
    save_csv(context_sensitivity, out / "context_sensitivity.csv")

    used_mix_ids = sorted(set(allocations.mix_id) | set(budget_path.mix_id))
    selected_library = library[library.mix_id.isin(used_mix_ids)].copy()
    rename = {column: column.removeprefix(DOMAIN_PREFIX) for column in domain_columns}
    selected_library = selected_library.rename(columns=rename)
    save_csv(selected_library, out / "selected_mixtures.csv")

    cost_table = []
    q_reference = float(model["mixture_bridge"]["Q_reference"])
    for name, spec in COST_MODELS.items():
        cost_table.append({
            "cost_model": name, "cost_model_cn": spec["label"],
            "gamma": spec["gamma"], "lambda": spec["lambda"],
            "g_Q0": float(quality_cost_value(q_reference, name)),
            "g_1_minus_g_Q0": float(quality_cost_value(1.0, name) - quality_cost_value(q_reference, name)),
        })
    save_csv(cost_table, out / "quality_cost_models.csv")
    save_csv(
        c7.groupby("max_position_embeddings", as_index=False).size().rename(columns={"size": "model_count"}),
        out / "C7_context_distribution.csv",
    )
    print(f"[5/6] 结构性转移候选：{len(transitions)} 行")

    if not args.no_plots:
        make_plots(out, budget_path, context_sensitivity)
    write_results_summary(out, allocations, transitions, contexts, problem2_output)

    summary = {
        "problem2_output": problem2_output,
        "c7_path": c7_path,
        "contexts_from_C7": c7_contexts,
        "contexts_used": contexts,
        "attention_eta": ETA_ATTENTION,
        "critical_context_length": L_CONTEXT_CRITICAL,
        "budgets": sorted(set(args.budgets)),
        "budget_grid": budget_grid,
        "cost_models": COST_MODELS,
        "allocation_policies": policies,
        "mixture_candidates_retained": int((library.allocation_policy == "candidates").sum()),
        "transition_definition": {
            "share_total_variation_threshold": args.share_transition_threshold,
            "resource_elasticity_jump_threshold": args.elasticity_transition_threshold,
            "also_triggered_by": ["quality active-set change", "mixture candidate change"],
        },
        "scaling_parameters": params.__dict__,
        "evidence_bounds": bounds.__dict__,
        "output_rows": {
            "optimal_allocations": len(allocations),
            "budget_paths": len(budget_path),
            "structural_transitions": len(transitions),
        },
    }
    write_json(out / "analysis_summary.json", summary)
    print(f"[6/6] 完成，结果目录：{out}")


if __name__ == "__main__":
    configure_utf8()
    main(parse_args())
