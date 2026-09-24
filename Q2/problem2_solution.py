#!/usr/bin/env python3
"""问题二：从问题一输出构建 N-D-Q-p 广义标度律。

直接运行即可；只读取 B 附件与 Problem1 输出，不读取 A 原始数据。
依赖 numpy pandas scipy matplotlib；本机支持沿用 modeling-q1 Conda 环境。
默认生成带时间戳的新结果目录。详细假设及证据边界见配套 Markdown。
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def activate_local_environment():
    """仅在本项目已有环境存在时沿用它；不安装依赖、不修改系统环境。"""
    if sys.platform != "win32" or os.environ.get("PROBLEM2_ACTIVE") == "1":
        return
    env_dir = Path(r"D:\anaconda3\envs\modeling-q1")
    conda = env_dir.parent.parent / "Scripts" / "conda.exe"
    current = os.environ.get("CONDA_PREFIX", "")
    if current and Path(current).resolve() == env_dir.resolve():
        return
    if conda.is_file() and (env_dir / "python.exe").is_file():
        child = os.environ.copy()
        child["PROBLEM2_ACTIVE"] = "1"
        child["PYTHONIOENCODING"] = "utf-8"
        child["OPENBLAS_NUM_THREADS"] = "1"
        print("[启动] 沿用 modeling-q1 环境。", flush=True)
        result = subprocess.run([str(conda), "run", "--no-capture-output", "-p", str(env_dir),
                                 "python", "-u", str(Path(__file__).resolve()), *sys.argv[1:]], env=child)
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    activate_local_environment()

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parent
PARAM_NAMES = ["E", "A", "B", "alpha", "beta"]
FILES = {
    "B1": "pythia_training_log_existing.csv", "B2": "cerebras_training_log.csv",
    "B4": "scaling_baseline.csv", "B5": "published_scaling_data.csv",
    "B6": "supplementary_NQ_experiment.csv", "B7": "supplementary_NQ_experiment_expanded.csv",
    "B8": "supplementary_NQ_experiment_large.csv", "B9": "supplementary_large_models.csv",
    "B10": "supplementary_large_baseline.csv", "B11": "open_model_family_metadata.csv",
    "B12": "pythia_checkpoint_index.csv",
}
KINDS = {"B1": "observed", "B2": "semi_synthetic", "B3": "interpolated",
         "B4": "observed_cross_source", "B5": "published_cross_source", "B6": "semi_synthetic",
         "B7": "semi_synthetic_with_overlap", "B8": "semi_synthetic_mixed",
         "B9": "metadata_only", "B10": "estimated_loss", "B11": "metadata_only", "B12": "metadata_only"}


def save_csv(frame, path):
    pd.DataFrame(frame).to_csv(path, index=False, encoding="utf-8-sig")


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def json_write(path, value):
    path.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def metrics(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    if not len(y):
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "r2": np.nan, "spearman": np.nan, "bias": np.nan}
    residual = pred - y
    sst = np.sum((y - y.mean()) ** 2)
    rho = float(spearmanr(y, pred).statistic) if np.std(y) > 1e-12 and np.std(pred) > 1e-12 else np.nan
    return {"n": len(y), "rmse": np.sqrt(np.mean(residual**2)), "mae": np.mean(abs(residual)),
            "r2": 1 - np.sum(residual**2) / sst if sst > 0 else np.nan,
            "spearman": rho, "bias": residual.mean()}


def nd(frame):
    return frame.N_params_B.to_numpy(float), frame.D_tokens_B.to_numpy(float) / 100.0


def classic(theta, n, d):
    e, a, b, alpha, beta = theta
    return e + a * np.asarray(n, float)**(-alpha) + b * np.asarray(d, float)**(-beta)


def generalized(theta, gamma, n, d, q, q0, r=0.0, transfer=1.0, quality_scale=1.0):
    """q0 为当前配方的常规质量 Qbar(p)，可为数组；n=N_B，d=D_B/100。"""
    if np.any(np.asarray(n) <= 0) or np.any(np.asarray(d) <= 0):
        raise ValueError("N、D 必须为正数")
    if np.any((np.asarray(q) < 0) | (np.asarray(q) > 1)):
        raise ValueError("Q 必须在 [0,1] 内")
    factor = np.exp(gamma * quality_scale * (q0 - np.asarray(q)) + transfer * np.asarray(r))
    return theta[0] + (classic(theta, n, d) - theta[0]) * factor


def balanced_weights(frame):
    """每个模型规模、每个对数 D 区间均衡，避免密集后期检查点主导拟合。"""
    n, d = nd(frame)
    logd = np.log(d)
    bins = np.minimum(np.floor(8 * (logd - logd.min()) / max(np.ptp(logd), 1e-12)).astype(int), 7)
    # Bootstrap 重复抽中的轨迹仍是不同抽样簇，不能因按 N 合并而抵消重复次数。
    groups = frame["_cluster_id"].to_numpy() if "_cluster_id" in frame else n
    info = pd.DataFrame({"n": groups, "bin": bins})
    count = info.groupby(["n", "bin"]).n.transform("size").to_numpy()
    num_bins = info.groupby("n")["bin"].transform("nunique").to_numpy()
    w = 1.0 / (count * num_bins)
    return w / w.mean()


def fit_classic(frame, start=None, robust=True, multistart=True):
    n, d = nd(frame)
    y = frame.val_loss.to_numpy(float)
    w = np.sqrt(balanced_weights(frame))
    lower, upper = [0, 1e-8, 1e-8, .005, .005], [max(y.min() - 1e-6, .01), 30, 30, 2, 2]
    starts = [np.array(start)] if start is not None else []
    starts += [np.array([min(1.5, upper[0] * .7), .5, .3, x, z])
               for x, z in ([(.2, .2), (.5, .3), (.1, .7)] if multistart else [(.3, .3)])]
    fits = []
    for x in starts:
        result = least_squares(lambda t: w * (classic(t, n, d) - y),
                               np.clip(x, np.array(lower) + 1e-9, np.array(upper) - 1e-9),
                               bounds=(lower, upper), loss="soft_l1" if robust else "linear",
                               f_scale=.1, max_nfev=2500, x_scale="jac")
        if result.success and np.all(np.isfinite(result.x)):
            fits.append(result)
    if not fits:
        raise RuntimeError("经典标度律拟合未收敛")
    return min(fits, key=lambda f: f.cost)


def quality_predict(params, theta, frame, q0):
    e, a, b, gamma = params
    n, d = nd(frame)
    return e + (a*n**(-theta[3]) + b*d**(-theta[4])) * np.exp(gamma*(q0-frame.Q_score.to_numpy(float)))


def fit_quality(frame, theta, q0, signed=False, start=None, fixed_gamma=None):
    """质量校准允许独立 E_s、A_s、B_s；不把半合成绝对 Loss 并入 B1。"""
    y = frame.val_loss.to_numpy(float)
    lower = np.array([0, 1e-8, 1e-8, -8 if signed else 0.0])
    upper = np.array([20, 30, 30, 8.0])
    starts = [start] if start is not None else [[min(y.min()*.7, 1.5), .5, .2, g] for g in ([.3, 1, -.3] if signed else [.2, .8])]
    results = []
    for initial in starts:
        if fixed_gamma is None:
            fun = lambda t: quality_predict(t, theta, frame, q0) - y
            x = np.clip(initial, lower + 1e-9, upper - 1e-9)
            fit = least_squares(fun, x, bounds=(lower, upper), loss="soft_l1", f_scale=.05, max_nfev=2500, x_scale="jac")
        else:
            fun = lambda t: quality_predict([*t, fixed_gamma], theta, frame, q0) - y
            fit = least_squares(fun, np.clip(initial[:3], lower[:3]+1e-9, upper[:3]-1e-9),
                                bounds=(lower[:3], upper[:3]), loss="soft_l1", f_scale=.05,
                                max_nfev=2500, x_scale="jac")
            fit.x = np.r_[fit.x, fixed_gamma]
        if fit.success and np.all(np.isfinite(fit.x)):
            results.append(fit)
    if not results:
        raise RuntimeError("质量模型拟合未收敛")
    return min(results, key=lambda f: f.cost)


def fit_quality_free_exponents(frame, theta, qparams, q0):
    """检验 B1 的两个指数能否迁移；此备选模型只用于 B6 分组验证。"""
    n,d=nd(frame); y=frame.val_loss.to_numpy(float); q=frame.Q_score.to_numpy(float)
    initial=[*qparams[:3],*theta[3:],qparams[3]]
    result=least_squares(lambda t: t[0]+(t[1]*n**(-t[3])+t[2]*d**(-t[4]))*np.exp(t[5]*(q0-q))-y,
                         initial,bounds=([0,1e-8,1e-8,.005,.005,0],[20,30,30,2,2,8]),
                         loss="soft_l1",f_scale=.05,max_nfev=2500,x_scale="jac")
    if not result.success:
        raise RuntimeError("自由指数质量模型未收敛")
    return result


class MixtureBridge:
    """复用问题一二次 ILR 岭模型；额外质量项只度量相对本配方常规质量的改进。"""
    def __init__(self, root):
        spec = importlib.util.spec_from_file_location("problem1_reused", BASE.parent/"Problem1"/"problem1_solution.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.ilr = module.ilr_transform
        self.meta = json.loads((root/"problem2_interface.json").read_text(encoding="utf-8"))
        self.samples = pd.read_csv(root/"problem2_mixture_samples.csv")
        self.columns = self.meta["mixture_columns"]
        self.names = [c.replace("train_the_pile_", "") for c in self.columns]
        arrays = np.load(root/"problem2_mixture_model.npz", allow_pickle=False)
        self.model = module.QuadraticRidgeModel(float(arrays["alpha"]), arrays["x_mean"], arrays["x_std"], arrays["y_mean"], arrays["coef"])
        self.eps = self.meta["pseudocount"]
        mapping = {x["mixture_domain"]: x["quality_domain"] for x in self.meta["mapping"]}
        self.qvec = np.array([self.meta["domain_quality"].get(mapping.get(c), np.nan) for c in self.names])
        self.known = np.isfinite(self.qvec)
        self.qmid = np.where(self.known, self.qvec, self.meta["global_quality"])
        self.qmin, self.qmax = min(self.meta["domain_quality"].values()), max(self.meta["domain_quality"].values())
        train = self.samples[self.samples.dataset == "train_1m"]
        self.train_p = self.close(train[self.columns].to_numpy(float))
        self.p0 = self.train_p.mean(axis=0)
        self.q0 = float(self.p0 @ self.qmid)
        self.f0 = float(self.predict(self.p0)[0])

    @staticmethod
    def close(p):
        p = np.atleast_2d(np.asarray(p, float))
        if not np.isfinite(p).all() or np.any(p < 0) or np.any(p.sum(axis=1) <= 0):
            raise ValueError("配比必须非负且总量为正")
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, p):
        f = self.model.predict(self.ilr(self.close(p), self.eps)).mean(axis=1)
        if np.any(f <= 0):
            raise ValueError("问题一模型在该配比预测非正 Loss；已超出可用范围")
        return f

    def r(self, p):
        p = self.close(p)
        return np.log(self.predict(p)/self.f0)

    def quality(self, p):
        p = self.close(p)
        known = p[:, self.known] @ self.qvec[self.known]
        missing = 1-p[:, self.known].sum(axis=1)
        return known+missing*self.qmin, p@self.qmid, known+missing*self.qmax

    def metadata(self):
        return {"reference_p": dict(zip(self.names, self.p0)), "Q_reference": self.q0,
                "fA_reference": self.f0, "quality_reference_rule": "Qbar(p)=p dot domain_quality_mid",
                "mixture_transfer_lambda": 1.0, "lambda_status": "scenario_assumption_not_identifiable_from_disjoint_A_B",
                "Q_scale_status": "identity_bridge_assumption; sensitivity kappa=0.5,1,2"}


def load_data(root, out):
    data, audit, manifest = {}, [], []
    for label, name in FILES.items():
        path = root/name
        df = pd.read_csv(path)
        manifest.append({"label": label, "file": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        raw_n = len(df)
        required = [c for c in ["N_params_B", "D_tokens_B", "val_loss", "Q_score"] if c in df]
        for col in required:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        valid = np.ones(len(df), bool)
        for col in required:
            valid &= np.isfinite(df[col]) & (df[col] >= 0 if col == "Q_score" else df[col] > 0)
        if "Q_score" in df:
            valid &= df.Q_score <= 1
        data[label] = df[valid].copy().reset_index(drop=True)
        audit.append({"source": label, "evidence_type": KINDS[label], "raw_rows": raw_n,
                      "valid_rows": int(valid.sum()), "invalid_rows": int((~valid).sum())})
        if (~valid).any():
            save_csv(df[~valid], out/f"{label}_excluded_rows.csv")
    frames = []
    for path in sorted((root/"training_trajectories").glob("*.csv")):
        frames.append(pd.read_csv(path))
        manifest.append({"label": "B3", "file": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    data["B3"] = pd.concat(frames, ignore_index=True)
    audit.append({"source": "B3", "evidence_type": KINDS["B3"], "raw_rows": len(data["B3"]),
                  "valid_rows": len(data["B3"]), "invalid_rows": 0})
    keys = ["N_params_B", "D_tokens_B", "Q_score"]
    joined = data["B7"].merge(data["B6"][keys+["val_loss"]], on=keys, how="left", suffixes=("", "_B6"), validate="one_to_one")
    overlap = joined.val_loss_B6.notna()
    save_csv(joined[overlap], out/"B6_B7_overlap.csv")
    data["B7_new"] = joined[~overlap].drop(columns="val_loss_B6").copy()
    save_csv(audit, out/"data_audit.csv")
    json_write(out/"input_manifest.json", manifest)
    return data, {"B6_B7_overlap_rows": int(overlap.sum()), "B7_new_rows": int((~overlap).sum()),
                  "B6_B7_overlap_max_loss_difference": float(abs(joined.loc[overlap, "val_loss"]-joined.loc[overlap, "val_loss_B6"]).max())}


def quality_direction_audit(data):
    rows = []
    for label in ["B6", "B7_new", "B8"]:
        frame = data[label]
        for (n, d), group in frame.groupby(["N_params_B", "D_tokens_B"]):
            if group.Q_score.nunique() < 2:
                continue
            q, y = group.Q_score.to_numpy(), group.val_loss.to_numpy()
            slope = np.dot(q-q.mean(), y-y.mean()) / np.dot(q-q.mean(), q-q.mean())
            rows.append({"source": label, "N_params_B": n, "D_tokens_B": d,
                         "n": len(group), "slope_dL_dQ": slope,
                         "spearman_Q_L": spearmanr(q, y).statistic,
                         "quality_direction_conflict": bool(slope > 0)})
    return pd.DataFrame(rows)


def validate_baseline(data, theta, out):
    rows, predictions, fits = [], [], []
    b1 = data["B1"]
    for size in sorted(b1.N_params_B.unique()):
        train, test = b1[b1.N_params_B != size], b1[b1.N_params_B == size]
        fit = fit_classic(train, start=theta, multistart=False)
        pred = classic(fit.x, *nd(test))
        rows.append({"dataset": "B1_leave_one_N_out", "group": str(size), "evidence_type": "observed_group_holdout", **metrics(test.val_loss, pred)})
        fits.append({"held_N": size, **dict(zip(PARAM_NAMES, fit.x))})
        predictions.append(pd.DataFrame({"N_params_B": test.N_params_B, "D_tokens_B": test.D_tokens_B,
                                         "observed": test.val_loss, "predicted": pred}))
    combined = pd.concat(predictions, ignore_index=True)
    save_csv(combined, out/"B1_group_holdout_predictions.csv")
    save_csv(fits, out/"B1_leave_size_out_parameters.csv")
    rows.append({"dataset": "B1_leave_one_N_out_all", "group": "all", "evidence_type": "observed_group_holdout", **metrics(combined.observed, combined.predicted)})
    train_idx, test_idx = [], []
    for _, group in b1.groupby("N_params_B"):
        order = group.sort_values("D_tokens_B").index.to_numpy()
        cut = int(.7*len(order))
        train_idx.extend(order[:cut]); test_idx.extend(order[cut:])
    time_fit = fit_classic(b1.loc[train_idx], start=theta)
    time_test = b1.loc[test_idx]
    rows.append({"dataset": "B1_forward_last30pct", "group": "all", "evidence_type": "observed_time_holdout", **metrics(time_test.val_loss, classic(time_fit.x, *nd(time_test)))})
    for label in ["B1", "B2", "B3", "B4", "B5", "B10"]:
        df = data[label]
        pred = classic(theta, *nd(df))
        save_csv(df.assign(predicted_loss=pred, residual=pred-df.val_loss), out/f"{label}_predictions.csv")
        rows.append({"dataset": label, "group": "all", "evidence_type": KINDS[label], **metrics(df.val_loss, pred)})
        if label in ["B4", "B5"]:
            for family, group in df.groupby("family"):
                rows.append({"dataset": label, "group": family, "evidence_type": KINDS[label], **metrics(group.val_loss, classic(theta, *nd(group)))})
    return pd.DataFrame(rows)


def validate_quality(data, theta, qparams, q0, out, seed):
    b6 = data["B6"].copy()
    groups = list(b6.groupby(["N_params_B", "D_tokens_B"]).indices.values())
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(groups)), 5)
    rows = []
    for i, fold in enumerate(folds):
        test_ids = np.concatenate([groups[j] for j in fold])
        test, train = b6.iloc[test_ids], b6.drop(b6.index[test_ids])
        fit = fit_quality(train, theta, q0, start=qparams)
        null = fit_quality(train, theta, q0, start=qparams, fixed_gamma=0)
        for label, result in [("quality_exp", fit), ("without_quality", null)]:
            rows.append({"dataset": "B6_group_CV", "group": i, "model": label, "evidence_type": "semi_synthetic_group_holdout", **metrics(test.val_loss, quality_predict(result.x, theta, test, q0))})
        free=fit_quality_free_exponents(train,theta,fit.x,q0).x
        rows.append({"dataset":"B6_group_CV","group":i,"model":"free_exponents_quality",
                     "evidence_type":"semi_synthetic_group_holdout",
                     **metrics(test.val_loss,quality_predict([*free[:3],free[5]],free[:5],test,q0))})
    for label in ["B6", "B7_new", "B8"]:
        subsets = [("all", data[label])]
        if label == "B8":
            subsets += list(data[label].groupby("data_type"))
        for kind, frame in subsets:
            pred = quality_predict(qparams, theta, frame, q0)
            rows.append({"dataset": label, "group": kind, "model": "quality_exp", "evidence_type": "semi_synthetic", **metrics(frame.val_loss, pred)})
        save_csv(data[label].assign(predicted_loss=quality_predict(qparams, theta, data[label], q0)), out/f"{label}_quality_predictions.csv")
    signed = fit_quality(data["B8"], theta, q0, signed=True)
    json_write(out/"B8_signed_diagnostic.json", {"parameters": dict(zip(["E_s", "A_s", "B_s", "gamma"], signed.x)),
                                               "role": "direction_conflict_diagnostic_only",
                                               **metrics(data["B8"].val_loss, quality_predict(signed.x, theta, data["B8"], q0))})
    free=fit_quality_free_exponents(b6,theta,qparams,q0).x
    augmented=fit_quality(pd.concat([b6,data["B7_new"]],ignore_index=True),theta,q0,start=qparams).x
    json_write(out/"quality_sensitivity.json",{
        "B6_free_exponents_parameters":dict(zip(["E_s","A_s","B_s","alpha","beta","gamma"],free)),
        "B6_plus_B7_new_gamma":augmented[3],"role":"sensitivity_only_main_gamma_stays_B6"})
    return pd.DataFrame(rows)


def equivalence(theta, gamma, n, d, delta_q=.1):
    """固定 D,p：质量提升 delta_q 与模型扩容的精确等 Loss 关系。"""
    _, a, b, alpha, beta = theta
    x, y = a*n**(-alpha), b*d**(-beta)
    t = np.exp(-gamma*delta_q)
    rhs = t*(x+y)-y
    maximum = np.log1p(x/y)/gamma if gamma > 0 else np.inf
    if rhs <= 0:
        return {"equivalent_N_B": np.nan, "N_multiplier": np.nan,
                "status": "beyond_fixed_D_parameter_limit", "max_replaceable_delta_Q": maximum}
    log_multiplier = -np.log(rhs/x)/alpha
    if log_multiplier > 700:
        return {"equivalent_N_B": np.nan, "N_multiplier": np.nan,
                "status": "finite_but_numerically_unrepresentable", "max_replaceable_delta_Q": maximum}
    ratio = np.exp(log_multiplier)
    return {"equivalent_N_B": n*ratio, "N_multiplier": ratio,
            "status": "finite", "max_replaceable_delta_Q": maximum}


def substitution_table(theta, gamma, bridge, observed_limits):
    rows = []
    for n in [.1, 1., 7., 12., 70., 175.]:
        for tokens in [10., 100., 300., 1000.]:
            d = tokens/100
            x, y = theta[1]*n**(-theta[3]), theta[2]*d**(-theta[4])
            l = theta[0]+x+y
            rows.append({"N_params_B": n, "D_tokens_B": tokens, "Q": bridge.q0, "delta_Q": .1,
                         "L_before": l, "L_after_quality": theta[0]+(x+y)*np.exp(-.1*gamma),
                         "marginal_gain_N_per_B": theta[3]*x/n,
                         "marginal_gain_D_per_B": theta[4]*y/tokens,
                         "marginal_gain_Q": gamma*(x+y),
                         "elasticity_N_excess": theta[3]*x/(x+y),
                         "elasticity_D_excess": theta[4]*y/(x+y),
                         "elasticity_Q_excess": gamma*bridge.q0,
                         "elasticity_N_loss": theta[3]*x/l,
                         "elasticity_D_loss": theta[4]*y/l,
                         "elasticity_Q_loss": gamma*bridge.q0*(x+y)/l,
                         "local_dlnN_per_dQ": gamma*(x+y)/(theta[3]*x),
                         "cost_ratio_Q_per_unit_over_N_per_B_threshold": gamma*(x+y)/(theta[3]*x/n),
                         "within_B1_ND_ranges": bool(observed_limits[0] <= n <= observed_limits[1] and observed_limits[2] <= tokens <= observed_limits[3]),
                         **equivalence(theta, gamma, n, d)})
    return pd.DataFrame(rows)


def bootstrap(data, theta, qparams, q0, count, seed, out):
    """B1 按整条 N 轨迹重采样，B6 按整组 N,D 重采样；不逐点独立采样。"""
    rng = np.random.default_rng(seed)
    bgroups = [x for _, x in data["B1"].groupby("N_params_B")]
    qgroups = [x for _, x in data["B6"].groupby(["N_params_B", "D_tokens_B"])]
    rows, failures = [], []
    for i in range(count):
        try:
            bsample = pd.concat([bgroups[j].assign(_cluster_id=k) for k,j in enumerate(
                rng.integers(len(bgroups), size=len(bgroups)))], ignore_index=True)
            qsample = pd.concat([qgroups[j] for j in rng.integers(len(qgroups), size=len(qgroups))], ignore_index=True)
            bt = fit_classic(bsample, start=theta, multistart=False).x
            qt = fit_quality(qsample, bt, q0, start=qparams).x
            rows.append({"draw": i, **dict(zip(PARAM_NAMES, bt)), "gamma": qt[3],
                         **equivalence(bt, qt[3], 7., 3.)})
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            failures.append({"draw": i, "error": str(exc)})
        if (i+1) % 50 == 0:
            print(f"  不确定性估计 {i+1}/{count}", flush=True)
    result = pd.DataFrame(rows)
    save_csv(result, out/"bootstrap_draws.csv")
    json_write(out/"bootstrap_failures.json", failures)
    if count and len(rows) < .8*count:
        raise RuntimeError("Bootstrap 成功率不足 80%，请检查模型拟合")
    return result


def mixture_analysis(bridge, theta, gamma, out):
    samples = bridge.samples.copy()
    p = bridge.close(samples[bridge.columns].to_numpy(float))
    qlo, qm, qhi = bridge.quality(p)
    r = bridge.r(p)
    samples["Q_closed_lower"], samples["Q_closed_mid"], samples["Q_closed_upper"] = qlo, qm, qhi
    samples["mixture_log_relative_loss"] = r
    rows = []
    for lam in [0., .5, 1., 2.]:
        score = np.exp(lam*r)
        fixed_q_score = np.exp(gamma*(qm-bridge.q0)+lam*r)
        for label, idx in samples.groupby("dataset").groups.items():
            observed = samples.loc[idx, "observed_mean_loss"].to_numpy()
            rows.append({"dataset": label, "lambda": lam, "n": len(idx),
                         "spearman": float(spearmanr(observed, score[idx]).statistic) if lam else np.nan,
                         "fixed_Q_spearman": float(spearmanr(observed, fixed_q_score[idx]).statistic),
                         "role": "ranking_transfer_only_no_absolute_A_B_loss_pooling"})
    # 所有配方在同一个假设 N=7B,D=300B 情景评估；不把 A 的未知 D 补为该数值。
    for kappa in [.5, 1., 2.]:
        for lam in [0., .5, 1., 2.]:
            samples[f"scenario_L_quality_plus0p1_k{kappa:g}_lambda{lam:g}"] = generalized(theta, gamma, 7., 3., np.minimum(qm+.1,1), qm, r, lam, kappa)
    samples["scenario_L_usual_quality"] = generalized(theta, gamma, 7., 3., qm, qm, r)
    samples["scenario_L_mapping_low"] = generalized(theta, gamma, 7., 3., qhi, qm, r)
    samples["scenario_L_mapping_high"] = generalized(theta, gamma, 7., 3., qlo, qm, r)
    save_csv(samples, out/"mixture_scenarios.csv")
    save_csv(rows, out/"mixture_rank_validation.csv")
    p0 = bridge.p0
    base_loss = float(generalized(theta, gamma, 7., 3., bridge.q0, bridge.q0))
    effects, interactions = [], []
    delta = .0001
    for j, name in enumerate(bridge.names):
        direction = -p0/(1-p0[j]); direction[j] = 1
        plus, minus = p0+delta*direction, p0-delta*direction
        fixed_plus = generalized(theta, gamma, 7., 3., bridge.q0, plus@bridge.qmid, bridge.r(plus))[0]
        fixed_minus = generalized(theta, gamma, 7., 3., bridge.q0, minus@bridge.qmid, bridge.r(minus))[0]
        total_plus = generalized(theta, gamma, 7., 3., plus@bridge.qmid, plus@bridge.qmid, bridge.r(plus))[0]
        total_minus = generalized(theta, gamma, 7., 3., minus@bridge.qmid, minus@bridge.qmid, bridge.r(minus))[0]
        effects.append({"domain": name, "p_reference": p0[j], "Q_mid_assigned": bridge.qmid[j],
                        "Q_observed_mapping": bool(bridge.known[j]),
                        "dL_dp_fixed_Q": (fixed_plus-fixed_minus)/(2*delta),
                        "dL_dp_with_Q": (total_plus-total_minus)/(2*delta),
                        "delta_Q_per_share": float(direction@bridge.qmid)})
    for j in range(len(p0)):
        for k in range(j+1, len(p0)):
            # 两域共享固定供给池；四角使用相同方向，确保是真正的混合差分。
            donor = p0.copy(); donor[[j,k]] = 0; donor /= donor.sum()
            vj, vk = -donor.copy(), -donor.copy(); vj[j] = 1; vk[k] = 1
            corners = np.array([p0+delta*vj+delta*vk, p0+delta*vj-delta*vk,
                                p0-delta*vj+delta*vk, p0-delta*vj-delta*vk])
            for mode in ["fixed_Q", "with_Q"]:
                q = bridge.q0 if mode == "fixed_Q" else corners@bridge.qmid
                loss = generalized(theta, gamma, 7., 3., q, corners@bridge.qmid, bridge.r(corners))
                cross = float((loss[0]-loss[1]-loss[2]+loss[3])/(4*delta**2))
                interactions.append({"domain_j": bridge.names[j], "domain_k": bridge.names[k],
                                     "mode": mode, "mixed_derivative": cross,
                                     "interpretation": "local_complementarity" if cross < 0 else "local_substitution",
                                     "reference_loss": base_loss, "delta": delta})
    save_csv(effects, out/"domain_marginal_effects.csv")
    save_csv(interactions, out/"domain_interactions.csv")
    return pd.DataFrame(rows), pd.DataFrame(effects), pd.DataFrame(interactions)


def plot_results(out, data, theta, qparams, bridge, substitution, interactions):
    os.environ.setdefault("MPLCONFIGDIR",str(out/".matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 11, "savefig.dpi": 200})
    figdir = out/"figures"; figdir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10,6))
    for n, group in data["B1"].groupby("N_params_B"):
        group = group.sort_values("D_tokens_B")
        line, = ax.plot(group.D_tokens_B, classic(theta, *nd(group)), label=f"{n:.3g}B")
        ax.scatter(group.D_tokens_B, group.val_loss, color=line.get_color(), s=9, alpha=.35)
    ax.set(xscale="log", xlabel="训练数据量 D（十亿 tokens）", ylabel="验证 Loss", title="B1 真实轨迹与经典标度律拟合")
    ax.legend(ncol=4, fontsize=9); fig.tight_layout(); fig.savefig(figdir/"B1_scaling_fit.png"); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(12,5))
    for ax, label in zip(axes,["B6","B8"]):
        df = data[label]
        slopes = []
        for _, g in df.groupby(["N_params_B","D_tokens_B"]):
            g=g.sort_values("Q_score")
            if len(g)>1:
                ax.plot(g.Q_score, g.val_loss-g.val_loss.mean(), color="#237a91" if label=="B6" else "#c46744",alpha=.15)
        ax.set(xlabel="质量 Q", ylabel="组内中心化 Loss", title=f"{label} 半合成数据的质量方向")
    fig.tight_layout(); fig.savefig(figdir/"quality_direction_conflict.png"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5.5))
    for tokens, group in substitution.groupby("D_tokens_B"):
        finite=group[group.status=="finite"]
        ax.plot(finite.N_params_B, finite.N_multiplier, marker="o",label=f"D={tokens:g}B")
    ax.set(xscale="log",yscale="log",xlabel="原参数量 N（十亿）",ylabel="等效参数倍数",title="质量提高 0.1 的等效扩容（固定 D 与配比）")
    ax.legend(); fig.tight_layout(); fig.savefig(figdir/"quality_parameter_equivalence.png"); plt.close(fig)
    frame=interactions[interactions["mode"]=="with_Q"]
    mat=np.zeros((len(bridge.names),len(bridge.names)))
    for row in frame.itertuples():
        j,k=bridge.names.index(row.domain_j),bridge.names.index(row.domain_k)
        mat[j,k]=mat[k,j]=row.mixed_derivative
    bound=max(np.percentile(abs(mat[np.triu_indices(len(mat),1)]),95),1e-9)
    fig,ax=plt.subplots(figsize=(11,9))
    im=ax.imshow(mat,cmap="RdBu_r",vmin=-bound,vmax=bound)
    ax.set_xticks(range(len(mat)),bridge.names,rotation=60,ha="right",fontsize=8)
    ax.set_yticks(range(len(mat)),bridge.names,fontsize=8)
    ax.set_title("参考配比附近的领域混合偏导（常规质量路径）")
    fig.colorbar(im,ax=ax,label="负值：局部互补；正值：局部替代",shrink=.75,extend="both")
    fig.text(.5,.005,"颜色在绝对值第 95 百分位处饱和；原始数值见 CSV。",ha="center",fontsize=9)
    fig.tight_layout(); fig.savefig(figdir/"domain_interactions.png"); plt.close(fig)


def find_problem1_output(explicit):
    if explicit is not None:
        candidates = [explicit]
    else:
        root = BASE.parent/"Problem1"/"problem1_outputs"
        candidates = sorted(root.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)
    for p in candidates:
        required = ["analysis_summary.json", "problem2_interface.json", "problem2_mixture_samples.csv", "problem2_mixture_model.npz"]
        if all((p/x).is_file() for x in required):
            if not json.loads((p/"analysis_summary.json").read_text(encoding="utf-8")).get("formal_run"):
                continue
            return p.resolve()
    raise FileNotFoundError("缺少问题一全量接口输出。请先运行更新后的 Problem1/problem1_solution.py，再运行问题二。")


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,default=BASE.parent/"real_attachments"/"B_scaling_laws")
    parser.add_argument("--problem1-output",type=Path)
    parser.add_argument("--output-dir",type=Path,default=BASE/"problem2_outputs"/datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    parser.add_argument("--bootstrap",type=int,default=200)
    parser.add_argument("--seed",type=int,default=20260924)
    parser.add_argument("--no-plots",action="store_true")
    return parser.parse_args()


def main(args):
    if args.bootstrap < 0:
        raise ValueError("--bootstrap 不得为负")
    out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    p1=find_problem1_output(args.problem1_output)
    print(f"[1/7] 读取问题一输出：{p1}",flush=True)
    bridge=MixtureBridge(p1)
    data,overlap=load_data(args.data_root.resolve(),out)
    manifest=json.loads((out/"input_manifest.json").read_text(encoding="utf-8"))
    for name in ["analysis_summary.json","problem2_interface.json","problem2_mixture_samples.csv","problem2_mixture_model.npz"]:
        file=p1/name
        manifest.append({"label":"Problem1_output","file":str(file),"sha256":hashlib.sha256(file.read_bytes()).hexdigest()})
    for file in [Path(__file__),BASE.parent/"Problem1"/"problem1_solution.py"]:
        manifest.append({"label":"code","file":str(file.resolve()),"sha256":hashlib.sha256(file.read_bytes()).hexdigest()})
    json_write(out/"input_manifest.json",manifest)
    audit=quality_direction_audit(data); save_csv(audit,out/"quality_direction_audit.csv")
    print("[2/7] B1 拟合、按规模留出验证及时间外推验证。",flush=True)
    fit=fit_classic(data["B1"]); theta=fit.x
    validation=validate_baseline(data,theta,out); save_csv(validation,out/"baseline_validation.csv")
    print("[3/7] B6 质量校准、B7 去重验证、B8 方向冲突检验。",flush=True)
    qfit=fit_quality(data["B6"],theta,bridge.q0); qparams=qfit.x; gamma=qparams[3]
    qvalidation=validate_quality(data,theta,qparams,bridge.q0,out,args.seed)
    save_csv(qvalidation,out/"quality_validation.csv")
    print("[4/7] 配比迁移、边际效用、领域互补与质量扩容替代。",flush=True)
    rankval,effects,interactions=mixture_analysis(bridge,theta,gamma,out)
    limits=[data["B1"].N_params_B.min(),data["B1"].N_params_B.max(),data["B1"].D_tokens_B.min(),data["B1"].D_tokens_B.max()]
    substitution=substitution_table(theta,gamma,bridge,limits); save_csv(substitution,out/"quality_parameter_substitution.csv")
    save_csv([{"quality_scale_kappa":k,"gamma_effective":gamma*k,
               **equivalence(theta,gamma*k,7.,3.)} for k in [.5,1.,2.]],out/"quality_scale_sensitivity.csv")
    print(f"[5/7] 分组 Bootstrap，计划 {args.bootstrap} 次。",flush=True)
    draws=bootstrap(data,theta,qparams,bridge.q0,args.bootstrap,args.seed,out)
    intervals=[]
    for i,name in enumerate(PARAM_NAMES+["gamma"]):
        ci=np.quantile(draws[name],[.025,.975]) if len(draws) else [np.nan,np.nan]
        intervals.append({"parameter":name,"estimate":theta[i] if i<5 else gamma,"ci95_low":ci[0],"ci95_high":ci[1],
                          "interval_type":"conditional_cluster_bootstrap"})
    save_csv(intervals,out/"parameter_estimates.csv")
    print("[6/7] B9/B10 大模型情景与模型交付。",flush=True)
    big=data["B9"].copy()
    big["scenario_loss_Qref_p0"]=classic(theta,*nd(big))
    big["scenario_loss_Qref_plus_0p1"]=generalized(theta,gamma,*nd(big),min(bridge.q0+.1,1),bridge.q0)
    big["evidence_note"]="N,D metadata; both loss columns are scenario predictions, not observed"
    save_csv(big,out/"B9_large_model_scenarios.csv")
    model={"model":"L=E+(A*n^(-alpha)+B*d^(-beta))*exp(gamma*kappa*(Qbar(p)-Q)+lambda*r(p))",
           "units":{"n":"N_params_B","d":"D_tokens_B / 100","raw_N":"n * 1e9","raw_D":"d * 1e11"},
           "baseline_parameters":dict(zip(PARAM_NAMES,theta)),"gamma":gamma,
           "quality_source_nuisance":dict(zip(["E_B6","A_B6","B_B6"],qparams[:3])),
           "mixture_bridge":bridge.metadata(),"problem1_output":str(p1),
           "bootstrap_success":len(draws),"bootstrap_requested":args.bootstrap,
           "baseline_jacobian_condition":float(np.linalg.cond(fit.jac)),
           "quality_jacobian_condition":float(np.linalg.cond(qfit.jac)),
           "evidence_limits":["gamma is identified from semi-synthetic B6 only",
                              "B8 shows opposite quality direction and is not pooled into primary estimation",
                              "lambda and cross-source quality scale are assumptions, not jointly identified",
                              "B2 semi-synthetic; B3 interpolated; B10 estimated; none is independent real-world evidence",
                              "B4/B5 unknown tokenizers and validation corpora limit absolute loss comparability"], **overlap}
    json_write(out/"generalized_model.json",model)
    summary={**model,"B1_fit":metrics(data["B1"].val_loss,classic(theta,*nd(data["B1"]))),
             "quality_direction_conflicts":audit.groupby("source").quality_direction_conflict.agg(["sum","count","mean"]).reset_index().to_dict(orient="records"),
             "B1_N_range":[data["B1"].N_params_B.min(),data["B1"].N_params_B.max()],
             "B1_D_range":[data["B1"].D_tokens_B.min(),data["B1"].D_tokens_B.max()],
             "reference_equivalence_N7_D300":equivalence(theta,gamma,7.,3.),
             "reference_equivalence_bootstrap_finite_fraction":float((draws.status=="finite").mean()) if len(draws) else None,
             "reference_equivalence_bootstrap_ci95":np.quantile(draws.loc[draws.status=="finite","N_multiplier"],[.025,.975]).tolist() if len(draws) and (draws.status=="finite").any() else None}
    json_write(out/"analysis_summary.json",summary)
    print("[7/7] 输出图形与可读结果摘要。",flush=True)
    if not args.no_plots:
        plot_results(out,data,theta,qparams,bridge,substitution,interactions)
    write_result_report(out,summary,validation,qvalidation,rankval,intervals)
    print(json.dumps(clean_json({"output_dir":str(out),"parameters":model["baseline_parameters"],"gamma":gamma,
                                 "equivalence":summary["reference_equivalence_N7_D300"],"overlap":overlap}),ensure_ascii=False,indent=2),flush=True)


def write_result_report(out,summary,validation,qvalidation,rankval,intervals):
    def table(df, columns):
        lines=["| "+" | ".join(columns)+" |", "| "+" | ".join(["---"]*len(columns))+" |"]
        for row in df[columns].itertuples(index=False,name=None):
            lines.append("| "+" | ".join((f"{v:.6g}" if np.isfinite(v) else "未定义")
                                        if isinstance(v,(float,np.floating)) else str(v) for v in row)+" |")
        return "\n".join(lines)
    eq=summary["reference_equivalence_N7_D300"]
    text=["# 问题二本次计算结果", "", f"问题一输入目录：`{summary['problem1_output']}`。",
          "本页由代码生成；模型公式与证据边界以配套说明为准。", "", "## 参数估计", "",
          table(pd.DataFrame(intervals),["parameter","estimate","ci95_low","ci95_high"]), "",
          "区间为给定模型与跨源假设下的分组 Bootstrap 区间，不涵盖半合成生成机制误差。", "",
          "## 标度律验证", "",
          table(validation[validation.group=="all"],["dataset","evidence_type","n","rmse","r2","spearman"]), "",
          "B1 的全量拟合成绩不是独立验证；B2 为半合成，B3 为插值，B10 为估算。B4/B5 仅检验直接迁移，不使用检验标签校准。", "",
          "## 质量验证", "",table(qvalidation,["dataset","group","model","n","rmse","r2"]), "",
          f"B7 与 B6 重合 {summary['B6_B7_overlap_rows']} 条，只有 {summary['B7_new_rows']} 条作为新增质量水平验证。", "",
          "## 数据冲突", "",table(pd.DataFrame(summary["quality_direction_conflicts"]),["source","sum","count","mean"]), "",
          "sum 表示固定 N、D 后质量升高伴随 Loss 上升的组数。B8 不进入主质量参数拟合。", "",
          "## 质量提高 0.1 的替代结果", "",
          f"参考情景 N=7B、D=300B、Q={summary['mixture_bridge']['Q_reference']:.6f}、p=p0。",
          f"精确等效扩容倍数：{eq['N_multiplier']:.6g}；等效参数量：{eq['equivalent_N_B']:.6g}B。" if eq["status"]=="finite" else f"结果：{eq['status']}。",
          f"Bootstrap 倍数区间（仅有限解）：{summary['reference_equivalence_bootstrap_ci95']}。",
          f"Bootstrap 有限解比例：{summary['reference_equivalence_bootstrap_finite_fraction']}。", "",
          "此替代结果以 D 与 p 固定、B6 质量规律可迁移为前提；不是实际训练收益承诺。", "",
          "## 配比排序迁移", "",table(rankval,["dataset","lambda","n","spearman"]), "",
          "主列 spearman 沿 Q=Qbar(p) 的常规质量路径计算；lambda>0 时排序不随其数值改变，因此排序不能识别迁移强度。CSV 另报告固定 Q 路径的排序。", "",
          "## 可引用来源", "",
          "- Hoffmann et al. (2022), [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556).",
          "- Liu et al. (2024), [RegMix: Data Mixture as Regression for Language Model Pre-training](https://arxiv.org/abs/2407.01492).",
          "- 数值结论来自赛题本地附件及问题一输出。"]
    (out/"results_summary.md").write_text("\n".join(text)+"\n",encoding="utf-8")


if __name__=="__main__":
    main(parse_args())
