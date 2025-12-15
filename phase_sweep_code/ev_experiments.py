"""
Experiment utilities for the EV Stag Hunt model.
Optimized for robustness and cleaner parallel execution in Notebooks.
STRICTLY WINDOWS-SAFE: Uses ThreadPoolExecutor only.
Includes Progress Tracking via TQDM.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Callable, Dict, Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm  # <--- Added for progress tracking

# Import optimized core
from ev_core import EVStagHuntModel

# -----------------------------
# Policy factories
# -----------------------------

def policy_subsidy_factory(start: int, end: int, delta_a0: float = 0.3, delta_beta_I: float = 0.0) -> Callable:
    def policy(model, step):
        if not hasattr(model, "_base_a0"):
            model._base_a0 = model.a0
        if not hasattr(model, "_base_beta_I"):
            model._base_beta_I = model.beta_I

        if start <= step < end:
            model.a0 = model._base_a0 + delta_a0
            model.beta_I = model._base_beta_I + delta_beta_I
        else:
            model.a0 = model._base_a0
            model.beta_I = model._base_beta_I
    return policy

# -----------------------------
# Trial Runners
# -----------------------------

def run_timeseries_trial(
    T: int = 200,
    scenario_kwargs: Optional[Dict] = None,
    seed: Optional[int] = None,
    policy: Optional[Callable] = None,
    strategy_choice_func: str = "imitate",
    tau: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    
    defaults = {
        "a0": 2.0, "ratio": None, "beta_I": 3.0, "b": 1.0, "g_I": 0.1, "I0": 0.05,
        "network_type": "random", "n_nodes": 100, "p": 0.05, "m": 2, "collect": True,
        "X0_frac": 0.0, "init_method": "random",
    }
    scenario = defaults.copy()
    if scenario_kwargs: scenario.update(scenario_kwargs)

    # Calculate a0 from ratio if needed
    if scenario.get("ratio") is not None:
        calc_a0 = float(scenario["ratio"]) * float(scenario["b"]) - float(scenario["beta_I"]) * float(scenario["I0"])
    else:
        calc_a0 = scenario["a0"]

    initial_ev = int(scenario["n_nodes"] * scenario.get("X0_frac", 0.0))
    
    model = EVStagHuntModel(
        initial_ev=initial_ev,
        a0=calc_a0,
        beta_I=scenario["beta_I"],
        b=scenario["b"],
        g_I=scenario["g_I"],
        I0=scenario["I0"],
        seed=seed,
        network_type=scenario["network_type"],
        n_nodes=scenario["n_nodes"],
        p=scenario["p"],
        m=scenario["m"],
        collect=True,
        strategy_choice_func=strategy_choice_func,
        tau=tau,
    )

    for t in range(T):
        if policy is not None: policy(model, t)
        model.step()

    df = model.datacollector.get_model_vars_dataframe().copy()
    df["time"] = df.index
    return df["X"].to_numpy(), df["I"].to_numpy(), df

# -----------------------------
# Threaded Helpers
# -----------------------------

def _ts_worker(kwargs):
    return run_timeseries_trial(**kwargs)

def collect_intervention_trials(
    n_trials: int = 10,
    T: int = 200,
    scenario_kwargs: Optional[Dict] = None,
    subsidy_params: Optional[Dict] = None,
    max_workers: int = 1,
    seed_base: int = 42,
    strategy_choice_func: str = "imitate",
    tau: float = 1.0,
) -> Tuple[List, List, List, List, pd.DataFrame, pd.DataFrame]:
    
    sub_spec = subsidy_params or {"start": 30, "end": 80, "delta_a0": 0.3}
    baseline_tasks = []
    
    for i in range(n_trials):
        seed = seed_base + i
        baseline_tasks.append({
            "T": T, "scenario_kwargs": scenario_kwargs, "seed": seed,
            "policy": None, "strategy_choice_func": strategy_choice_func, "tau": tau
        })

    baseline_X, baseline_I, subsidy_X, subsidy_I = [], [], [], []

    # STRICTLY ThreadPool for Windows compatibility
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        # Baseline
        fut_b = {ex.submit(_ts_worker, args): i for i, args in enumerate(baseline_tasks)}
        
        # Subsidy (create policy wrapper inside loop)
        def _sub_worker(args, p_params):
            args["policy"] = policy_subsidy_factory(**p_params)
            return run_timeseries_trial(**args)
            
        fut_s = {ex.submit(_sub_worker, args.copy(), sub_spec): i for i, args in enumerate(baseline_tasks)}

        results_b = [None] * n_trials
        results_s = [None] * n_trials
        
        # Add TQDM here as well for consistency
        total_tasks = len(fut_b) + len(fut_s)
        with tqdm(total=total_tasks, desc="Running Intervention Trials") as pbar:
            for f in as_completed(list(fut_b.keys()) + list(fut_s.keys())):
                if f in fut_b:
                    results_b[fut_b[f]] = f.result()
                else:
                    results_s[fut_s[f]] = f.result()
                pbar.update(1)

    baseline_X = [r[0] for r in results_b]
    baseline_I = [r[1] for r in results_b]
    subsidy_X  = [r[0] for r in results_s]
    subsidy_I  = [r[1] for r in results_s]

    def summarize(X_list):
        mat = np.vstack(X_list)
        return pd.DataFrame({
            "X_mean": mat.mean(axis=0),
            "X_q10": np.quantile(mat, 0.10, axis=0),
            "X_q90": np.quantile(mat, 0.90, axis=0),
        })

    return baseline_X, baseline_I, subsidy_X, subsidy_I, summarize(baseline_X), summarize(subsidy_X)

def traces_to_long_df(baseline_X, subsidy_X):
    rows = []
    for i, trace in enumerate(baseline_X):
        for t, val in enumerate(trace):
            rows.append({"group": "baseline", "trial": i, "time": t, "X": val})
    for i, trace in enumerate(subsidy_X):
        for t, val in enumerate(trace):
            rows.append({"group": "subsidy", "trial": i, "time": t, "X": val})
    return pd.DataFrame(rows)

# -----------------------------
# Phase Sweep (Threaded)
# -----------------------------

def _sweep_worker(kwargs):
    X0, ratio, scenario, T = kwargs["X0"], kwargs["ratio"], kwargs["scenario_kwargs"], kwargs["T"]
    
    # Calculate a0
    I0, b, beta_I = scenario.get("I0", 0.05), scenario.get("b", 1.0), scenario.get("beta_I", 2.0)
    a0 = ratio * b - beta_I * I0
    
    # Init model
    init_ev = int(round(X0 * scenario.get("n_nodes", 100)))
    model = EVStagHuntModel(
        initial_ev=init_ev, a0=a0, beta_I=beta_I, b=b, 
        g_I=scenario.get("g_I", 0.05), I0=I0, 
        network_type=scenario.get("network_type", "random"),
        n_nodes=scenario.get("n_nodes", 100),
        p=scenario.get("p", 0.05), m=scenario.get("m", 2),
        collect=False
    )
    for _ in range(T): model.step()
    return model.get_adoption_fraction()

def phase_sweep_df(
    X0_values, ratio_values, scenario_kwargs=None, 
    batch_size=5, T=200, max_workers=1, backend="thread"
) -> pd.DataFrame:
    
    if scenario_kwargs is None: scenario_kwargs = {}
    tasks = []
    
    for x0 in X0_values:
        for r in ratio_values:
            for _ in range(batch_size):
                tasks.append({
                    "X0": x0, "ratio": r, 
                    "scenario_kwargs": scenario_kwargs, "T": T
                })
    
    results = []
    # STRICTLY ThreadPool for Windows compatibility
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_sweep_worker, t): t for t in tasks}
        
        # <--- Progress Bar Added Here --->
        for f in tqdm(as_completed(futures), total=len(tasks), desc="Sweeping Phase Space"):
            task = futures[f]
            results.append({"X0": task["X0"], "ratio": task["ratio"], "X_final": f.result()})
            
    df = pd.DataFrame(results)
    return df.groupby(["X0", "ratio"])["X_final"].mean().reset_index()