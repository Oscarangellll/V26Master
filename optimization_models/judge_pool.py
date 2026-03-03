from __future__ import annotations

import os
import time
import multiprocessing as mp
from dataclasses import asdict
from typing import Any, Dict, List, Tuple, Optional

from config.case_config import CaseConfig
from scenario_models.price_model import PriceModel
from scenario_models.weather_model import WeatherModel

_WORKER: Dict[str, Any] = {} 

def _init_worker(
    case: CaseConfig,
    weather_model: WeatherModel,
    price_model: PriceModel,
    judge_seed: int,
    mip_gap: float,
):
    """
    Runs once per worker process.
    Builds judge OptimizationModel and stores it in _WORKER.
    """
    from config.scenario_config import ScenarioConfig
    from optimization_models.optimization_model import OptimizationModel

    scenario_cfg = ScenarioConfig(case, weather_model, price_model, scenarios=[judge_seed])
    m = OptimizationModel(case, scenario_cfg)
    m.build_model()

    m.model.setParam("MIPGap", float(mip_gap))
    m.model.setParam("Threads", 1)  # CRITICAL: one core per judge

    _WORKER["m"] = m
    _WORKER["seed"] = judge_seed  # Store seed for debugging

def _extract_first_stage(m) -> Dict[Tuple[str, Any], int]:
    sol: Dict[Tuple[str, Any], int] = {}
    for b in m.case.B:
        sol[("eta", b)] = int(round(m.eta[b].X))
    for h in m.case.H:
        for b in m.case.B:
            sol[("gamma_LT", (h, b))] = int(round(m.gamma_LT[h, b].X))
    for h in m.case.H:
        for b in m.case.B:
            for t in m.case.T:
                sol[("gamma_ST", (h, b, t))] = int(round(m.gamma_ST[h, b, t].X))
    return sol

def _solve_one(fix_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Called many times. Applies FixState bounds, optimizes, returns results.
    fix_payload = {"fixed": {...}, "ub": {...}}
    """
    from optimization_models.bound_manager import FixState, BoundManager  # your modules

    m = _WORKER["m"] 
    seed = _WORKER.get("seed", "?")
    fix = FixState(fixed=dict(fix_payload["fixed"]), ub=dict(fix_payload["ub"]))

    bm = BoundManager(m)
    try:
        bm.apply_persistent_state(fix)
        m.model.update()

        t0 = time.perf_counter()
        m.model.optimize()
        t1 = time.perf_counter()

        out = {
            "status": int(m.model.Status),
            "runtime": t1 - t0,
            "gap": float(getattr(m.model, "MIPGap", float("nan"))),
        }

        if m.model.SolCount == 0:
            print(f"[WORKER seed={seed}] INFEASIBLE: Status={m.model.Status}")
            print(f"  Fixed: {dict(fix.fixed)}")
            print(f"  UB: {dict(fix.ub)}")
            out["obj"] = float("inf")
            out["sol"] = None
            return out

        out["obj"] = float(m.model.ObjVal)
        out["sol"] = _extract_first_stage(m)
        return out
    finally:
        bm.restore()


def pick_workers(n_judges: int, cap: int = 12) -> int:
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    avail = int(slurm) if slurm else (os.cpu_count() or 1)
    return max(1, min(n_judges, avail, cap))


class JudgePool:
    """
    One worker per judge seed. Each worker holds its own Gurobi model in-memory.
    """

    def __init__(
        self,
        case: CaseConfig,
        weather_model: WeatherModel,
        price_model: PriceModel,
        judge_seeds: List[int],
        *,
        mip_gap_judges: float,
        cap_workers: int = 12,
        mp_start_method: str = "spawn",  # safest with Gurobi
    ):
        self.case = case
        self.weather_model = weather_model
        self.price_model = price_model
        self.judge_seeds = list(judge_seeds)
        self.mip_gap_judges = float(mip_gap_judges)
        self.cap_workers = int(cap_workers)
        self.mp_start_method = mp_start_method

        self._ctx = mp.get_context(self.mp_start_method) #returns a context object for multiprocessing with the specified start method
        self._pools: List[mp.pool.Pool] = []
        self._started = False

    def start(self):
        if self._started:
            return

        # We want exactly one process per judge seed, but maybe cap by cores.
        # Strategy: If judges > workers, we can still do it with fewer pools by batching,
        # BUT best is 1 worker per judge to reuse the built model.
        # Therefore: cap should be >= max judges you actually run per job, or accept batching.
        # Here we do: one pool per judge (lightweight) is NOT good.
        # Better: one pool with N processes, but then each process must know which judge it is.
        # Easiest: create N processes == len(judge_seeds) if feasible.

        n_j = len(self.judge_seeds)
        n_workers = pick_workers(n_j, cap=self.cap_workers)

        if n_workers < n_j:
            raise RuntimeError(
                f"Need >= #judges workers to keep one model per judge. "
                f"Got workers={n_workers} judges={n_j}. "
                f"Increase cap_workers or reduce judges per run."
            )

        # Single pool with n_j workers; initializer differs per worker is tricky.
        # So we build one pool PER JUDGE? No. Instead: build n_j pools of 1 process each (still ok up to 20).
        # With 20 judges, 20 pools is fine; overhead is small compared to MIP solves.

        for seed in self.judge_seeds:
            pool = self._ctx.Pool( 
                processes=1,
                initializer=_init_worker, 
                initargs=(self.case, self.weather_model, self.price_model, seed, self.mip_gap_judges),
            )
            self._pools.append(pool)

        self._started = True

    def close(self):
        for p in self._pools:
            p.close()
        for p in self._pools:
            p.join()
        self._pools.clear()
        self._started = False

    def solve_all(self, fix) -> List[Dict[str, Any]]:
        """
        Runs one solve on each judge-worker in parallel (one task per pool).
        Returns list of outputs in same order as judge_seeds.
        """
        if not self._started:
            raise RuntimeError("JudgePool not started. Call start() first.")
        payload = {"fixed": fix.fixed, "ub": fix.ub}
        asyncs = [p.apply_async(_solve_one, (payload,)) for p in self._pools]
        # Q: why do we not have any callback or async result handling? 
        # A: because we want to wait for all to finish and then gather results in order. 
        # apply_async returns AsyncResult objects, and we can call get() on them to retrieve results. 
        # By calling get() in the same order as judge_seeds, we ensure results are ordered correctly.
        return [a.get() for a in asyncs]