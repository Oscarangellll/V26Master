from __future__ import annotations

import time
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------
# Expected imports from your project (keep as-is / adjust paths)
# ---------------------------------------------------------------------
# from config.case_config import CaseConfig
# from config.scenario_config import ScenarioConfig
# from model.optimization_model import OptimizationModel
# from models import WeatherModel, PriceModel

DecisionKey = Tuple[str, Any]  # (group, key) where group in {"eta","gamma_LT","gamma_ST"}


# =====================================================================
# Fix state (persistent decisions + bounds)
# =====================================================================
@dataclass
class FixState:
    fixed: Dict[DecisionKey, int] = field(default_factory=dict)   # x == v
    ub: Dict[DecisionKey, int] = field(default_factory=dict)      # x <= ub

    def copy(self) -> "FixState":
        return FixState(fixed=dict(self.fixed), ub=dict(self.ub))

    def apply_fix(self, k: DecisionKey, v: int) -> None:
        v = int(v)
        self.fixed[k] = v
        # keep UB consistent if present
        if k in self.ub:
            self.ub[k] = min(self.ub[k], v)

    def apply_ub(self, k: DecisionKey, ub: int) -> None:
        ub = int(ub)
        if k in self.fixed:
            # if fixed, ub is redundant but must not contradict
            self.ub[k] = min(self.ub.get(k, ub), self.fixed[k])
        else:
            self.ub[k] = min(self.ub.get(k, ub), ub)

    def is_solution_consistent(self, sol: Dict[DecisionKey, int]) -> bool:
        for k, v in self.fixed.items():
            if sol.get(k, None) != v:
                return False
        for k, u in self.ub.items():
            sv = sol.get(k, None)
            if sv is None:
                continue
            if sv > u:
                return False
        return True


# =====================================================================
# Per-judge result + cache keying
# =====================================================================
@dataclass
class JudgeSolveResult:
    obj: float
    sol: Dict[DecisionKey, int]
    status: int
    gap: float
    runtime: float


@dataclass(frozen=True)
class CacheKey:
    """
    CacheKey is a normalized, hashable snapshot of the "state" imposed on a judge.
    Using this lets you cache both baseline and experiment solves cleanly.

    Note:
      - fixed and ub are sorted tuples for determinism.
      - This is more robust than "is_consistent" reuse when you do many experiments.
    """
    fixed_items: Tuple[Tuple[DecisionKey, int], ...]
    ub_items: Tuple[Tuple[DecisionKey, int], ...]

    @staticmethod
    def from_fixstate(fix: FixState) -> "CacheKey":
        fixed_items = tuple(sorted(((k, int(v)) for k, v in fix.fixed.items()), key=lambda kv: (kv[0][0], str(kv[0][1]))))
        ub_items = tuple(sorted(((k, int(v)) for k, v in fix.ub.items()), key=lambda kv: (kv[0][0], str(kv[0][1]))))
        return CacheKey(fixed_items=fixed_items, ub_items=ub_items)


# =====================================================================
# Bound manager (in-model updates + rollback)
# =====================================================================
class BoundManager:
    """
    Saves/restore bounds on a per-model basis to support fast experiments.
    Use patterns:

      bm = BoundManager(m)
      bm.apply_fix(("eta", b), 1)
      ... solve ...
      bm.restore()

    You can also:
      bm.apply_persistent_state(fixstate)  # apply fixed + ub
      bm.restore()  # restore to the state when bm was created (typically "current")
    """

    def __init__(self, m):
        self.m = m
        self._saved: Dict[DecisionKey, Tuple[float, float, Optional[float]]] = {}  # (LB,UB,Start)

    def _get_var(self, group: str, key: Any, *, strict: bool = True):
        try:
            vardict = getattr(self.m, group)
        except AttributeError:
            if strict:
                raise ValueError(f"Unknown variable group: {group}")
            return None

        try:
            return vardict[key]
        except KeyError:
            if strict:
                raise KeyError(f"Variable not found: {group}{key}")
            return None

    def _save_if_needed(self, dk: DecisionKey, *, strict: bool = True) -> None:
        if dk in self._saved:
            return
        group, key = dk
        var = self._get_var(group, key, strict=strict)
        if var is None:
            return
        self._saved[dk] = (var.LB, var.UB, var.Start)

    def apply_fix(self, dk: DecisionKey, value: int, *, strict: bool = True, use_start: bool = False) -> None:
        self._save_if_needed(dk, strict=strict)
        group, key = dk
        var = self._get_var(group, key, strict=strict)
        if var is None:
            return
        v = int(value)
        var.LB = v
        var.UB = v
        if use_start:
            var.Start = v

    def apply_ub(self, dk: DecisionKey, ub: int, *, strict: bool = True) -> None:
        self._save_if_needed(dk, strict=strict)
        group, key = dk
        var = self._get_var(group, key, strict=strict)
        if var is None:
            return
        var.UB = min(var.UB, int(ub))

    def apply_persistent_state(self, fix: FixState, *, strict: bool = True, use_start: bool = False) -> None:
        """
        Applies both fixed equalities and UB tightenings.
        This modifies model variables directly.
        """
        # Apply UB first, then equalities (equalities dominate anyway)
        for dk, u in fix.ub.items():
            self.apply_ub(dk, u, strict=strict)
        for dk, v in fix.fixed.items():
            self.apply_fix(dk, v, strict=strict, use_start=use_start)

    def restore(self) -> None:
        """Restore all saved variables to previous (LB,UB,Start)."""
        for (group, key), (lb, ub, st) in self._saved.items():
            var = getattr(self.m, group)[key]
            var.LB = lb
            var.UB = ub
            var.Start = st
        self._saved.clear()
        self.m.model.update()


# =====================================================================
# Consensus model
# =====================================================================
class ConsensusModel:
    """
    Implements:
      - One OptimizationModel per judge (one scenario each)
      - Update bounds in-place (B), cache by imposed state
      - Phase A: fix eta with critical-decision experiments
      - Phase B: fix gamma_LT with critical-decision experiments
      - Phase C: gamma_ST post-processing given A+B: unanim-0 fix + UB tightening
      - Phase D: master solve with all scenarios and persistent state
    """

    def __init__(
        self,
        case,
        judge_seeds_1scenario_each: List[int],
        weather_model,
        price_model,
        *,
        mip_gap_judges: float = 0.01,
        output_flag: int = 0,
        use_start: bool = True,
        strict: bool = True,
        log: bool = True,
    ):
        self.case = case
        self.weather_model = weather_model
        self.price_model = price_model

        self.judges: List[Tuple[int]] = [(s,) for s in judge_seeds_1scenario_each]
        self.mip_gap_judges = float(mip_gap_judges)
        self.output_flag = int(output_flag)
        self.use_start = bool(use_start)
        self.strict = bool(strict)
        self.log = bool(log)

        self.models: Dict[Tuple[int], Any] = {}  # judge -> OptimizationModel
        self._build_judge_models()

        # Per-judge cache: judge -> { CacheKey -> JudgeSolveResult }
        self.cache: Dict[Tuple[int], Dict[CacheKey, JudgeSolveResult]] = {j: {} for j in self.judges}

        self._t0 = time.perf_counter()

    # -------------------------
    # Construction / extraction
    # -------------------------
    def _build_judge_models(self) -> None:
        from config.scenario_config import ScenarioConfig
        from model.optimization_model import OptimizationModel

        for judge in self.judges:
            scenario_cfg = ScenarioConfig(self.case, self.weather_model, self.price_model, scenarios=list(judge))
            m = OptimizationModel(self.case, scenario_cfg)
            m.build_model()
            m.model.setParam("OutputFlag", self.output_flag)
            m.model.setParam("MIPGap", self.mip_gap_judges)
            self.models[judge] = m

    def _extract_first_stage(self, m) -> Dict[DecisionKey, int]:
        out: Dict[DecisionKey, int] = {}

        for b in m.case.B:
            out[("eta", b)] = int(round(m.eta[b].X))

        for h in m.case.H:
            for b in m.case.B:
                out[("gamma_LT", (h, b))] = int(round(m.gamma_LT[h, b].X))

        for h in m.case.H:
            for b in m.case.B:
                for t in m.case.T:
                    out[("gamma_ST", (h, b, t))] = int(round(m.gamma_ST[h, b, t].X))

        return out

    def _now(self) -> float:
        return time.perf_counter() - self._t0

    # -------------------------
    # Core solve w/ caching
    # -------------------------
    def solve_judge(self, judge: Tuple[int], fix: FixState) -> JudgeSolveResult:
        ck = CacheKey.from_fixstate(fix)
        cached = self.cache[judge].get(ck, None)
        if cached is not None:
            return cached

        m = self.models[judge]

        bm = BoundManager(m)
        try:
            bm.apply_persistent_state(fix, strict=self.strict, use_start=self.use_start)
            m.model.update()

            t0 = time.perf_counter()
            m.model.optimize()
            t1 = time.perf_counter()

            if m.model.SolCount == 0:
                raise RuntimeError(f"No solution for judge {judge}. Status={m.model.Status}")

            res = JudgeSolveResult(
                obj=float(m.model.ObjVal),
                sol=self._extract_first_stage(m),
                status=int(m.model.Status),
                gap=float(getattr(m.model, "MIPGap", float("nan"))),
                runtime=t1 - t0,
            )
            self.cache[judge][ck] = res
            return res
        finally:
            bm.restore()

    def solve_all_judges(self, fix: FixState) -> Dict[Tuple[int], JudgeSolveResult]:
        return {j: self.solve_judge(j, fix) for j in self.judges}

    # -------------------------
    # Stats helpers
    # -------------------------
    @staticmethod
    def _mode(vals: List[int]) -> int:
        counts: Dict[int, int] = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1 
        # tie-break: smallest value
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    @staticmethod
    def _second_mode(vals: List[int]) -> Optional[int]:
        counts: Dict[int, int] = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        if len(counts) <= 1:
            return None
        items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return items[1][0]

    @staticmethod
    def _frac(vals: List[int], candidate: int) -> float:
        return vals.count(candidate) / len(vals)

    def consensus_stats(
        self,
        results: Dict[Tuple[int], JudgeSolveResult],
        keys: Iterable[DecisionKey],
    ) -> Dict[DecisionKey, Dict[str, Any]]:
        stats: Dict[DecisionKey, Dict[str, Any]] = {}
        for k in keys:
            vals = [results[j].sol[k] for j in self.judges]
            maj = self._mode(vals)
            p = self._frac(vals, maj)
            sec = self._second_mode(vals)
            stats[k] = {"vals": vals, "maj": maj, "p": p, "sec": sec}
        return stats

    # -------------------------
    # Critical decision selection
    # -------------------------
    def pick_critical(
        self,
        stats: Dict[DecisionKey, Dict[str, Any]],
        *,
        unfixed_only: bool = True,
        fix: Optional[FixState] = None,
        min_p: float = 0.55,
        max_p: float = 0.95,
        top_k: int = 1,
    ) -> List[DecisionKey]:
        """
        Default heuristic: "highest consensus but not near-unanimous", tie-break on minority size.
        """
        candidates = []
        for k, s in stats.items():
            if unfixed_only and fix is not None and k in fix.fixed:
                continue
            p = float(s["p"])
            if p < min_p or p > max_p:
                continue
            vals = s["vals"]
            maj = s["maj"]
            minority = len(vals) - vals.count(maj) 
            candidates.append(((p, minority), k))

        candidates.sort(key=lambda x: (-x[0][0], -x[0][1])) 
        return [k for _, k in candidates[:top_k]] 

    # -------------------------
    # Contrafactual experiment (two-way compare)
    # -------------------------
    def choose_alt_value(self, vals: List[int], maj: int) -> int:
        """
        Defines what 'opposite' means:
        - Prefer runner-up (2nd mode).
        - If none: binary flip if {0,1}.
        - Else: clamp(maj-1, 0) as a safe fallback.
        """
        sec = self._second_mode(vals)
        if sec is not None:
            return int(sec)
        if maj in (0, 1):
            return int(1 - maj)
        return int(max(0, maj - 1))

    def experiment_fix_value(
        self,
        base_fix: FixState,
        k: DecisionKey,
        v: int,
    ) -> float:
        """
        Solve all judges with base_fix + (k=v) and return average objective.
        """
        exp_fix = base_fix.copy()
        exp_fix.apply_fix(k, v)
        res = self.solve_all_judges(exp_fix)
        return float(statistics.mean(r.obj for r in res.values()))

    def decide_by_experiment(
        self,
        base_fix: FixState,
        stats_for_k: Dict[str, Any],
        k: DecisionKey,
        *,
        aggregator: str = "mean",  # "mean" or "median"
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Runs EXP_MAJ and EXP_ALT, returns chosen value + diagnostic info.
        """
        vals = list(stats_for_k["vals"])
        maj = int(stats_for_k["maj"])
        alt = self.choose_alt_value(vals, maj)

        # EXPs
        exp_fix_maj = base_fix.copy()
        exp_fix_maj.apply_fix(k, maj)
        res_maj = self.solve_all_judges(exp_fix_maj)
        objs_maj = [r.obj for r in res_maj.values()]

        exp_fix_alt = base_fix.copy()
        exp_fix_alt.apply_fix(k, alt)
        res_alt = self.solve_all_judges(exp_fix_alt)
        objs_alt = [r.obj for r in res_alt.values()]

        if aggregator == "median":
            agg_maj = float(statistics.median(objs_maj))
            agg_alt = float(statistics.median(objs_alt))
        else:
            agg_maj = float(statistics.mean(objs_maj))
            agg_alt = float(statistics.mean(objs_alt))

        chosen = maj if agg_maj <= agg_alt else alt
        info = {
            "maj": maj,
            "alt": alt,
            "agg": aggregator,
            "score_maj": agg_maj,
            "score_alt": agg_alt,
        }
        return chosen, info

    # -------------------------
    # Phase A/B loops
    # -------------------------
    def _phase_loop(
        self,
        fix: FixState,
        keys: List[DecisionKey],
        *,
        phase_name: str,
        max_iters: int,
        top_k: int,
        min_p: float,
        max_p: float,
        aggregator: str,
    ) -> FixState:
        for it in range(1, max_iters + 1):
            # stop condition
            remaining = [k for k in keys if k not in fix.fixed]
            if not remaining:
                if self.log:
                    print(f"[{phase_name}] done (all fixed). it={it-1}, t={self._now():.2f}s")
                return fix

            results = self.solve_all_judges(fix)
            stats = self.consensus_stats(results, remaining)

            crit = self.pick_critical(stats, fix=fix, min_p=min_p, max_p=max_p, top_k=top_k)
            if not crit:
                # fallback: pick highest consensus among remaining
                crit = sorted(remaining, key=lambda k: -stats[k]["p"])[:1]

            # evaluate crit candidates and pick the best improvement
            best = None  # (best_score, k, chosen_val, info)
            for k in crit:
                chosen_val, info = self.decide_by_experiment(fix, stats[k], k, aggregator=aggregator)
                best_score = min(info["score_maj"], info["score_alt"])
                cand = (best_score, k, chosen_val, info)
                if best is None or cand[0] < best[0]:
                    best = cand

            assert best is not None
            _, k_star, v_star, info_star = best
            fix.apply_fix(k_star, v_star)

            if self.log:
                print(
                    f"[{phase_name}] it={it} fixed {k_star}={v_star} "
                    f"(maj={info_star['maj']} alt={info_star['alt']} "
                    f"{info_star['agg']}maj={info_star['score_maj']:.3f} "
                    f"{info_star['agg']}alt={info_star['score_alt']:.3f}) "
                    f"remaining={len(remaining)-1} t={self._now():.2f}s"
                )

        if self.log:
            print(f"[{phase_name}] reached max_iters={max_iters}. remaining={sum(1 for k in keys if k not in fix.fixed)}")
        return fix

    def fix_eta(
        self,
        fix: FixState,
        *,
        max_iters: int = 50,
        top_k: int = 1,
        min_p: float = 0.55,
        max_p: float = 0.95,
        aggregator: str = "mean",
    ) -> FixState:
        keys = [("eta", b) for b in self.case.B]
        return self._phase_loop(
            fix,
            keys,
            phase_name="ETA",
            max_iters=max_iters,
            top_k=top_k,
            min_p=min_p,
            max_p=max_p,
            aggregator=aggregator,
        )

    def fix_gamma_lt(
        self,
        fix: FixState,
        *,
        max_iters: int = 200,
        top_k: int = 1,
        min_p: float = 0.55,
        max_p: float = 0.95,
        aggregator: str = "mean",
    ) -> FixState:
        keys = [("gamma_LT", (h, b)) for h in self.case.H for b in self.case.B]
        return self._phase_loop(
            fix,
            keys,
            phase_name="GAMMA_LT",
            max_iters=max_iters,
            top_k=top_k,
            min_p=min_p,
            max_p=max_p,
            aggregator=aggregator,
        )

    # -------------------------
    # Phase C: gamma_ST tightening after ETA+LT fixed
    # -------------------------
    def fix_gamma_st_post(
        self,
        fix: FixState,
        *,
        tighten_ub: bool = True,
        unanim_fix_zero: bool = True,
    ) -> FixState:
        keys = [("gamma_ST", (h, b, t)) for h in self.case.H for b in self.case.B for t in self.case.T]

        # solve once with current fix, then post-process
        results = self.solve_all_judges(fix)
        stats = self.consensus_stats(results, keys)

        n_fix0 = 0
        n_ub = 0

        for k in keys:
            if k in fix.fixed:
                continue

            vals = stats[k]["vals"]
            mx = max(vals)

            # unanim 0 -> fix 0
            if unanim_fix_zero and mx == 0:
                fix.apply_fix(k, 0)
                n_fix0 += 1
                continue

            if tighten_ub:
                fix.apply_ub(k, mx)
                n_ub += 1

        if self.log:
            print(f"[GAMMA_ST] post: fixed0={n_fix0}, ub_tighten={n_ub}, t={self._now():.2f}s")
        return fix

    # -------------------------
    # Master solve
    # -------------------------
    def solve_master(
        self,
        fix: FixState,
        master_scenarios: List[int],
        *,
        mip_gap_master: float = 0.002,
        output_flag: int = 0,
        use_start: bool = True,
    ):
        from config.scenario_config import ScenarioConfig
        from model.optimization_model import OptimizationModel

        master_cfg = ScenarioConfig(self.case, self.weather_model, self.price_model, scenarios=master_scenarios)
        master = OptimizationModel(self.case, master_cfg)
        master.build_model()
        master.model.setParam("OutputFlag", int(output_flag))
        master.model.setParam("MIPGap", float(mip_gap_master))

        # Apply persistent state directly on master
        bm = BoundManager(master)
        try:
            bm.apply_persistent_state(fix, strict=True, use_start=use_start)
            master.model.update()
            master.model.optimize()
            print("status rett etter optimize inne i solve_master()=", master.model.Status)
            if master.model.SolCount == 0:
                raise RuntimeError(f"Master solve: no solution. Status={master.model.Status}")
            return master
        finally:
            # Not strictly needed (master is one-shot), but keeps pattern consistent
            bm.restore()

    # -------------------------
    # Full pipeline
    # -------------------------
    def optimize(
        self,
        master_scenarios: List[int],
        *,
        eta_max_iters: int = 50,
        lt_max_iters: int = 200,
        top_k_eta: int = 1,
        top_k_lt: int = 1,
        min_p: float = 0.55,
        max_p: float = 0.95,
        aggregator: str = "mean",  # consider "median" if objectives noisy
        tighten_ub_st: bool = True,
        unanim_fix_zero_st: bool = True,
        mip_gap_master: float = 0.002,
        output_flag_master: int = 0,
    ):
        fix = FixState()

        # A: ETA
        fix = self.fix_eta(
            fix,
            max_iters=eta_max_iters,
            top_k=top_k_eta,
            min_p=min_p,
            max_p=max_p,
            aggregator=aggregator,
        )

        # B: gamma_LT
        fix = self.fix_gamma_lt(
            fix,
            max_iters=lt_max_iters,
            top_k=top_k_lt,
            min_p=min_p,
            max_p=max_p,
            aggregator=aggregator,
        )

        # C: gamma_ST post-processing
        fix = self.fix_gamma_st_post(
            fix,
            tighten_ub=tighten_ub_st,
            unanim_fix_zero=unanim_fix_zero_st,
        )

        # D: master
        if self.log:
            print(f"[MASTER] solving with {len(master_scenarios)} scenarios. fixed={len(fix.fixed)} ub={len(fix.ub)} t={self._now():.2f}s")

        master = self.solve_master(
            fix,
            master_scenarios=master_scenarios,
            mip_gap_master=mip_gap_master,
            output_flag=output_flag_master,
            use_start=True,
        )

        return master, fix


# =====================================================================
# Example usage (inside your driver/CLI code)
# =====================================================================
if __name__ == "__main__":
    # This block is illustrative; you likely have your own CLI harness.
    from config.case_config import CaseConfig
    from models import WeatherModel, PriceModel

    case_path = "cases/tests/test01.yaml"
    case = CaseConfig(case_path)

    weather_model = WeatherModel()
    price_model = PriceModel()

    judge_seeds = [11, 22, 33, 44, 55]          # one scenario per judge
    master_scenarios = judge_seeds[:]           # or a larger set

    cm = ConsensusModel(
        case,
        judge_seeds_1scenario_each=judge_seeds,
        weather_model=weather_model,
        price_model=price_model,
        mip_gap_judges=0.01,
        output_flag=0,
        use_start=True,
        strict=True,
        log=True,
    )

    master, fix = cm.optimize(
        master_scenarios=master_scenarios,
        eta_max_iters=50,
        lt_max_iters=200,
        top_k_eta=1,
        top_k_lt=1,
        min_p=0.6,
        max_p=0.95,
        aggregator="mean",
        tighten_ub_st=True,
        unanim_fix_zero_st=True,
        mip_gap_master=0.002,
        output_flag_master=0,
    )

    print("Master status:", master.model.Status)
    print("Master obj:", master.model.ObjVal)
    print("Fixed decisions:", len(fix.fixed))
    print("UB tightened:", len(fix.ub))
