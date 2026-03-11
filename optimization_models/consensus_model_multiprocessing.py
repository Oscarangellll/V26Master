from __future__ import annotations

import time
import statistics
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from optimization_models.bound_manager import DecisionKey, FixState, BoundManager
from optimization_models.judge_pool import JudgePool

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
        scenario,
        judge_seeds_1scenario_each: List[int],
        *,
        mip_gap_judges: float = 0.01,
        cap_workers: int = 12,
        log: bool = True,
    ):
        self.case = case
        self.scenario = scenario
        
        self.judge_seeds = list(judge_seeds_1scenario_each)
        self.judges: List[Tuple[int]] = [(s,) for s in self.judge_seeds]
        
        self.mip_gap_judges = float(mip_gap_judges)
        self.cap_workers = cap_workers
        self.log = bool(log)
        print(f" using judge seeds: {self.judge_seeds}")
        self.pool = JudgePool(
            case=case,
            scenario=scenario,
            judge_seeds=self.judge_seeds,
            mip_gap_judges=mip_gap_judges,
            cap_workers=cap_workers,
            mp_start_method="spawn"
        )
        self.pool.start()

        # NOT Per-judge cache: judge -> { CacheKey -> JudgeSolveResult }
        self._solve_cache: Dict[CacheKey, Dict[Tuple[int], JudgeSolveResult]] = {}

        self._t0 = time.perf_counter()
        
    def close(self) -> None:
        if getattr(self, "pool", None) is not None:
            self.pool.close()
            self.pool = None

    def _now(self) -> float:
        return time.perf_counter() - self._t0

    # -------------------------
    # Core solve w/ caching
    # -------------------------

    def solve_all_judges(self, fix: FixState) -> Dict[Tuple[int], JudgeSolveResult]:
        ck = CacheKey.from_fixstate(fix)
        if ck in self._solve_cache:
            return self._solve_cache[ck]
        
        outs = self.pool.solve_all(fix)
        
        results: Dict[Tuple[int], JudgeSolveResult] = {}
        for seed, out in zip(self.judge_seeds, outs):
            judge = (seed,)
            if out["sol"] is None:
                raise RuntimeError(f"Judge {judge} infeasible. Status={out['status']}")
            results[judge] = JudgeSolveResult(
                obj=out["obj"],
                sol=out["sol"],
                status=out["status"],
                gap=out["gap"],
                runtime=out["runtime"],
            )
        self._solve_cache[ck] = results
        return results

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

    # ------------------------------------------------------------------
    # ETA one_base propagation
    # ------------------------------------------------------------------
    def _propagate_one_base_eta(self, fix: FixState, k_star: DecisionKey, v_star: int) -> None:
        """
        If case.one_base is enforced and we fix eta[b]=1, then all other eta[b']=0
        is implied and can be fixed immediately to avoid redundant solves.
        """
        if not getattr(self.case, "one_base", False):
            return
        group, b = k_star
        if group != "eta":
            return
        if int(v_star) != 1:
            return

        for b2 in self.case.B:
            if b2 != b:
                fix.apply_fix(("eta", b2), 0)
        print(f"Propagated one_base: fixed eta[{b}]=1 => all other eta=0")

    def _constraint_propagation_from_eta(self, fix: FixState) -> None:
        """
        If eta[b]=0, then all gamma_LT[h,b] and gamma_ST[h,b,t] must be 0.
        This prevents infeasibility when we later try to fix gamma variables
        for bases that have already been deselected.
        """
        for b in self.case.B:
            eta_key = ("eta", b)
            if fix.fixed.get(eta_key) == 0:
                # Base b is deselected; fix all charters in this base to 0
                for h in self.case.H:
                    gamma_lt_key = ("gamma_LT", (h, b))
                    gamma_st_keys = [("gamma_ST", (h, b, t)) for t in self.case.T]
                    
                    if gamma_lt_key not in fix.fixed:
                        fix.apply_fix(gamma_lt_key, 0)
                    
                    for gst_key in gamma_st_keys:
                        if gst_key not in fix.fixed:
                            fix.apply_fix(gst_key, 0)

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

            # Baseline solve on current persistent fix-state
            results = self.solve_all_judges(fix)
            stats = self.consensus_stats(results, remaining)

            # Pick critical candidates (exclude near-unanimous by max_p)
            crit = self.pick_critical(stats, fix=fix, min_p=min_p, max_p=max_p, top_k=top_k)

            if not crit:
                # Case A: crit is empty because EVERYTHING remaining is unanimous (p==1)
                if all(stats[k]["p"] >= 0.999 for k in remaining):
                    for k in remaining:
                        fix.apply_fix(k, int(stats[k]["maj"]))
                    if self.log:
                        print(
                            f"[{phase_name}] auto-fixed {len(remaining)} unanimous decisions; "
                            f"phase complete. t={self._now():.2f}s"
                        )
                    return fix

                # Case B: crit is empty because no variable meets thresholds (p too low, etc.)
                if self.log:
                    print(f"[{phase_name}] no critical vars found; using fallback on highest consensus. t={self._now():.2f}s")
                crit = sorted(remaining, key=lambda k: -stats[k]["p"])[:1]

            # Evaluate crit candidates and pick the best improvement
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
            
            # If we just fixed eta[b]=1, then all other eta must be 0.
            self._propagate_one_base_eta(fix, k_star, v_star)
            # Compute remaining AFTER propagation (may drop by >1)
            remaining_after = sum(1 for k in keys if k not in fix.fixed)
            
            if self.log:
                print(
                    f"[{phase_name}] it={it} fixed {k_star}={v_star} "
                    f"(maj={info_star['maj']} alt={info_star['alt']} "
                    f"{info_star['agg']}maj={info_star['score_maj']:.3f} "
                    f"{info_star['agg']}alt={info_star['score_alt']:.3f}) "
                    f"remaining={remaining_after} t={self._now():.2f}s"
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
        # Constraint propagation: if eta[b]=0, fix all gamma[*, b] to 0
        self._constraint_propagation_from_eta(fix)
        
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
        # Constraint propagation: if eta[b]=0, fix all gamma[*, b, *] to 0
        self._constraint_propagation_from_eta(fix)
        
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
    ):
        # IMPORTANT: stop pool first so master can use threads freely
        self.close()
        
        from optimization_models.optimization_model import OptimizationModel
        
        master = OptimizationModel(self.case, self.scenario, master_scenarios)
        master.build_model()
        master.model.setParam("MIPGap", float(mip_gap_master))
        master.model.setParam("Threads", 0)  # use all cores
        master.model.setParam("TimeLimit", 14400) #max 4 timer master solve

        # Apply persistent state directly on master
        bm = BoundManager(master)
        bm.apply_persistent_state(fix)
        master.model.update()
        master.model.optimize()
        
        if master.model.SolCount == 0:
            raise RuntimeError(f"Master solve: no solution. Status={master.model.Status}")
        
        return master

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
        min_p: float = 0.60,
        max_p: float = 0.99,
        aggregator: str = "mean",  # consider "median" if objectives noisy
        tighten_ub_st: bool = True,
        unanim_fix_zero_st: bool = True,
        mip_gap_master: float = 0.01,
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
        )

        return master, self._now()


# # =====================================================================
# # Example usage (inside your driver/CLI code)
# # =====================================================================
# if __name__ == "__main__":
#     # This block is illustrative; you likely have your own CLI harness.
#     from config.case_config import CaseConfig
#     from scenario_models.weather_model import WeatherModel
#     from scenario_models.price_model import PriceModel

#     case_path = "cases/tests/test01.yaml"
#     case = CaseConfig(case_path)

#     weather_model = WeatherModel()
#     price_model = PriceModel()

#     judge_seeds = [11, 22, 33, 44, 55]          # one scenario per judge
#     master_scenarios = judge_seeds[:]           # or a larger set

#     cm = ConsensusModel(
#         case,
#         judge_seeds_1scenario_each=judge_seeds,
#         weather_model=weather_model,
#         price_model=price_model,
#         mip_gap_judges=0.01,
#         log=True,
#     )

#     try :
#         master, fix, runtime = cm.optimize(
#         master_scenarios=master_scenarios,
#         eta_max_iters=50,
#         lt_max_iters=200,
#         top_k_eta=1,
#         top_k_lt=1,
#         min_p=0.6,
#         max_p=0.99,
#         aggregator="mean",
#         tighten_ub_st=True,
#         unanim_fix_zero_st=True,
#         mip_gap_master=0.002,
#     )
#     finally:
#         cm.close()

#     print("Master status:", master.model.Status)
#     print("Master obj:", master.model.ObjVal)
#     print("Fixed decisions:", len(fix.fixed))
#     print("UB tightened:", len(fix.ub))
#     print("Runtime:", runtime)