from collections import defaultdict
from dataclasses import dataclass
import math
import multiprocessing as mp
import signal
import time 

from config import ScenarioConfig
from optimization_models import OptimizationModel

def _judge(s, case, results_queue, fix_queue, sem):
     
    sem.acquire()
    
    scenario_cfg = ScenarioConfig(case, [s])
    model = OptimizationModel(case, scenario_cfg, [s], weights={s: 1.0})
    
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 0.04
    model.Params.Threads = 1
    model.Params.Timelimit = 1 * 3600

    sem.release()
    
    while True:

        fix = fix_queue.get()
        fix.apply_to(model)
        
        result = None
        try: 
            sem.acquire()
            try:
                model.optimize()
            finally:
                sem.release()

            if model.SolCount > 0:
                solution = {
                    (group, idx): int(var.X) 
                    for group in ["eta", "gamma_LT", "gamma_ST"] 
                    for idx, var in getattr(model, group).items()
                }
                result = JudgeSolveResult(
                    model.ObjVal,
                    solution,
                    GRB_STATUS[model.Status],
                    model.MIPGap,
                    model.Runtime
                )
        finally:   
            results_queue.put((s, fix, result))
            fix.remove_from(model)
            fix_queue.task_done()

DecisionKey = tuple[str, object] # (group, index), identifies a variable

class FixExperiment:
    def __init__(self, fixed: dict[DecisionKey, int]):
        self.fixed = fixed.copy()

    def extended(self, key: DecisionKey, val: int):
        fixed = self.fixed.copy()
        fixed[key] = val
        return FixExperiment(fixed) 

    def apply_to(self, model):
        for key, val in self.fixed.items():
            group, idx = key
            var = getattr(model, group)[idx]
            var.LB = val
            var.UB = val

    def remove_from(self, model):
        for key, val in self.fixed.items():
            group, idx = key 
            var = getattr(model, group)[idx]
            var.LB = 0
            var.UB = 1e100

    def __hash__(self):
        return hash(frozenset(self.fixed.items()))

    def __eq__(self, other):
        return self.fixed == other.fixed
    
    def is_subset_of(self, other_fix):
        for k, v in self.fixed.items():
            if k not in other_fix.fixed or other_fix.fixed[k] != v:
                return False
        return True 

class FixState:
    def __init__(self):
        self.fixed: dict[DecisionKey, int] = {}
        
        self.cache: dict[tuple[int, FixExperiment], JudgeSolveResult] = {}
    
    # Checks if a fix is in cache, and in addition insert the fix
    # in the cache if there is a consistent solution
    def in_cache(self, s_new, fix_new: FixExperiment):
        
        if (s_new, fix_new) in self.cache:
            return True
        
        for (s_old, fix_old), res in self.cache.items():
            
            if s_new != s_old:
                continue

            if not fix_old.is_subset_of(fix_new):
                continue
            
            consistent_solution = True
            for key, val in fix_new.fixed.items():
                if res.solution.get(key) != val:
                    consistent_solution = False
                    break

            if not consistent_solution:
                continue

            self.cache[(s_new, fix_new)] = res
            return True
        
        return False
        
class ConsensusModelMP:
    def __init__(self, case, scenario_ids, weights):
        self.case = case
        self.scenario_ids = scenario_ids
        self.weights = weights

        self.results_queue = mp.SimpleQueue()
        self.fix_queues = {s: mp.JoinableQueue() for s in self.scenario_ids}

        self.sem = mp.BoundedSemaphore(mp.cpu_count() - 1)

        self.judges = {}

        self.state = FixState()
        self.bounds = {}

        self.t0 = time.perf_counter()
        self.time_to_fix_eta = None 
        self.time_to_fix_gamma_LT = None
        self.time_to_tighten_gamma_ST = None
        self.total_consensus_time = None

        self.fix_iteration_summaries = []

        self.fix_and_bounds_time_limit = 3 * 3_600
        self.master_model_time_limit = 3_600

    @staticmethod
    def _safe_float(value):
        try:
            parsed = float(value)
        except Exception:
            return None
        if not math.isfinite(parsed):
            return None
        return parsed

    @staticmethod
    def _percentile(values, percentile):
        if not values:
            return None
        sorted_vals = sorted(values)
        idx = max(0, min(len(sorted_vals) - 1, math.ceil(percentile * len(sorted_vals)) - 1))
        return float(sorted_vals[idx])

    def optimize(self):
        def _alarm_handler(signum, frame):
            raise TimeoutError
        use_sigalrm = hasattr(signal, "SIGALRM")
        if use_sigalrm:
            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(self.fix_and_bounds_time_limit)
        
        try:
            for s in self.scenario_ids:
                self.judges[s] = mp.Process(
                    target=_judge, 
                    args=(
                        s, 
                        self.case, 
                        self.results_queue, 
                        self.fix_queues[s],
                        self.sem
                    )
            )
                self.judges[s].start()
            
            self.fix_eta()
            t1 = time.perf_counter()
            self.time_to_fix_eta = t1 - self.t0

            self.fix_gamma_LT()
            t2 = time.perf_counter()
            self.time_to_fix_gamma_LT = t2 - t1

            self.tighten_gamma_ST()
            t3 = time.perf_counter()
            self.time_to_tighten_gamma_ST = t3 - t2

        except TimeoutError:
            self._shutdown_judges()
        except KeyboardInterrupt:
            self._shutdown_judges()
            exit()
        else:
            self._shutdown_judges()
        finally:
            if use_sigalrm:
                signal.alarm(0)
        
        # Solve master with fixations and bounds
        scenario_cfg = ScenarioConfig(self.case, self.scenario_ids)
        master_model = OptimizationModel(self.case, scenario_cfg, self.scenario_ids, self.weights)

        fix = FixExperiment(self.state.fixed)
        fix.apply_to(master_model)

        for key, (lb, ub) in self.bounds.items():
            group, idx = key
            var = getattr(master_model, group)[idx]
            var.LB = lb
            var.UB = ub
    
        master_model.Params.OutputFlag = 0
        master_model.Params.MIPGap = 0.02
        master_model.Params.TimeLimit = self.master_model_time_limit
        
        master_model.optimize()

        self.master_model = master_model

        self.total_consensus_time = time.perf_counter() - self.t0
    
    def fix_eta(self):
        keys: set[DecisionKey] = {("eta", b) for b in self.case.B}
        
        self.fix("eta", keys) 

    def fix_gamma_LT(self):
        keys: set[DecisionKey] = {
            ("gamma_LT", (h, b)) 
            for h in self.case.H 
            for b in self.case.B
        }

        self.fix("gamma_LT", keys) 
    
    def tighten_gamma_ST(self):
        fix = FixExperiment(self.state.fixed)
        
        keys: set[DecisionKey] = {
            ("gamma_ST", (h, b, t))
            for h in self.case.H
            for b in self.case.B
            for t in self.case.T
        }
    
        # Solve all judges with the most recent fix

        to_solve = []

        # Cache or queue
        for s in self.scenario_ids:
            if not self.state.in_cache(s, fix):
                self.fix_queues[s].put(fix)
                to_solve.append(s)

        # Wait for judges
        # Wait for judges
        for s in to_solve:
            self.fix_queues[s].join()

        # Collect results
        for _ in to_solve:
            s, fix, res = self.results_queue.get()

            # Dont need to cache a None result
            if res is None:
                continue

            self.state.cache[s, fix] = res
        
        for key in keys:
            vals = []
            for s in self.scenario_ids:
                res = self.state.cache.get((s, fix))
                if res is None:
                    continue

                for k, v in res.solution.items():
                    if key == k:
                        vals.append(v)
            if vals: 
                self.bounds[key] = (min(vals), max(vals))
        
    def fix(self, group: str, keys: set[DecisionKey]):
        iter_idx = 0

        # Remove keys when they become fixed. Loop exits when all keys are fixed
        while keys:
            iter_idx += 1
            fix = FixExperiment(self.state.fixed)

            stats = VoteStats(self.scenario_ids)
            to_solve = []
            n_judges_total = len(self.scenario_ids)
            n_judges_solved = 0
            n_judges_failed = 0
            cache_hits = 0
            judge_gaps = []
            judge_runtimes = []
            keys_remaining_start = len(keys)
            fixed_before = len(self.state.fixed)

            # Cache or queue
            for s in self.scenario_ids:
                if self.state.in_cache(s, fix):
                    cache_hits += 1
                    res = self.state.cache[(s, fix)]
                    if res is None:
                        n_judges_failed += 1
                    else:
                        n_judges_solved += 1
                        stats.add_from_judge(res)
                        gap = self._safe_float(res.gap)
                        runtime = self._safe_float(res.runtime)
                        if gap is not None:
                            judge_gaps.append(gap)
                        if runtime is not None:
                            judge_runtimes.append(runtime)
                else:
                    self.fix_queues[s].put(fix)
                    to_solve.append(s)
            
            if to_solve:
                start_wait = time.perf_counter()
            # Wait for judges
            for s in to_solve:
                self.fix_queues[s].join()

            if to_solve:
                elapsed = time.perf_counter() - start_wait

            # Collect results
            for _ in to_solve:
                s, fix, res = self.results_queue.get()

                # If a judge does not produce a solution,
                # do not consider its vote
                if res is None:
                    n_judges_failed += 1
                    continue

                n_judges_solved += 1
                self.state.cache[s, fix] = res
                stats.add_from_judge(res)
                gap = self._safe_float(res.gap)
                runtime = self._safe_float(res.runtime)
                if gap is not None:
                    judge_gaps.append(gap)
                if runtime is not None:
                    judge_runtimes.append(runtime)

            stats.calculate_stats()

            critical_key = stats.pick_critical_from(group=group, fix=self.state) 

            # All remaining keys is unanimous 
            if critical_key is None:
                for key in keys:
                    if stats.maj.get(key) is not None:
                        self.state.fixed[key] = stats.maj[key]

                self.fix_iteration_summaries.append(
                    {
                        "group": group,
                        "iteration": iter_idx,
                        "n_judges_total": n_judges_total,
                        "n_judges_solved": n_judges_solved,
                        "n_judges_failed": n_judges_failed,
                        "cache_hits": cache_hits,
                        "cache_hit_rate": cache_hits / n_judges_total if n_judges_total else None,
                        "judge_gap_median": self._percentile(judge_gaps, 0.50),
                        "judge_gap_p90": self._percentile(judge_gaps, 0.90),
                        "judge_runtime_median": self._percentile(judge_runtimes, 0.50),
                        "judge_runtime_p90": self._percentile(judge_runtimes, 0.90),
                        "keys_remaining_start": keys_remaining_start,
                        "keys_remaining_end": 0,
                        "fixed_this_iter": len(self.state.fixed) - fixed_before,
                        "unanimous": True,
                        "critical_key": None,
                    }
                )
                return

            # Otherwise test maj vs sec fix
            fix_maj = fix.extended(critical_key, stats.maj[critical_key]) 
            fix_sec = fix.extended(critical_key, stats.sec[critical_key]) 

            res_maj, res_sec = {}, {}
            to_solve = []
            for s in self.scenario_ids:
                if self.state.in_cache(s, fix_maj):
                    res_maj[s] = self.state.cache[(s, fix_maj)]
                else:
                    self.fix_queues[s].put(fix_maj)
                    to_solve.append((s, fix_maj))
                    
                if self.state.in_cache(s, fix_sec):
                    res_sec[s] = self.state.cache[(s, fix_sec)]
                else:
                    self.fix_queues[s].put(fix_sec)
                    to_solve.append((s, fix_sec))
            
            if to_solve:
                start_maj_sec = time.perf_counter()
            # Wait for judges
            judges_to_wait = {s for (s, _) in to_solve}
            for s in judges_to_wait:
                self.fix_queues[s].join()

            if to_solve:
                elapsed = time.perf_counter() - start_maj_sec

            # Collect results
            for _ in to_solve:
                s, fix, res = self.results_queue.get()
                if fix == fix_maj:
                    res_maj[s] = res 
                elif fix == fix_sec:
                    res_sec[s] = res 

            # check if weighted average obj of maj or sec is bigger
            has_maj = any(r is not None for r in res_maj.values())
            has_sec = any(r is not None for r in res_sec.values())
            obj_maj = sum(r.objective * self.weights[s] for s, r in res_maj.items() if r is not None)
            obj_sec = sum(r.objective * self.weights[s] for s, r in res_sec.items() if r is not None)

            if has_maj and not has_sec:
                val = stats.maj[critical_key]
            elif has_sec and not has_maj:
                val = stats.sec[critical_key]
            elif not has_maj and not has_sec:
                val = stats.maj[critical_key]
            else:
                val = stats.maj[critical_key] if obj_maj < obj_sec else stats.sec[critical_key]
            
            self.state.fixed[critical_key] = val
            keys.remove(critical_key)

            self.fix_iteration_summaries.append(
                {
                    "group": group,
                    "iteration": iter_idx,
                    "n_judges_total": n_judges_total,
                    "n_judges_solved": n_judges_solved,
                    "n_judges_failed": n_judges_failed,
                    "cache_hits": cache_hits,
                    "cache_hit_rate": cache_hits / n_judges_total if n_judges_total else None,
                    "judge_gap_median": self._percentile(judge_gaps, 0.50),
                    "judge_gap_p90": self._percentile(judge_gaps, 0.90),
                    "judge_runtime_median": self._percentile(judge_runtimes, 0.50),
                    "judge_runtime_p90": self._percentile(judge_runtimes, 0.90),
                    "keys_remaining_start": keys_remaining_start,
                    "keys_remaining_end": len(keys),
                    "fixed_this_iter": len(self.state.fixed) - fixed_before,
                    "unanimous": False,
                    "critical_key": str(critical_key),
                    "critical_obj_maj": obj_maj,
                    "critical_obj_sec": obj_sec,
                }
            )
            


    def _shutdown_judges(self):
        for _, judge in self.judges.items():
            judge.terminate()

    def __getattr__(self, name):
        return getattr(self.master_model, name)

@dataclass(frozen=True)
class JudgeSolveResult:
    objective: float
    solution: dict[DecisionKey, int]
    status: str 
    gap: float
    runtime: float

class VoteStats:
    def __init__(self, scenario_ids):
        self.scenario_ids = scenario_ids 

        self.values: dict[DecisionKey, list[int]] = defaultdict(list)
        self.maj: dict[DecisionKey, int] = {}
        self.sec: dict[DecisionKey, int] = {}
        self.p: dict[DecisionKey, float] = {}
    
    def add_from_judge(self, res: JudgeSolveResult):
        for key, val in res.solution.items():
            self.values[key].append(val)

    def _modes(self, vals: list[int]):
        if not vals:
            return None, None, None

        counts = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1 
        
        sorted_counts = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    
        maj = sorted_counts[0][0]
        sec = sorted_counts[1][0] if len(sorted_counts) > 1 else None

        p = vals.count(maj) / len(self.scenario_ids) 
        return maj, sec, p
        
    def calculate_stats(self):
        for key, vals in self.values.items():
            maj, sec, p = self._modes(vals)
            self.maj[key] = maj 
            self.sec[key] = sec 
            self.p[key] = p 

    def pick_critical_from(self, group: str, fix: FixState):
        candidates = []
        for key, p in self.p.items():
            if key[0] != group:
                continue
            if key in fix.fixed:
                continue
            maj, sec = self.maj[key], self.sec[key]
            if maj is None or sec is None:
                continue
            if p < 1.0:
                candidates.append((key, p))
            
        if not candidates:
            return None

        key, _ = max(candidates, key=lambda x: x[1])

        return key 

GRB_STATUS = {
    1: "LOADED",
    2: "OPTIMAL",
    3: "INFEASIBLE",
    4: "INF_OR_UNBD",
    5: "UNBOUNDED",
    6: "CUTOFF",
    7: "ITERATION_LIMIT",
    8: "NODE_LIMIT",
    9: "TIME_LIMIT",
    10: "SOLUTION_LIMIT",
    11: "INTERRUPTED",
    12: "NUMERIC",
    13: "SUBOPTIMAL",
    14: "INPROGRESS",
    15: "USER_OBJ_LIMIT",
    16: "WORK_LIMIT",
    17: "MEM_LIMIT",
    18: "LOCALLY_OPTIMAL",
    19: "LOCALLY_INFEASIBLE",
}

