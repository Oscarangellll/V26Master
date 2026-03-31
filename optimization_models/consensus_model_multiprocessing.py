from collections import defaultdict
from dataclasses import dataclass
import multiprocessing as mp
import signal
import time 

from config import ScenarioConfig
from optimization_models import OptimizationModel

def _judge(s, case, results_queue, fix_queue, sem):
        
    scenario_cfg = ScenarioConfig(case, [s])
    model = OptimizationModel(case, scenario_cfg, [s], weights={s: 1.0})
    
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 0.02
    model.Params.Threads = 1
    
    while True:

        fix = fix_queue.get()
        
        sem.acquire()
        try:
            fix.apply_to(model)

            model.optimize()
           
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
            else:
                result = None
                
            results_queue.put((s, fix, result))

            fix.remove_from(model)
        
        finally:
            sem.release()

        fix_queue.task_done()

DecisionKey = (str, any) # (group, index), identifies a variable

class FixExperiment:
    def __init__(self, fixed: {DecisionKey: int}):
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
        self.fixed: {DecisionKey: int} = {}
        
        self.cache: {(int, FixExperiment): JudgeSolveResult} = {} 
    
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

        self.fix_and_bounds_time_limit = 10 #18_000
        self.master_model_time_limit = 3_600

    def optimize(self):
        def _alarm_handler(signum, frame):
            raise TimeoutError
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
        
        # Remove keys when they become fixed. Loop exits when all keys are fixed
        while keys:
            fix = FixExperiment(self.state.fixed)
                    
            stats = VoteStats(self.scenario_ids)
            to_solve = []

            # Cache or queue
            for s in self.scenario_ids:
                if self.state.in_cache(s, fix):
                    res = self.state.cache[(s, fix)]
                    stats.add_from_judge(res) 
                else:
                    self.fix_queues[s].put(fix)
                    to_solve.append(s)
                    
            # Wait for judges
            for s in to_solve:
                self.fix_queues[s].join()

            # Collect results
            for _ in to_solve:
                s, fix, res = self.results_queue.get()
                        
                # If a judge does not produce a solution,
                # do not consider its vote
                if res is None:
                    continue
                        
                self.state.cache[s, fix] = res
                stats.add_from_judge(res)

            stats.calculate_stats()
            
            critical_key = stats.pick_critical_from(group=group, fix=self.state) 
            
            # All remaining keys is unanimous 
            if critical_key is None:
                for key in keys:
                    if stats.maj.get(key) is not None:
                        self.state.fixed[key] = stats.maj[key]
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
            
            # Wait for judges
            for s in {s for (s, _) in to_solve}:
                self.fix_queues[s].join()

            # Collect results
            for _ in to_solve:
                s, fix, res = self.results_queue.get()
                if fix == fix_maj:
                    res_maj[s] = res 
                elif fix == fix_sec:
                    res_sec[s] = res 

            # check if weighted average obj of maj or sec is bigger
            obj_maj = sum(r.objective * self.weights[s] for s, r in res_maj.items())
            obj_sec = sum(r.objective * self.weights[s] for s, r in res_sec.items())

            val = stats.maj[critical_key] if obj_maj < obj_sec else stats.sec[critical_key]
            
            self.state.fixed[critical_key] = val
            keys.remove(critical_key)
            


    def _shutdown_judges(self):
        for _, judge in self.judges.items():
            judge.terminate()

    def __getattr__(self, name):
        return getattr(self.master_model, name)

@dataclass(frozen=True)
class JudgeSolveResult:
    objective: float
    solution: {DecisionKey: int}
    status: str 
    gap: float
    runtime: float

class VoteStats:
    def __init__(self, scenario_ids):
        self.scenario_ids = scenario_ids 

        self.values: {DecisionKey: []} = defaultdict(list) 
        self.maj: {DecisionKey: int} = {}
        self.sec: {DecisionKey: int} = {}
        self.p: {DecisionKey: float} = {}
    
    def add_from_judge(self, res: JudgeSolveResult):
        for key, val in res.solution.items():
            self.values[key].append(val)

    def _modes(self, vals: [int]):
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

