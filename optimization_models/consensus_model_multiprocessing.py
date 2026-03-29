from collections import defaultdict
from dataclasses import dataclass
import multiprocessing as mp
import time 

from config import ScenarioConfig
from optimization_models import OptimizationModel

def _judge(s, case, results_queue, fix_queue, sem):
        
    scenario_cfg = ScenarioConfig(case, [s])
    model = OptimizationModel(case, scenario_cfg, [s])
    
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 0.02
    model.Params.Threads = 1
    
    while True:
        print("Begin judge loop", s)

        fix = fix_queue.get()
        if fix is None: # Sentinel value, consensus is done 
            break
        
        print("Check semaphore, wait on available resource", s)
        sem.acquire()
        try:
            print("Apply fix", s)
            fix.apply_to(model)

            model.optimize()
           
            if model.SolCount > 0:
                solution = frozenset(
                    ((group, idx), int(var.X))
                    for group in ["eta", "gamma_LT", "gamma_ST"] 
                    for idx, var in getattr(model, group).items()
                )
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

            print("Reverse fix", s)
            fix.remove_from(model)
        
        finally:
            sem.release()

        fix_queue.task_done()

DecisionKey = (str, any) # (group, index), identifies a variable

class FixExperiment:
    def __init__(self, fixed):
        # self.fixed is frozenset((DecisionKey, int), ...)
        if isinstance(fixed, dict):
            self.fixed = frozenset(fixed.items())
        else:
            self.fixed = frozenset(fixed)

    def extended(self, key: DecisionKey, val: int):
        return FixExperiment(self.fixed | {(key, val)}) 

    def apply_to(self, model):
        for (key, val) in self.fixed:
            group, idx = key
            var = getattr(model, group)[idx]
            var.LB = val
            var.UB = val

    def remove_from(self, model):
        for (key, val) in self.fixed:
            group, idx = key 
            var = getattr(model, group)[idx]
            var.LB = 0
            var.UB = 1e100

    def __eq__(self, other):
        return isinstance(other, FixExperiment) and self.fixed == other.fixed

    def __hash__(self):
        return hash(self.fixed)

class FixState:
    def __init__(self):
        self.fixed: {DecisionKey: int} = {}
        
        self.cache: {(int, FixExperiment): JudgeSolveResult} = {} 
    
    def in_cache(self, s_new, fix_new: FixExperiment):
        
        if (s_new, fix_new) in self.cache:
            return True
        
        for (s_old, fix_old), res in self.cache.items():
            
            if s_new != s_old:
                continue

            if not fix_old.fixed.issubset(fix_new.fixed):
                continue
            
            sol = dict(res.solution)
            consistent_solution = True
            for (key, val) in fix_new.fixed:
                if sol.get(key) != val:
                    consistent_solution = False

            if not consistent_solution:
                continue

            self.cache[(s_new, fix_new)] = res
            return True
        
        return False
        

class ConsensusModelMP:
    def __init__(self, case, scenario_ids):
        self.case = case
        self.scenario_ids = scenario_ids

        self.results_queue = mp.SimpleQueue()
        self.fix_queues = {s: mp.JoinableQueue() for s in self.scenario_ids}

        self.sem = mp.BoundedSemaphore(mp.cpu_count() - 1)

        self.judges = {}

        self.state = FixState()

    def optimize(self):
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

            print("CURRENT FIX: ", self.state.fixed)
            self.fix_gamma_LT()

            print("CURRENT FIX: ", self.state.fixed)
            
            bounds = self.tighten_gamma_ST()
            print(bounds)
            
        except KeyboardInterrupt:
            self._shutdown_judges_immediately()
            exit()
        else:
            self._shutdown_judges_gracefully()

        # Solve master with fixations and bounds
        scenario_cfg = ScenarioConfig(self.case, self.scenario_ids)
        master_model = OptimizationModel(self.case, scenario_cfg, self.scenario_ids)
        
        fix = FixExperiment(self.state.fixed)
        fix.apply_to(master_model)

        for key, (lb, ub) in bounds.items():
            group, idx = key
            var = getattr(master_model, group)[idx]
            var.LB = lb
            var.UB = ub

        master_model.Params.MIPGap = 0.02
        
        master_model.optimize()

        self.master_model = master_model

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
        
        bounds = {}
        for key in keys:
            vals = []
            for s in self.scenario_ids:
                res = self.state.cache.get((s, fix))
                if res is None:
                    continue

                for k, v in res.solution:
                    if key == k:
                        vals.append(v)
        
            bounds[key] = (min(vals), max(vals))
        
        return bounds

    def fix(self, group: str, keys: set[DecisionKey]):
        
        # Remove keys when they become fixed. Loop exits when all keys are fixed
        while keys:
            fix = FixExperiment(self.state.fixed)
                    
            stats = VoteStats()
            to_solve = []

            # Cache or queue
            for s in self.scenario_ids:
                if self.state.in_cache(s, fix):
                    stats.add_from_judge(s, self.state.cache[(s, fix)]) 
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
                stats.add_from_judge(s, res)

            stats.calculate_stats()
            
            critical_key = stats.pick_critical_from(group=group, fix=self.state) 
            
            # All remaining keys is unanimous 
            if critical_key is None:
                for key in keys:
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

            # check if average obj of maj or sec is bigger,
            obj_maj = sum(r.objective for r in res_maj.values()) / len(res_maj)
            obj_sec = sum(r.objective for r in res_sec.values()) / len(res_sec)
            
            val = stats.maj[critical_key] if obj_maj < obj_sec else stats.sec[critical_key]
            
            self.state.fixed[critical_key] = val
            keys.remove(critical_key)


    def _shutdown_judges_immediately(self):
        for _, judge in self.judges.items():
            judge.terminate()

    def _shutdown_judges_gracefully(self):
        for _, queue in self.fix_queues.items():
            queue.put(None)
        
        for _, judge in self.judges.items():
            judge.join()

    def __getattr__(self, name):
        return getattr(self.master_model, name)

@dataclass(frozen=True)
class JudgeSolveResult:
    objective: float
    solution: frozenset[(DecisionKey, int)]
    status: str 
    gap: float
    runtime: float

class VoteStats:
    def __init__(self):
        self.values: {DecisionKey: []} = defaultdict(list) 
        self.maj: {DecisionKey: int} = {}
        self.sec: {DecisionKey: int} = {}
        self.p: {DecisionKey: float} = {}
    
    def add_from_judge(self, s, res: JudgeSolveResult):
        for sol in res.solution:
            key, val = sol
            self.values[key].append(val)

    @staticmethod
    def _modes(vals: [int]):
        counts = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1 
        
        sorted_counts = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    
        maj = sorted_counts[0][0]
        sec = sorted_counts[1][0] if len(sorted_counts) > 1 else None
        return maj, sec
        
    @staticmethod
    def _frac(vals: [int], maj: int):
        return vals.count(maj) / len(vals) 

    def calculate_stats(self):
        for key, vals in self.values.items():
            maj, sec = self._modes(vals)
            self.maj[key] = maj 
            self.sec[key] = sec 
            self.p[key] = self._frac(vals, maj)

    def pick_critical_from(self, group: str, fix: FixState):
        candidates = [
            (key, p)
            for key, p in self.p.items()
            if key[0] == group
            and key not in fix.fixed 
            and p < 1.0
        ]
        
        if not candidates:
            return None

        key, _ = max(candidates, key=lambda x: x[1])

        # returning none means that the remaining variables in "group"
        # is unanimous, or that all variables from "group" is fixed, but 
        # this should not happen
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

