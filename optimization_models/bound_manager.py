from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

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

    def _get_var(self, group: str, key: Any):
        vardict = getattr(self.m, group)
        return vardict[key]

    def _save_if_needed(self, dk: DecisionKey) -> None:
        if dk in self._saved:
            return
        group, key = dk
        var = self._get_var(group, key)
        if var is None:
            return
        self._saved[dk] = (var.LB, var.UB, var.Start)

    def apply_fix(self, dk: DecisionKey, value: int) -> None:
        self._save_if_needed(dk)
        group, key = dk
        var = self._get_var(group, key)
        if var is None:
            return
        v = int(value)
        var.LB = v
        var.UB = v

    def apply_ub(self, dk: DecisionKey, ub: int) -> None:
        self._save_if_needed(dk)
        group, key = dk
        var = self._get_var(group, key)
        if var is None:
            return
        var.UB = min(var.UB, int(ub))

    def apply_persistent_state(self, fix: FixState) -> None:
        """
        Applies both fixed equalities and UB tightenings.
        This modifies model variables directly.
        """
        # Apply UB first, then equalities (equalities dominate anyway)
        for dk, u in fix.ub.items():
            self.apply_ub(dk, u)
        for dk, v in fix.fixed.items():
            self.apply_fix(dk, v)

    def restore(self) -> None:
        """Restore all saved variables to previous (LB,UB,Start)."""
        for (group, key), (lb, ub, st) in self._saved.items():
            var = getattr(self.m, group)[key]
            var.LB = lb
            var.UB = ub
            var.Start = st
        self._saved.clear()
        self.m.model.update()
        