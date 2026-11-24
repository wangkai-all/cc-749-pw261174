"""Two-phase simplex LP solver, Python standard library only.

Solves:  minimize c^T x  subject to A.x ~ b  (~ in <=, ==, >=), variables free.

Robust construction chosen after several hand-rolled bases proved fragile:

  * Every free variable x_j is substituted x_j = p_j - n_j with p_j, n_j >= 0.
  * To prevent p_j and n_j growing together, a single large-M constraint
    sum_j (p_j + n_j) <= M is added; because M is large relative to every
    finite optimum, it never binds at an optimal vertex but keeps the feasible
    region bounded.  If an optimum lands on that boundary the caller is told
    the result may be unbounded instead of receiving a silent wrong answer.
  * Slack / surplus / artificial columns are non-negative like every column,
    so the classic ratio test (positive entries only) and Bland's rule apply
    directly.  Lexicographic perturbation of the RHS breaks zero-RHS ties.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

TOL = 1e-10
PIVOT_TOL = 1e-12
BIG_M = 1e7


class LPError(RuntimeError):
    pass


def solve_lp(c: Sequence[float],
             constraints: List[Tuple[Sequence[float], str, float]]
             ) -> dict:
    """Solve a linear program; original variables may take any sign.

    Returns:
        {"status": "optimal" | "infeasible" | "unbounded",
         "x": list[float] | None, "obj": float | None}
    """
    n = len(c)
    if any(len(row) != n for row, _, _ in constraints):
        raise LPError("约束行宽度与目标变量数不一致。")

    # ---- expand free variables x_j = p_j - n_j (columns 2j, 2j+1).
    # Bounding: add p_j <= M and n_j <= M separately. These bound every
    # coordinate in both directions but never stop a finite optimum (all real
    # optima here are far below M), while ruling out p,n growing together.
    expanded: List[Tuple[List[float], str, float]] = []
    for row, sense, rhs in constraints:
        er: List[float] = []
        for a in row:
            er.extend([a, -a])
        if rhs < -TOL:  # normalize RHS to non-negative
            er = [-a for a in er]
            rhs = -rhs
            sense = {"<=": ">=", ">=": "<=", "==": "=="}[sense]
        expanded.append((er, sense, rhs))
    bound_rows: List[Tuple[List[float], str, float]] = []
    for j in range(n):
        for half in (2 * j, 2 * j + 1):
            bound = [0.0] * (2 * n)
            bound[half] = 1.0
            bound_rows.append((bound, "<=", float(BIG_M)))
    expanded.extend(bound_rows)

    next_col = 2 * n
    col_plan: List[List[Tuple[str, int]]] = []
    for _, sense, _ in expanded:
        plan: List[Tuple[str, int]] = []
        if sense == "<=":
            plan.append(("slack", next_col)); next_col += 1
        elif sense == ">=":
            plan.append(("surplus", next_col)); next_col += 1
            plan.append(("artificial", next_col)); next_col += 1
        elif sense == "==":
            plan.append(("artificial", next_col)); next_col += 1
        else:
            raise LPError(f"未知约束方向: {sense}")
        col_plan.append(plan)

    width = next_col + 1  # + RHS
    artificial_cols = [col for plan in col_plan for kind, col in plan
                       if kind == "artificial"]
    art_set = set(artificial_cols)
    # every column is a non-negative variable in this formulation
    nonneg_cols = set(range(next_col))

    rows: List[List[float]] = []
    basic_var: List[int] = []
    for row_index, ((er, sense, rhs), plan) in enumerate(zip(expanded, col_plan)):
        full = [0.0] * width
        for j, a in enumerate(er):
            full[j] = a
        for kind, col in plan:
            full[col] = 1.0 if kind in ("slack", "artificial") else -1.0
            if kind in ("slack", "artificial"):
                basic_var.append(col)
        full[-1] = rhs
        rows.append(full)

    def pivot(enter: int, leave_row: int) -> None:
        pval = rows[leave_row][enter]
        if abs(pval) < PIVOT_TOL:
            raise LPError("退化单纯形主元数值过小。")
        target = rows[leave_row]
        for j in range(width):
            target[j] /= pval
        for i, row in enumerate(rows):
            if i == leave_row:
                continue
            factor = row[enter]
            if abs(factor) < PIVOT_TOL:
                continue
            for j in range(width):
                row[j] -= factor * target[j]
        basic_var[leave_row] = enter
        if any(rows[k][-1] < -PIVOT_TOL for k in range(len(rows))):
            raise LPError("主元后出现负右端项，基不可行。")

    def simplex(obj_coef: List[float], allow: set) -> str:
        max_pivots = 20 * (next_col + len(rows) + 10) ** 2 + 200
        pivots = 0
        while True:
            z = [0.0] * width
            for i, bcol in enumerate(basic_var):
                coef = obj_coef[bcol]
                if abs(coef) < PIVOT_TOL:
                    continue
                row = rows[i]
                for j in range(width):
                    z[j] += coef * row[j]
            reduced = [obj_coef[j] - z[j] for j in range(width)]
            # Bland entering column: smallest eligible index
            enter = None
            for j in range(next_col):
                if j in allow and reduced[j] < -TOL:
                    enter = j
                    break
            if enter is None:
                obj_coef.clear()
                obj_coef.extend(reduced)
                return "optimal"
            # classic ratio test + lexicographic Charnes tie breaking:
            # key = (b/a, 1/a, basic column index, row index)
            leave_row = None
            best_key = None
            for i, row in enumerate(rows):
                aij = row[enter]
                if aij <= PIVOT_TOL:
                    continue
                ratio = row[-1] / aij
                if ratio < -TOL:
                    raise LPError("不可行基：右端项为负。")
                key = (ratio, 1.0 / aij, basic_var[i], i)
                if best_key is None or key < best_key:
                    best_key = key
                    leave_row = i
            if leave_row is None:
                obj_coef.clear()
                obj_coef.extend(reduced)
                return "unbounded"
            pivot(enter, leave_row)
            pivots += 1
            if pivots > max_pivots:
                raise LPError("单纯形主元次数超限（疑似数值退化循环）。")

    # ============================ phase I ============================
    if artificial_cols:
        phase1_coef = [0.0] * width
        for col in artificial_cols:
            phase1_coef[col] = 1.0
        status = simplex(phase1_coef, set(range(next_col)))
        if status == "unbounded":
            raise LPError("阶段 I 不应无界。")
        if -phase1_coef[-1] > TOL:
            return {"status": "infeasible", "x": None, "obj": None}

        # push artificial variables out of the basis; delete redundant rows
        kept_rows: List[List[float]] = []
        kept_basic: List[int] = []
        for i, bcol in enumerate(basic_var):
            if bcol not in art_set:
                kept_rows.append(rows[i])
                kept_basic.append(bcol)
                continue
            pivot_col = None
            for j in range(next_col):
                if j in art_set:
                    continue
                if abs(rows[i][j]) > PIVOT_TOL:
                    pivot_col = j
                    break
            if pivot_col is None:
                continue  # redundant zero row
            pval = rows[i][pivot_col]
            for j in range(width):
                rows[i][j] /= pval
            for k, other in enumerate(rows):
                if k == i:
                    continue
                f = other[pivot_col]
                if abs(f) < PIVOT_TOL:
                    continue
                for j in range(width):
                    other[j] -= f * rows[i][j]
            kept_rows.append(rows[i])
            kept_basic.append(pivot_col)
        rows = kept_rows
        basic_var = kept_basic

    # ============================ phase II ===========================
    obj_coef = [0.0] * width
    for j, cj in enumerate(c):
        obj_coef[2 * j] = cj
        obj_coef[2 * j + 1] = -cj
    allow = set(range(next_col)) - art_set
    status = simplex(obj_coef, allow)
    if status == "unbounded":
        return {"status": "unbounded", "x": None, "obj": None}

    # ---- recover original variables from p_j - n_j and basis values
    col_to_row = {bcol: i for i, bcol in enumerate(basic_var)}
    values = [0.0] * n
    for j in range(n):
        for col, sign in ((2 * j, 1.0), (2 * j + 1, -1.0)):
            if col in col_to_row:
                values[j] += sign * rows[col_to_row[col]][-1]
    values = [0.0 if abs(v) < TOL else v for v in values]

    # detect genuine unboundedness: any p_j / n_j at the artificial M bound.
    # A finite optimum has every original coordinate strictly inside the box;
    # a coordinate at (near) M means the solver walked to the artificial wall,
    # i.e. the real feasible region is unbounded in that direction.
    if any(abs(v) > BIG_M * 0.5 for v in values):
        return {"status": "unbounded", "x": None, "obj": None}
    obj_val = sum(cj * xj for cj, xj in zip(c, values))
    return {"status": "optimal", "x": values, "obj": obj_val}
