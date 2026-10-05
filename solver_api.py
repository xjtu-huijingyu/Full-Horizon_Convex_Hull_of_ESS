import gurobipy as gp
import coptpy as cp
from typing import Any

class SolverAPI:
    def __init__(self, solver):
        self.solver = solver

        if solver == "gurobi":
            # Solver-specific attribute names
            self.api = gp
            self.C = gp.GRB
            self.var_val = "X"
            self.objval = "ObjVal"
            self.runtime = "Runtime"
            self.name = "name"
            self.nameprefix = "name"
            # Solver parameters
            self.TimeLimit = "TimeLimit"
            self.Threads = "Threads"
            self.MIPGap = "MIPGap"
            self.IntTol = "IntFeasTol"
            self.LpMethod = "Method"  # -1 automatic, 1 dual simplex, 2 barrier

        elif solver == "copt":
            self.api = cp
            self.C = cp.COPT
            self.var_val = "x"
            self.objval = "objval"
            self.runtime = "SolvingTime"
            self.name = "name"
            self.nameprefix = "nameprefix"
            # Solver parameters
            self.TimeLimit = "TimeLimit"
            self.Threads = "Threads"
            self.MIPGap = "RelGap"
            self.IntTol = "IntTol"
            self.LpMethod = "LpMethod"  # -1 automatic, 1 dual simplex, 2 barrier

        else:
            raise ValueError(f"Unsupported solver: {solver}")

    def create_model(self, model_name: str) -> Any:
        if self.solver == "gurobi":
            return gp.Model(model_name)
        elif self.solver == "copt":
            self.env = cp.Envr()
            return self.env.createModel(model_name)
        else:
            raise ValueError(f"Unsupported solver: {self.solver}")

    def solve(self, model):
        if self.solver == "gurobi":
            model.optimize()
        elif self.solver == "copt":
            model.solve()
        else:
            raise ValueError(f"Unsupported solver: {self.solver}")
