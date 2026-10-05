import model_single_milp
import model_single_ch
import model_single_lp
import gurobipy as gp
from solver_api import SolverAPI
import pandas as pd
import numpy as np
import os
import coptpy as cp
from coptpy import COPT
from typing import Literal
import time

MODEL_TYPE: Literal["Generalized", "Basic"] = "Generalized"  # Model type; Basic omits initial-state and minimum-duration constraints
SOLVE_METHOD: Literal["MILP", "LP", "CH"] = "MILP"  # Solution method: mixed-integer linear programming, linear programming, or convex hull
CH_METHOD: Literal["Network", "VCH", "HCH"] = "Network"  # Convex-hull method: network flow, vertex representation, or half-space representation
OBJECTIVE_TYPE: Literal["Profit", "SetPointTracking"] = "Profit"  # Objective type: maximize profit or minimize set-point tracking error
SOLVER: Literal["gurobi", "copt"] = "gurobi"  # Optimization solver: Gurobi or COPT

class SelfScheduleStorage:
    def __init__(self, storage, T, price_ch = None, price_dis = None, signal=None, model_name="SelfScheduleStorage", config=None):
        # Core parameters
        self.T = T
        self.storage = storage
        self.price_ch = price_ch
        self.price_dis = price_dis
        self.signal = signal
        self.model_name = model_name
        self.config = {
            "model_type": MODEL_TYPE if config is None or "model_type" not in config else config["model_type"],
            "solve_method": SOLVE_METHOD if config is None or "solve_method" not in config else config["solve_method"],
            "ch_method": CH_METHOD if config is None or "ch_method" not in config else config["ch_method"],
            "obj_type": OBJECTIVE_TYPE if config is None or "obj_type" not in config else config["obj_type"],
            "solver": SOLVER if config is None or "solver" not in config else config["solver"],
            "silent": False if config is None or "silent" not in config else config["silent"],
            "save_results": True if config is None or "save_results" not in config else config["save_results"],
            "output_dir": "output" if config is None or "output_dir" not in config else config["output_dir"],
            "appointed_pdis": None if config is None or "appointed_pdis" not in config else config["appointed_pdis"],
            "appointed_pch": None if config is None or "appointed_pch" not in config else config["appointed_pch"],
            "appointed_e": None if config is None or "appointed_e" not in config else config["appointed_e"]
            }
        # Solver interface
        self.solver = SolverAPI(self.config["solver"])

        # Initialize the model object
        self.model = self.solver.create_model(model_name)
        # Core results
        self.solve_time: float = 0.0
        self.objective_value: float = 0.0
        # Build and solve the model
        self._unit_construction()
        self._objective_construction()
        self._set_params()
        self._appoint_vars()
        self._solve()
        self._save_results()

    def resolve(self, signal=None, price_ch=None, price_dis=None):
        if signal is not None:
            self.signal = signal
        if price_ch is not None:
            self.price_ch = price_ch
        if price_dis is not None:
            self.price_dis = price_dis
        self._objective_construction()
        self._solve()
        self._save_results()

    def _unit_construction(self):
        config = self.config
        model_type = config["model_type"]
        solve_method = config["solve_method"]

        if solve_method == "MILP":
            if model_type == "Generalized":
                pass
            elif model_type == "Basic":
                self.es_model = model_single_milp.SingleStorageBasicMILP(self.model, self.storage, self.T, config)

        elif solve_method == "LP":
            if model_type == "Generalized":
                pass
            elif model_type == "Basic":
                self.es_model = model_single_lp.SingleStorageBasicLP(self.model, self.storage, self.T, config)

        elif solve_method == "CH":
            ch_method = config["ch_method"]

            if model_type == "Generalized":
                pass

            elif model_type == "Basic":
                if ch_method == "Network":
                    self.es_model = model_single_ch.SingleStorageBasicCH(self.model, self.storage, self.T, config)
                elif ch_method == "VCH":
                    self.es_model = model_single_ch.SingleStorageBasicVCH(self.model, self.storage, self.T, config)
                elif ch_method == "HCH":
                    self.es_model = model_single_ch.SingleStorageBasicHCH(self.model, self.storage, self.T, config)


    def _objective_construction(self):
        # Objective function
        objective_type = self.config["obj_type"]
        if objective_type == "Profit":
            if self.price_dis is None or self.price_ch is None:
                raise ValueError("Price signals for charging and discharging must be provided for Profit objective.")
            self.model.setObjective(
                self.solver.api.quicksum(self.price_dis[t] * self.es_model.pdis[t] 
                                         + self.price_ch[t] * self.es_model.pch[t] 
                                         for t in range(self.T)
                                         ), 
                                         self.solver.C.MAXIMIZE
            )
        elif objective_type == "SetPointTracking":
            if self.signal is None:
                raise ValueError("Signal must be provided for SetPointTracking objective.")
            self.model.setObjective(
                self.solver.api.quicksum((self.es_model.pdis[t] - self.es_model.pch[t] - self.signal[t])**2 for t in range(self.T)),
                self.solver.C.MINIMIZE
            )

    def _set_params(self):
        sol = self.solver
        if self.config["silent"]:
            if self.config["solver"] == "gurobi":
                self.model.setParam("OutputFlag", 0)
            elif self.config["solver"] == "copt":
                self.model.setParam("Logging", 0)
        self.model.setParam(sol.TimeLimit, self.config["timelimit"] if self.config.get("timelimit") is not None else 600)
        self.model.setParam(sol.Threads, self.config["threads"] if self.config.get("threads") is not None else 16)
        self.model.setParam("FeasTol", 1e-8)
        self.model.setParam(sol.IntTol, 1e-8)
        if self.config["solve_method"] == "MILP":
            self.model.setParam(sol.MIPGap, self.config["mipgap"] if self.config.get("mipgap") is not None else 1e-6)
        elif self.config["solve_method"] == "CH":
            self.model.setParam(sol.LpMethod, self.config["lp_method"] if self.config.get("lp_method") is not None else 2)

    def _appoint_vars(self):
        """Fix variable values for specified periods across all formulations."""
        self.appointed_pdis = self.config.get("appointed_pdis", None)
        self.appointed_pch = self.config.get("appointed_pch", None)
        self.appointed_e = self.config.get("appointed_e", None)
        print(f"Appointed pdis: {self.appointed_pdis}")
        print(f"Appointed pch: {self.appointed_pch}")
        print(f"Appointed e: {self.appointed_e}")
        eps = 1e-8

        for t in range(self.T):

            if self.appointed_pdis is not None:
                self.es_model.pdis[t].lb = self.appointed_pdis[t] - eps
                self.es_model.pdis[t].ub = self.appointed_pdis[t] + eps

            if self.appointed_pch is not None:
                self.es_model.pch[t].lb = self.appointed_pch[t] - eps
                self.es_model.pch[t].ub = self.appointed_pch[t] + eps

            if self.appointed_e is not None:
                self.es_model.e[t].lb = self.appointed_e[t] - eps
                self.es_model.e[t].ub = self.appointed_e[t] + eps
        
    def _solve(self):
        # Solve the model
        self.solver.solve(self.model)

    def _save_results(self):
        self.solve_time = getattr(self.model, self.solver.runtime)
        self.objective_value = getattr(self.model, self.solver.objval)
        self.es_model.receive_result()

        if not self.config["save_results"]:
            return

        solve_dict = {"solve_time": self.solve_time, "objective_value": self.objective_value}
        solve_dict_df = pd.DataFrame([solve_dict])

        output_folder = self.config["output_dir"]
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, f"{self.model_name}.xlsx")

        es = self.es_model
        es_result_dict = {
            "t": np.arange(self.T),
            "xdis": getattr(es, "result_xdis", None),
            "xch": getattr(es, "result_xch", None),
            "xidle": getattr(es, "result_xidle", None),
            "pdis": es.result_pdis,
            "pch": es.result_pch,
            "e": es.result_e,
            "v_dis2ch": getattr(es, "result_v_dis2ch", None),
            "v_dis2idle": getattr(es, "result_v_dis2idle", None),
            "v_ch2dis": getattr(es, "result_v_ch2dis", None),
            "v_ch2idle": getattr(es, "result_v_ch2idle", None),
            "v_idle2dis": getattr(es, "result_v_idle2dis", None),
            "v_idle2ch": getattr(es, "result_v_idle2ch", None),
        }
        es_result_df = pd.DataFrame(es_result_dict)

        with pd.ExcelWriter(output_path) as writer:
            solve_dict_df.to_excel(writer, sheet_name="solve_info", index=False)
            es_result_df.to_excel(writer, sheet_name="storage_result", index=False)
