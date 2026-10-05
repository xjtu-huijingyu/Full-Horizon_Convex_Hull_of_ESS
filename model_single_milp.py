import numpy as np
from solver_api import SolverAPI

class SingleStorageBasicMILP:
    """Basic ESS MILP from Pozo (2023), where e[t] is the post-action energy state and the initial state is free, with no prescribed initial-terminal balance."""
    def __init__(self, model, storage, T, config):
        self.id = storage.id
        self.storage = storage
        self.model = model
        self.T = T
        self.config = config
        # Solver interface
        self.solver = SolverAPI(config["solver"])
        # Build the model
        self._build_model()

    def _build_model(self):
        sol = self.solver
        C = sol.C
        es = self.storage
        T = self.T
        delta = 1.0
        # Correct power limits that clearly exceed the usable energy capacity
        # pc = es.pch_max
        # pd = es.pdis_max
        capacity = es.emax - es.emin
        pc = min(es.pch_max, capacity / (delta * es.eff_ch))
        pd = min(es.pdis_max, capacity * es.eff_dis / delta)
        
        # Variables
        self.xdis = self.model.addVars(T, vtype=C.BINARY, **{sol.nameprefix: "simple_xdis"})
        self.xch = self.model.addVars(T, vtype=C.BINARY, **{sol.nameprefix: "simple_xch"})
        self.pdis = self.model.addVars(T, vtype=C.CONTINUOUS, **{sol.nameprefix: "simple_pdis"})
        self.pch = self.model.addVars(T, vtype=C.CONTINUOUS, **{sol.nameprefix: "simple_pch"})
        self.e = self.model.addVars(T+1, vtype=C.CONTINUOUS, **{sol.nameprefix: "simple_e"})
        # Charging/discharging complementarity, power, and energy constraints
        self.model.addConstrs((self.xdis[t] + self.xch[t] <= 1 for t in range(T)), **{sol.nameprefix: "simple_state_constraint"})
        self.model.addConstrs((self.pdis[t] <= pd * self.xdis[t] for t in range(T)), **{sol.nameprefix: "simple_max_discharge_power"})
        self.model.addConstrs((self.pch[t] <= pc * self.xch[t] for t in range(T)), **{sol.nameprefix: "simple_max_charge_power"})
        self.model.addConstrs((self.e[t] <= es.emax for t in range(T+1)), **{sol.nameprefix: "simple_max_energy"})
        self.model.addConstrs((self.e[t] >= es.emin for t in range(T+1)), **{sol.nameprefix: "simple_min_energy"})
        self.model.addConstrs((self.e[t+1] == self.e[t] + self.pch[t] * es.eff_ch * delta - self.pdis[t] / es.eff_dis * delta for t in range(T)), **{sol.nameprefix: "simple_energy_balance"})

    def receive_result(self):
        sol = self.solver
        self.result_xdis = np.array([getattr(self.xdis[t], sol.var_val) for t in range(self.T)])
        self.result_xch = np.array([getattr(self.xch[t], sol.var_val) for t in range(self.T)])
        self.result_pdis = np.array([getattr(self.pdis[t], sol.var_val) for t in range(self.T)])
        self.result_pch = np.array([getattr(self.pch[t], sol.var_val) for t in range(self.T)])
        self.result_e = np.array([getattr(self.e[t], sol.var_val) for t in range(self.T)])
