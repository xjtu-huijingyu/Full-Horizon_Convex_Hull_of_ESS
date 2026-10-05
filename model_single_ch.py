"""Construct convex-hull formulations for an ESS."""
import time
from solver_api import SolverAPI
from dataclasses import dataclass
import numpy as np
import func
EPS = 1e-10

# =============================== Proposed method for the Basic ESS ===============================
@dataclass
class StateES_Basic:
    """Basic ESS state, with idling represented by xdis=0 and xch=0."""
    e: float
    xdis: int = 0
    xch: int = 0
    id: int | None = None

    def key(self):
        return (self.e, self.xdis, self.xch)

class SingleStorageBasicCH:
    """Network formulation for a single Basic ESS unit."""
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
        api = sol.api
        model = self.model
        T = self.T
        es = self.storage

        # Construct the layered directed graph
        self._make_layered_network()
        edge_index = api.tuplelist(self.edges_by_t)
        
        # Variables in the original space
        self.xdis = model.addVars(T, lb=0, ub=1, vtype=C.CONTINUOUS, **{sol.nameprefix: f"storage_ch_xdis_{self.id}"})
        self.xch = model.addVars(T, lb=0, ub=1, vtype=C.CONTINUOUS, **{sol.nameprefix: f"storage_ch_xch_{self.id}"})

        self.pdis = model.addVars(T, lb=0, ub=es.pdis_max, vtype=C.CONTINUOUS, **{sol.nameprefix: f"storage_ch_pdis_{self.id}"})
        self.pch = model.addVars(T, lb=0, ub=es.pch_max, vtype=C.CONTINUOUS, **{sol.nameprefix: f"storage_ch_pch_{self.id}"})
        self.e = model.addVars(T, lb=es.emin, ub=es.emax, vtype=C.CONTINUOUS, **{sol.nameprefix: f"storage_ch_e_{self.id}"})

        # Arc-flow variables in the layered network
        self.w_edge = model.addVars(edge_index, lb=0, ub=1, vtype=C.CONTINUOUS, **{sol.nameprefix: f"storage_ch_w_edge_{self.id}"})

        t0 = min(self.nodes_by_t.keys())
        print(self.nodes_by_t.keys())
        tend = T+1  # Include the virtual terminal layer

        # Incoming and outgoing arc flows for each node
        out_flow = {}
        in_flow = {}

        for t in range(t0, tend - 1):
            for i in self.nodes_by_t[t]:
                out_flow[i, t] = self.w_edge.sum(i, "*", t)

        for t in range(t0 + 1, tend):
            for i in self.nodes_by_t[t]:
                in_flow[i, t] = self.w_edge.sum("*", i, t - 1)

        # Node occupancy: outgoing flow for nonterminal layers and incoming flow for the terminal layer
        node_flow = {}

        for t in range(t0, tend):
            for i in self.nodes_by_t[t]:
                if t < tend - 1:
                    node_flow[i, t] = out_flow[i, t]
                else:
                    node_flow[i, t] = in_flow[i, t]

        # Send one unit of flow from the initial layer
        model.addConstr(api.quicksum(node_flow[i, t0] for i in self.nodes_by_t[t0]) == 1, **{sol.name: f"storage_ch_source_{self.id}"})

        # Flow conservation at intermediate-layer nodes
        model.addConstrs((out_flow[i, t] == in_flow[i, t] for t in range(t0 + 1, tend - 1) for i in self.nodes_by_t[t]), **{sol.nameprefix: f"storage_ch_flow_balance_{self.id}"})

        # Projection of node attributes
        def state_project(t, attr):
            return api.quicksum(getattr(self.states[i], attr) * node_flow[i, t] for i in self.nodes_by_t[t])

        # Projection of arc attributes: period-t power corresponds to arcs from layer t-1 to layer t
        def edge_project(t, attr):
            return api.quicksum(self.transition_power[i, j][attr] * self.w_edge[i, j, q] for i, j, q in edge_index.select("*", "*", t))

        # Project energy and operating states from nodes in the destination layer
        model.addConstrs((self.e[t] == state_project(t, "e") for t in range(T)), **{sol.nameprefix: f"storage_ch_project_e_{self.id}"})
        model.addConstrs((self.xdis[t] == state_project(t, "xdis") for t in range(T)), **{sol.nameprefix: f"storage_ch_project_xdis_{self.id}"})
        model.addConstrs((self.xch[t] == state_project(t, "xch") for t in range(T)), **{sol.nameprefix: f"storage_ch_project_xch_{self.id}"})

        # Project charging and discharging power from arcs entering the current period
        model.addConstrs((self.pch[t] == edge_project(t, "pch") for t in range(T)), **{sol.nameprefix: f"storage_ch_project_pch_{self.id}"})
        model.addConstrs((self.pdis[t] == edge_project(t, "pdis") for t in range(T)), **{sol.nameprefix: f"storage_ch_project_pdis_{self.id}"})
    
    def receive_result(self):
        sol = self.solver
        self.result_xdis = np.array([getattr(self.xdis[t], sol.var_val) for t in range(self.T)])
        self.result_xch = np.array([getattr(self.xch[t], sol.var_val) for t in range(self.T)])
        self.result_pdis = np.array([getattr(self.pdis[t], sol.var_val) for t in range(self.T)])
        self.result_pch = np.array([getattr(self.pch[t], sol.var_val) for t in range(self.T)])
        self.result_e = np.array([getattr(self.e[t], sol.var_val) for t in range(self.T)])
    
    def _make_layered_network(self):
        """Construct the layered directed graph."""
        es = self.storage
        T = self.T
        # Create candidate nodes and arcs
        self._make_candidates()
        # Create the layered directed graph
        req_init = {}
        req_by_t = {}
        self.nodes_by_t, self.edges_by_t = func.filter_layered_graph(T, self.states, self.transitions, req_init, req_by_t=req_by_t, eps=EPS)
    
    def _make_candidates(self):
        """Create candidate nodes and arcs."""
        es = self.storage
        T = self.T
        eps = EPS
        add_e_set = []  # The Basic model introduces no additional energy points
        init_e = None  # The initial energy state is free in the Basic model
        end_e = None  # The Basic model imposes no terminal-energy restriction
        # Candidate values of energy at extreme points
        ru_max = es.pch_max * es.eff_ch
        ru_min = 0  # The minimum charging power is zero in the Basic model
        rd_max = es.pdis_max / es.eff_dis
        rd_min = 0

        s_e = func.make_ContinuousVar_candidates(
            xmin = es.emin,
            xmax = es.emax,
            xinit = init_e,
            xend = end_e,
            add_x_set = add_e_set,
            ru_min = ru_min,
            ru_max = ru_max,
            rd_min = rd_min,
            rd_max = rd_max,
            T = T,
            eps=eps
        )      
        s_e = np.unique(s_e)

        # Create candidate nodes
        st =  time.time()
        states_dis = [StateES_Basic(e=e, xdis=1) for e in s_e]
        states_ch = [StateES_Basic(e=e, xch=1) for e in s_e]
        states_idle = [StateES_Basic(e=e) for e in s_e]
        states = states_dis + states_ch + states_idle

        for i, s in enumerate(states):
            s.id = i
        state_to_id = {s.key(): s.id for s in states}

        # Local helper; it must be defined after state_to_id is created
        def get_state_id(e, xdis=0, xch=0):
            key = (e, xdis, xch)
            state_id = state_to_id.get(key)
            if state_id is None:
                raise ValueError(f"State with key {key} not found in state_to_id mapping.")
            return state_id

        # Construct feasible transitions
        st = time.time()
        transitions = []
        for s in states_dis:
            for e in s_e[(s_e >= s.e - rd_max - eps) & (s_e <= s.e + eps)]:
                transitions.append((s.id, get_state_id(e=e)))
                transitions.append((s.id, get_state_id(e=e, xdis=1)))
                transitions.append((s.id, get_state_id(e=e, xch=1)))
        for s in states_ch:
            for e in s_e[(s_e >= s.e - eps) & (s_e <= s.e + ru_max + eps)]:
                transitions.append((s.id, get_state_id(e=e)))
                transitions.append((s.id, get_state_id(e=e, xdis=1)))
                transitions.append((s.id, get_state_id(e=e, xch=1)))
        for s in states_idle:
            for e in s_e[(s_e >= s.e - eps) & (s_e <= s.e + eps)]:
                transitions.append((s.id, get_state_id(e=e)))
                transitions.append((s.id, get_state_id(e=e, xdis=1)))
                transitions.append((s.id, get_state_id(e=e, xch=1)))
        
        # Associate each arc with pch and pdis
        st = time.time()
        transition_power = {}

        for i, j in transitions:
            si, sj = states[i], states[j]
            delta_e = sj.e - si.e

            pch = delta_e / es.eff_ch if si.xch == 1 else 0.0
            pdis = -delta_e * es.eff_dis if si.xdis == 1 else 0.0

            if abs(pch) <= eps:
                pch = 0.0
            if abs(pdis) <= eps:
                pdis = 0.0

            transition_power[(i, j)] = {"pch": pch, "pdis": pdis}


        self.states = states
        self.state_to_id = state_to_id
        self.transitions = transitions
        self.transition_power = transition_power

# =============================== Benchmark 1: equivalent VCH and HCH formulations from Pozo (2023) for the Basic ESS ===============================
class SingleStorageBasicVCH:
    """Vertex representation of the ESS convex hull from Pozo (2023)."""
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
        api = sol.api
        es = self.storage
        T = self.T
        delta = 1.0
        # General vertices from the authors' appendix; this set describes (pch_t, pdis_t, e_t)
        # Under the original definition, some vertices exceed the bounds when the maximum one-period energy transfer is larger than the usable capacity.
        # self.vertices = [(0.0, 0.0, es.emin),
        #                  (es.pch_max, 0.0, es.emin),
        #                  (es.pch_max, 0.0, es.emax - es.pch_max * es.eff_ch * delta),
        #                  (0.0, 0.0, es.emax),
        #                  (0.0, es.pdis_max, es.emax),
        #                  (0.0, es.pdis_max, es.emin + es.pdis_max / es.eff_dis * delta)]

        # Adjust the power limits to the usable capacity and use the adjusted values for both power and energy coordinates.
        capacity = es.emax - es.emin
        pc = min(es.pch_max, capacity / (delta * es.eff_ch))
        pd = min(es.pdis_max, capacity * es.eff_dis / delta)
        # pc = es.pch_max
        # pd = es.pdis_max
        self.vertices = [
            (0.0, 0.0, es.emin),
            (pc,  0.0, es.emin),
            # (pc,  0.0, min(es.emax, es.emax - es.pch_max * delta * es.eff_ch)),
            (pc,  0.0, es.emax - pc * delta * es.eff_ch),
            (0.0, 0.0, es.emax),
            (0.0, pd,  es.emax),
            # (0.0, pd,  max(es.emin, es.emin + es.pdis_max * delta / es.eff_dis)),
            (0.0, pd,  es.emin + pd * delta / es.eff_dis),
        ]
        
        # Variables
        self.pdis = self.model.addVars(T, vtype=C.CONTINUOUS, **{sol.nameprefix: "es_vch_pdis"})
        self.pch = self.model.addVars(T, vtype=C.CONTINUOUS, **{sol.nameprefix: "es_vch_pch"})
        self.e = self.model.addVars(T, vtype=C.CONTINUOUS, **{sol.nameprefix: "es_vch_e"})
        self.lam = self.model.addVars(T, len(self.vertices), vtype=C.CONTINUOUS, **{sol.nameprefix: "es_vch_lambda"})  # lambda[t, k] is the convex-combination coefficient associated with e[t-1]
        # Convex-hull constraints in vertex representation
        self.model.addConstrs((self.e[t+1] == self.e[t] + self.pch[t] * es.eff_ch * delta - self.pdis[t] / es.eff_dis * delta for t in range(T-1)), **{sol.nameprefix: "es_vch_energy_balance"})
        self.model.addConstrs((self.pch[t] == api.quicksum(self.vertices[k][0] * self.lam[t, k] for k in range(len(self.vertices))) for t in range(T)), **{sol.nameprefix: "es_vch_project_pch"})
        self.model.addConstrs((self.pdis[t] == api.quicksum(self.vertices[k][1] * self.lam[t, k] for k in range(len(self.vertices))) for t in range(T)), **{sol.nameprefix: "es_vch_project_pdis"})
        self.model.addConstrs((self.e[t] == api.quicksum(self.vertices[k][2] * self.lam[t, k] for k in range(len(self.vertices))) for t in range(T)), **{sol.nameprefix: "es_vch_project_e"})
        self.model.addConstrs((api.quicksum(self.lam[t, k] for k in range(len(self.vertices))) == 1 for t in range(T)), **{sol.nameprefix: "es_vch_lambda_sum"})

    def receive_result(self):
        sol = self.solver
        self.result_pdis = np.array([getattr(self.pdis[t], sol.var_val) for t in range(self.T)])
        self.result_pch = np.array([getattr(self.pch[t], sol.var_val) for t in range(self.T)])
        self.result_e = np.array([getattr(self.e[t], sol.var_val) for t in range(self.T)])
        self.result_lam = np.array([[getattr(self.lam[t, k], sol.var_val) for k in range(len(self.vertices))] for t in range(self.T)])
   
class SingleStorageBasicHCH:
    """Half-space representation of the ESS convex hull from Pozo (2023)."""
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
        # Use the same effective power limits as VCH to prevent capacity limits from weakening the power convex-hull constraint.
        capacity = es.emax - es.emin
        pc = min(es.pch_max, capacity / (delta * es.eff_ch))
        pd = min(es.pdis_max, capacity * es.eff_dis / delta)
        # pc = es.pch_max
        # pd = es.pdis_max
        # Variables
        self.pdis = self.model.addVars(T, vtype=C.CONTINUOUS, **{sol.nameprefix: "simple_hch_pdis"})
        self.pch = self.model.addVars(T, vtype=C.CONTINUOUS, **{sol.nameprefix: "simple_hch_pch"})
        self.e = self.model.addVars(T, vtype=C.CONTINUOUS, **{sol.nameprefix: "simple_hch_e"})
        # Convex-hull constraints in half-space representation
        self.model.addConstrs((self.pch[t] <= pc for t in range(T)), **{sol.nameprefix: "simple_hch_max_charge_power"})
        self.model.addConstrs((self.pdis[t] <= pd for t in range(T)), **{sol.nameprefix: "simple_hch_max_discharge_power"})
        self.model.addConstrs((self.e[t] <= es.emax for t in range(T)), **{sol.nameprefix: "simple_hch_max_energy"})
        self.model.addConstrs((self.e[t] >= es.emin for t in range(T)), **{sol.nameprefix: "simple_hch_min_energy"})
        self.model.addConstrs((self.pch[t] / pc + self.pdis[t] / pd <= 1 for t in range(T)), **{sol.nameprefix: "simple_hch_power_convex_hull"})
        self.model.addConstrs((self.e[t] <= es.emax - self.pch[t] * es.eff_ch * delta for t in range(T)), **{sol.nameprefix: "simple_hch_energy_after_charge"})
        self.model.addConstrs((self.e[t] >= es.emin + self.pdis[t] / es.eff_dis * delta for t in range(T)), **{sol.nameprefix: "simple_hch_energy_after_discharge"})
        self.model.addConstrs((self.e[t+1] == self.e[t] + self.pch[t] * es.eff_ch * delta - self.pdis[t] / es.eff_dis * delta for t in range(T-1)), **{sol.nameprefix: "simple_hch_energy_balance"})

    def receive_result(self):
        sol = self.solver
        self.result_pdis = np.array([getattr(self.pdis[t], sol.var_val) for t in range(self.T)])
        self.result_pch = np.array([getattr(self.pch[t], sol.var_val) for t in range(self.T)])
        self.result_e = np.array([getattr(self.e[t], sol.var_val) for t in range(self.T)])
