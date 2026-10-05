"""LP relaxation of the ESS model."""
import model_single_milp

class SingleStorageBasicLP(model_single_milp.SingleStorageBasicMILP):
    """LP relaxation of the Basic ESS model."""
    def __init__(self, model, storage, T, config):
        super().__init__(model, storage, T, config)
        # Convert binary variables to continuous variables
        self._relax_binary_variables()

    def _relax_binary_variables(self):
        C = self.solver.C
        # Convert binary variables to continuous variables
        for var in [self.xdis, self.xch]:
            for t in range(self.T):
                var[t].vtype = C.CONTINUOUS
                var[t].lb = 0.0
                var[t].ub = 1.0
