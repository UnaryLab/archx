from ortools.sat.python import cp_model
from loguru import logger

from archx.programming.object.architecture import Architecture
from archx.programming.object.event import Event
from archx.programming.object.metric import Metric
from archx.programming.object.workload import Workload
from archx.programming.object.parameter import ParameterEnumerator
from archx.utils import write_yaml

class AGraph:
    def __init__(self, path):
        self.path = path
        self.model = cp_model.CpModel()
        self.parameter_enumerator = ParameterEnumerator(self.model)

        self.architecture = Architecture(self.parameter_enumerator)
        self.event = Event()
        self.metric = Metric()
        self.workload = Workload(self.parameter_enumerator)

    def direct_constraint(self, parameters: list):
        """
        Add a direct constraint where all parameters must have the same index.
        This means they sweep together - when one changes, they all change.
        
        Example: if param_a has values [32, 64, 128] and param_b has values [4, 8, 16],
        valid combinations are: (32,4), (64,8), (128,16)
        """
        assert isinstance(parameters, list) and all(isinstance(p, cp_model.IntVar) for p in parameters), \
            "'parameters' must be a list of cp_model.IntVar."
        assert len(parameters) >= 2, "Need at least 2 parameters to create a constraint."
        
        first_param = parameters[0]
        for param in parameters[1:]:
            self.model.Add(first_param == param)

    def direct_constraint_conditional(self):
        raise NotImplementedError("Direct conditional constraints are not yet implemented.")

    def anti_constraint(self):
        raise NotImplementedError("Anti constraints are not yet implemented.")

    def anti_constraint_conditional(self):
        raise NotImplementedError("Anti conditional constraints are not yet implemented.")
    
    def conditional_constraint(self, a: cp_model.IntVar, b: cp_model.IntVar, condition):
        """
        Add a constraint where parameter combinations must satisfy a condition on actual values.
        
        Args:
            a: IntVar
            b: IntVar
            condition: Lambda that takes actual values and returns bool
                       e.g., lambda a, b: a[0] == b[0]
        
        Example:
            fifo_instance values: [[128], [256]]           (indices 0, 1)
            mux_instance values:  [[128,128], [256,256], [512,512]]  (indices 0, 1, 2)
            condition: lambda a, b: a[0] == b[0]
            
            Valid pairs: (0,0) because 128==128, (1,1) because 256==256
            Invalid: (0,1), (0,2), (1,0), (1,2)
        """
        assert isinstance(a, cp_model.IntVar), "'a' must be a cp_model.IntVar."
        assert isinstance(b, cp_model.IntVar), "'b' must be a cp_model.IntVar."
        assert callable(condition), "'condition' must be a callable (lambda or function)."
        
        # Get the actual values from the enumerator
        values_a = self.parameter_enumerator.get_values_for_var(a)
        values_b = self.parameter_enumerator.get_values_for_var(b)
        
        # Use list comprehension (faster than explicit loops)
        # Pre-compute valid (index_a, index_b) pairs where condition is True
        allowed_tuples = [
            (idx_a, idx_b)
            for idx_a, val_a in enumerate(values_a)
            for idx_b, val_b in enumerate(values_b)
            if condition(val_a, val_b)
        ]
        
        assert len(allowed_tuples) > 0, "No valid combinations found for the given condition."
        
        # Table constraint handled efficiently in C++ by OR-Tools
        self.model.AddAllowedAssignments([a, b], allowed_tuples)
        
    def generate(self):
        solutions = self.solve()
        self.save_solutions(solutions)
        
    def solve(self):
        """
        Generate all valid configurations by solving the constraint model.
        Returns unique architecture and workload configurations separately
        for memory efficiency.
        
        Returns:
            dict with 'architecture' and 'workload' keys, each containing
            a list of unique configurations.
        """
        solver = cp_model.CpSolver()
        
        # Separate variables by type (architecture vs workload)
        arch_vars = []
        workload_vars = []
        
        for var, values in self.parameter_enumerator.var_to_values.items():
            var_name = var.Name()
            if var_name.startswith('architecture_'):
                arch_vars.append(var)
            elif var_name.startswith('workload_'):
                workload_vars.append(var)
        
        all_vars = arch_vars + workload_vars
        
        if not all_vars:
            logger.warning("No parameters defined. Nothing to generate.")
            return {'architecture': [], 'workload': []}
        
        # Use a solution collector that tracks unique configs per category
        class UniqueSolutionCollector(cp_model.CpSolverSolutionCallback):
            def __init__(self, arch_vars, workload_vars, var_to_values):
                cp_model.CpSolverSolutionCallback.__init__(self)
                self._arch_vars = arch_vars
                self._workload_vars = workload_vars
                self._var_to_values = var_to_values
                # Use sets of tuples for deduplication (hashable)
                self._arch_seen = set()
                self._workload_seen = set()
                self._arch_solutions = []
                self._workload_solutions = []
            
            def _extract_config(self, variables):
                """Extract config as both a hashable tuple and a dict."""
                config_dict = {}
                config_tuple = []
                for var in variables:
                    idx = self.Value(var)
                    actual_value = self._var_to_values[var][idx]
                    config_dict[var.Name()] = actual_value
                    # Convert to hashable form for deduplication
                    config_tuple.append((var.Name(), idx))
                return config_dict, tuple(config_tuple)
            
            def on_solution_callback(self):
                # Extract and deduplicate architecture config
                if self._arch_vars:
                    arch_dict, arch_tuple = self._extract_config(self._arch_vars)
                    if arch_tuple not in self._arch_seen:
                        self._arch_seen.add(arch_tuple)
                        self._arch_solutions.append(arch_dict)
                
                # Extract and deduplicate workload config
                if self._workload_vars:
                    workload_dict, workload_tuple = self._extract_config(self._workload_vars)
                    if workload_tuple not in self._workload_seen:
                        self._workload_seen.add(workload_tuple)
                        self._workload_solutions.append(workload_dict)
            
            def get_solutions(self):
                return {
                    'architecture': self._arch_solutions,
                    'workload': self._workload_solutions
                }
        
        collector = UniqueSolutionCollector(arch_vars, workload_vars, 
                                            self.parameter_enumerator.var_to_values)
        
        # Find all solutions
        solver.parameters.enumerate_all_solutions = True
        status = solver.Solve(self.model, collector)
        
        solutions = collector.get_solutions()
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            logger.info(f"Generated {len(solutions['architecture'])} unique architecture configurations.")
            logger.info(f"Generated {len(solutions['workload'])} unique workload configurations.")
        elif status == cp_model.INFEASIBLE:
            logger.error("No valid configurations found. Constraints are infeasible.")
        else:
            logger.warning(f"Solver returned status: {solver.StatusName(status)}")

        return solutions
    
    def save_solutions(self, solutions: dict):

        # create output file path
        architecture_path = self.path + '/architecture'
        event_path = self.path + '/event'
        metric_path = self.path + '/metric'
        workload_path = self.path + '/workload'

        for i, architecture_config in enumerate(solutions['architecture']):
            architecture_dict = self.architecture.to_yaml(architecture_config)
            write_yaml(self.path + f'/architecture/config_{i}.yaml', architecture_dict)

        for i, workload_config in enumerate(solutions['workload']):
            workload_dict = self.workload.to_yaml(workload_config)
            write_yaml(self.path + f'/workload/config_{i}.yaml', workload_dict)



def _generate_runs():
    pass

def _gui():
    pass