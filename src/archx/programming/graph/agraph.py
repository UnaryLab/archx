from ortools.sat.python import cp_model
from loguru import logger
import pandas as pd
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

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
        Returns unique architecture and workload configurations separately,
        plus a list of valid (arch_idx, workload_idx) pairs.
        
        Returns:
            dict with:
                'architecture': list of unique architecture configs
                'workload': list of unique workload configs  
                'configurations': list of {'arch_idx': int, 'work_idx': int} pairs
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
            return {'architecture': [], 'workload': [], 'configurations': []}
        
        # Use a solution collector that tracks unique configs and valid pairs
        class UniqueSolutionCollector(cp_model.CpSolverSolutionCallback):
            def __init__(self, arch_vars, workload_vars, var_to_values):
                cp_model.CpSolverSolutionCallback.__init__(self)
                self._arch_vars = arch_vars
                self._workload_vars = workload_vars
                self._var_to_values = var_to_values
                # Use dicts to map tuple -> index for deduplication
                self._arch_seen = {}  # tuple -> index
                self._workload_seen = {}  # tuple -> index
                self._arch_solutions = []
                self._workload_solutions = []
                self._configurations = []  # list of (arch_idx, work_idx) pairs
            
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
                arch_idx = None
                work_idx = None
                
                # Extract and deduplicate architecture config
                if self._arch_vars:
                    arch_dict, arch_tuple = self._extract_config(self._arch_vars)
                    if arch_tuple not in self._arch_seen:
                        arch_idx = len(self._arch_solutions)
                        self._arch_seen[arch_tuple] = arch_idx
                        self._arch_solutions.append(arch_dict)
                    else:
                        arch_idx = self._arch_seen[arch_tuple]
                
                # Extract and deduplicate workload config
                if self._workload_vars:
                    workload_dict, workload_tuple = self._extract_config(self._workload_vars)
                    if workload_tuple not in self._workload_seen:
                        work_idx = len(self._workload_solutions)
                        self._workload_seen[workload_tuple] = work_idx
                        self._workload_solutions.append(workload_dict)
                    else:
                        work_idx = self._workload_seen[workload_tuple]
                
                # Record the valid (arch, workload) pair
                self._configurations.append({
                    'arch_idx': arch_idx,
                    'work_idx': work_idx
                })
            
            def get_solutions(self):
                return {
                    'architecture': self._arch_solutions,
                    'workload': self._workload_solutions,
                    'configurations': self._configurations
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
            logger.info(f"Generated {len(solutions['configurations'])} valid configuration pairs.")
        elif status == cp_model.INFEASIBLE:
            logger.error("No valid configurations found. Constraints are infeasible.")
        else:
            logger.warning(f"Solver returned status: {solver.StatusName(status)}")

        return solutions
    
    def save_solutions(self, solutions: dict):
        """
        Save architecture/workload YAML files and create a CSV mapping all configurations.
        """
        # Create output directories
        os.makedirs(self.path + '/architecture', exist_ok=True)
        os.makedirs(self.path + '/workload', exist_ok=True)
        os.makedirs(self.path + '/event', exist_ok=True)
        os.makedirs(self.path + '/metric', exist_ok=True)
        os.makedirs(self.path + '/runs', exist_ok=True)

        # Save unique architecture configs
        for i, architecture_config in enumerate(solutions['architecture']):
            architecture_dict = self.architecture.to_yaml(architecture_config)
            write_yaml(os.path.join(self.path, f'architecture/config_{i}.architecture.yaml'), architecture_dict)

        # Save unique workload configs
        for i, workload_config in enumerate(solutions['workload']):
            workload_dict = self.workload.to_yaml(workload_config)
            write_yaml(os.path.join(self.path, f'workload/config_{i}.workload.yaml'), workload_dict)

        # Save event and metric (single files)
        event_dict = self.event.to_yaml()
        event_path = os.path.abspath(os.path.join(self.path, 'event/event.yaml'))
        write_yaml(event_path, event_dict)

        metric_dict = self.metric.to_yaml()
        metric_path = os.path.abspath(os.path.join(self.path, 'metric/metric.yaml'))
        write_yaml(metric_path, metric_dict)

        # Build CSV with all configuration mappings
        df = pd.DataFrame()
        
        for i, config in enumerate(solutions['configurations']):
            arch_path = os.path.abspath(os.path.join(self.path, f'architecture/config_{config["arch_idx"]}.architecture.yaml'))
            work_path = os.path.abspath(os.path.join(self.path, f'workload/config_{config["work_idx"]}.workload.yaml'))
            run_path = os.path.abspath(os.path.join(self.path, f'runs/config_{i}'))
            checkpoint_path = os.path.abspath(os.path.join(self.path, f'runs/config_{i}/checkpoint.gt'))

            row = {
                'config_id': i,
                'arch_config': config['arch_idx'],
                'arch_path': arch_path,
                'work_config': config['work_idx'],
                'work_path': work_path,
                'event_path': event_path,
                'metric_path': metric_path,
                'run_path': run_path,
                'checkpoint_path': checkpoint_path
            }
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        # Save CSV
        csv_path = os.path.join(self.path, 'configurations.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved configuration mapping to {csv_path}")

        return df

def _generate_runs(df, path):
    """
    Write a runs.txt file from a pandas DataFrame with config paths.
    Each line: -a <arch_path> -e <event_path> -m <metric_path> -r <run_path> -w <work_path> -c <checkpoint_path> -s
    Args:
        df (pd.DataFrame): DataFrame with columns arch_path, event_path, metric_path, run_path, work_path, checkpoint_path
        out_path (str, optional): Output file path. Defaults to self.path + 'runs.txt'.
    """
    lines = []
    for _, row in df.iterrows():
        arch = str(row.get('arch_path', '') or '')
        event = str(row.get('event_path', '') or '')
        metric = str(row.get('metric_path', '') or '')
        run = str(row.get('run_path', '') or '')
        work = str(row.get('work_path', '') or '')
        checkpoint = str(row.get('checkpoint_path', '') or '')
        line = f"-a {arch} -e {event} -m {metric} -r {run} -w {work} -c {checkpoint} -s"
        lines.append(line)
    with open(path, 'w') as f:
        for line in lines:
            f.write(line + '\n')

def _gui(df, path):
    # Read configurations from CSV
    # Get unique values for arch_config and work_config
    arch_configs = sorted(df['arch_config'].unique().tolist())
    work_configs = sorted(df['work_config'].unique().tolist())
    
    # Convert to strings for dropdown options
    arch_options = [str(x) for x in arch_configs]
    work_options = [str(x) for x in work_configs]
    
    # Lists to store selected configs
    selected_arch_configs = []
    selected_work_configs = []
    
    # Dark theme colors
    BG_DARK = '#1e1e1e'
    BG_SECONDARY = '#252526'
    BG_TERTIARY = '#2d2d30'
    FG_PRIMARY = '#d4d4d4'
    FG_SECONDARY = '#808080'
    ACCENT = '#0078d4'
    ACCENT_HOVER = '#1c8ae6'
    BORDER = '#3c3c3c'
    
    # Create main window
    root = tk.Tk()
    root.title('Configuration Selection')
    root.geometry('1600x1200')
    root.configure(bg=BG_DARK)
    
    # Configure dark mode styles
    style = ttk.Style()
    style.theme_use('clam')
    
    # Frame styles
    style.configure('Dark.TFrame', background=BG_DARK)
    style.configure('Card.TFrame', background=BG_SECONDARY)
    
    # Label styles
    style.configure('Dark.TLabel', background=BG_DARK, foreground=FG_PRIMARY, font=('Segoe UI', 10))
    style.configure('Title.TLabel', background=BG_DARK, foreground=FG_PRIMARY, font=('Segoe UI', 16, 'bold'))
    style.configure('Header.TLabel', background=BG_SECONDARY, foreground=FG_PRIMARY, font=('Segoe UI', 10, 'bold'))
    style.configure('Card.TLabel', background=BG_SECONDARY, foreground=FG_PRIMARY, font=('Segoe UI', 10))
    
    # LabelFrame style
    style.configure('Card.TLabelframe', background=BG_SECONDARY, borderwidth=1, relief='solid')
    style.configure('Card.TLabelframe.Label', background=BG_SECONDARY, foreground=ACCENT, font=('Segoe UI', 11, 'bold'))
    
    # Button styles
    style.configure('Accent.TButton', 
                    background=ACCENT, 
                    foreground='white', 
                    font=('Segoe UI', 9, 'bold'),
                    padding=(12, 6))
    style.map('Accent.TButton',
                background=[('active', ACCENT_HOVER), ('pressed', ACCENT)],
                foreground=[('active', 'white')])
    
    style.configure('Secondary.TButton',
                    background=BG_TERTIARY,
                    foreground=FG_PRIMARY,
                    font=('Segoe UI', 9),
                    padding=(12, 6))
    style.map('Secondary.TButton',
                background=[('active', BORDER), ('pressed', BG_TERTIARY)])
    
    # Combobox style
    style.configure('Dark.TCombobox',
                    fieldbackground=BG_TERTIARY,
                    background=BG_TERTIARY,
                    foreground=FG_PRIMARY,
                    arrowcolor=FG_PRIMARY,
                    padding=5)
    style.map('Dark.TCombobox',
                fieldbackground=[('readonly', BG_TERTIARY)],
                selectbackground=[('readonly', ACCENT)])
    
    # Treeview style
    style.configure('Dark.Treeview',
                    background=BG_TERTIARY,
                    foreground=FG_PRIMARY,
                    fieldbackground=BG_TERTIARY,
                    borderwidth=0,
                    font=('Consolas', 10),
                    rowheight=25)
    style.configure('Dark.Treeview.Heading',
                    background=BG_SECONDARY,
                    foreground=FG_PRIMARY,
                    font=('Segoe UI', 9, 'bold'),
                    borderwidth=0,
                    padding=5)
    style.map('Dark.Treeview',
                background=[('selected', ACCENT)],
                foreground=[('selected', 'white')])
    
    # Scrollbar style
    style.configure('Dark.Vertical.TScrollbar',
                    background=BG_TERTIARY,
                    troughcolor=BG_SECONDARY,
                    borderwidth=0,
                    arrowcolor=FG_PRIMARY)
    
    # Create main frame
    main_frame = ttk.Frame(root, padding="20", style='Dark.TFrame')
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Title frame with logo
    title_frame = ttk.Frame(main_frame, style='Dark.TFrame')
    title_frame.pack(pady=(0, 20))
    
    # Load and display logo
    try:
        logo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logo.png')
        logo_img = Image.open(logo_path)
        logo_img = logo_img.resize((40, 40), Image.Resampling.LANCZOS)
        logo_photo = ImageTk.PhotoImage(logo_img)
        logo_label = ttk.Label(title_frame, image=logo_photo, style='Dark.TLabel')
        logo_label.image = logo_photo  # Keep a reference
        logo_label.pack(side=tk.LEFT, padx=(0, 10))
    except Exception as e:
        print(f"Could not load logo: {e}")  # Print error for debugging
    
    # Title label
    title_label = ttk.Label(title_frame, text='Configuration Selection', style='Title.TLabel')
    title_label.pack(side=tk.LEFT)
    
    # Create a vertical frame for the two sections
    sections_frame = ttk.Frame(main_frame, style='Dark.TFrame')
    sections_frame.pack(fill=tk.BOTH, expand=True)
    
    # ==================== ARCHITECTURE SECTION ====================
    arch_section = ttk.LabelFrame(sections_frame, text="  Architecture Configs  ", padding="15", style='Card.TLabelframe')
    arch_section.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 10))
    
    # Top row: dropdown and add button
    arch_top_frame = ttk.Frame(arch_section, style='Card.TFrame')
    arch_top_frame.pack(fill=tk.X, pady=(0, 15))
    
    ttk.Label(arch_top_frame, text='Select Config:', style='Card.TLabel').pack(side=tk.LEFT, padx=(0, 10))
    
    # Multi-select popup for architecture configs
    def open_arch_popup():
        popup = tk.Toplevel(root)
        popup.title('Select Architecture Configs')
        popup.configure(bg=BG_DARK)
        popup.geometry('300x400')
        popup.transient(root)
        popup.grab_set()
        vars = {}
        for i, cfg in enumerate(arch_options):
            var = tk.BooleanVar(value=cfg in selected_arch_configs)
            cb = tk.Checkbutton(popup, text=f'Config {cfg}', variable=var, bg=BG_DARK, fg=FG_PRIMARY, selectcolor=BG_TERTIARY, activebackground=BG_SECONDARY, anchor='w')
            cb.pack(fill='x', padx=20, pady=2, anchor='w')
            vars[cfg] = var
        def on_ok():
            selected_arch_configs.clear()
            for cfg, var in vars.items():
                if var.get():
                    selected_arch_configs.append(cfg)
            selected_arch_configs.sort(key=lambda x: int(x))
            arch_listbox.delete(0, tk.END)
            for c in selected_arch_configs:
                arch_listbox.insert(tk.END, f'  Config {c}')
            popup.destroy()
        ok_btn = ttk.Button(popup, text='OK', command=on_ok, style='Accent.TButton')
        ok_btn.pack(pady=10)
    arch_add_btn = ttk.Button(arch_top_frame, text='Select Configs', command=open_arch_popup, style='Accent.TButton')
    arch_add_btn.pack(side=tk.LEFT, padx=(0, 8))
    arch_remove_btn = ttk.Button(arch_top_frame, text='Clear All', command=lambda: (selected_arch_configs.clear(), arch_listbox.delete(0, tk.END)), style='Secondary.TButton')
    arch_remove_btn.pack(side=tk.LEFT)
    
    # Content area with preview and selected list
    arch_content = ttk.Frame(arch_section, style='Card.TFrame')
    arch_content.pack(fill=tk.BOTH, expand=True)
    
    # YAML preview area for architecture - with preview dropdown
    arch_yaml_frame = ttk.Frame(arch_content, style='Card.TFrame')
    arch_yaml_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))
    ttk.Label(arch_yaml_frame, text='Config Preview', style='Header.TLabel').pack(anchor='w', pady=(0, 8))
    # Dropdown for preview selection
    arch_preview_var = tk.StringVar(value=arch_options[0] if arch_options else '')
    arch_preview_dropdown = ttk.Combobox(arch_yaml_frame, textvariable=arch_preview_var, values=arch_options, width=12, state='readonly', style='Dark.TCombobox')
    arch_preview_dropdown.pack(anchor='w', pady=(0, 8))
    # Create Treeview with scrollbar
    arch_tree_frame = ttk.Frame(arch_yaml_frame, style='Card.TFrame')
    arch_tree_frame.pack(fill=tk.BOTH, expand=True)
    arch_tree = ttk.Treeview(arch_tree_frame, columns=('value',), show='tree headings', height=8, style='Dark.Treeview')
    arch_tree.heading('#0', text='Key', anchor='w')
    arch_tree.heading('value', text='Value', anchor='w')
    arch_tree.column('#0', width=180, stretch=True)
    arch_tree.column('value', width=120, stretch=True)
    arch_tree_scroll = ttk.Scrollbar(arch_tree_frame, orient='vertical', command=arch_tree.yview, style='Dark.Vertical.TScrollbar')
    arch_tree.configure(yscrollcommand=arch_tree_scroll.set)
    arch_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    arch_tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    
    # Listbox to show selected arch configs
    arch_selected_frame = ttk.Frame(arch_content, style='Card.TFrame')
    arch_selected_frame.pack(side=tk.RIGHT, fill=tk.Y)
    
    ttk.Label(arch_selected_frame, text='Selected', style='Header.TLabel').pack(anchor='w', pady=(0, 8))
    arch_listbox = tk.Listbox(arch_selected_frame, width=12, height=8,
                                font=('Consolas', 10), bg=BG_TERTIARY, fg=FG_PRIMARY,
                                selectbackground=ACCENT, selectforeground='white',
                                borderwidth=0, highlightthickness=0, relief='flat')
    arch_listbox.pack(fill=tk.Y, expand=True)
    
    # No longer needed: add_arch_config and remove_arch_config
    
    def populate_tree(tree, parent, data, prefix=''):
        """Recursively populate treeview with dict/list data."""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    node = tree.insert(parent, 'end', text=str(key), values=('',), open=False)
                    populate_tree(tree, node, value, prefix + '  ')
                else:
                    tree.insert(parent, 'end', text=str(key), values=(str(value),))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    node = tree.insert(parent, 'end', text=f'[{i}]', values=('',), open=False)
                    populate_tree(tree, node, item, prefix + '  ')
                else:
                    tree.insert(parent, 'end', text=f'[{i}]', values=(str(item),))
    
    def update_arch_preview(*args):
        # Show preview for the selected config in the dropdown
        for item in arch_tree.get_children():
            arch_tree.delete(item)
        cfg = arch_preview_var.get()
        if cfg:
            row = df[df['arch_config'] == int(cfg)].iloc[0]
            arch_path = row['arch_path']
            try:
                import yaml
                with open(arch_path, 'r') as f:
                    data = yaml.safe_load(f)
                if data:
                    populate_tree(arch_tree, '', data)
            except Exception as e:
                arch_tree.insert('', 'end', text='Error', values=(str(e),))
    arch_preview_var.trace('w', update_arch_preview)
    update_arch_preview()
    
    # ==================== WORKLOAD SECTION ====================
    work_section = ttk.LabelFrame(sections_frame, text="  Workload Configs  ", padding="15", style='Card.TLabelframe')
    work_section.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(10, 0))
    
    # Top row: dropdown and add button
    work_top_frame = ttk.Frame(work_section, style='Card.TFrame')
    work_top_frame.pack(fill=tk.X, pady=(0, 15))
    
    ttk.Label(work_top_frame, text='Select Config:', style='Card.TLabel').pack(side=tk.LEFT, padx=(0, 10))
    
    # Multi-select popup for workload configs
    def open_work_popup():
        popup = tk.Toplevel(root)
        popup.title('Select Workload Configs')
        popup.configure(bg=BG_DARK)
        popup.geometry('300x400')
        popup.transient(root)
        popup.grab_set()
        vars = {}
        for i, cfg in enumerate(work_options):
            var = tk.BooleanVar(value=cfg in selected_work_configs)
            cb = tk.Checkbutton(popup, text=f'Config {cfg}', variable=var, bg=BG_DARK, fg=FG_PRIMARY, selectcolor=BG_TERTIARY, activebackground=BG_SECONDARY, anchor='w')
            cb.pack(fill='x', padx=20, pady=2, anchor='w')
            vars[cfg] = var
        def on_ok():
            selected_work_configs.clear()
            for cfg, var in vars.items():
                if var.get():
                    selected_work_configs.append(cfg)
            selected_work_configs.sort(key=lambda x: int(x))
            work_listbox.delete(0, tk.END)
            for c in selected_work_configs:
                work_listbox.insert(tk.END, f'  Config {c}')
            popup.destroy()
        ok_btn = ttk.Button(popup, text='OK', command=on_ok, style='Accent.TButton')
        ok_btn.pack(pady=10)
    work_add_btn = ttk.Button(work_top_frame, text='Select Configs', command=open_work_popup, style='Accent.TButton')
    work_add_btn.pack(side=tk.LEFT, padx=(0, 8))
    work_remove_btn = ttk.Button(work_top_frame, text='Clear All', command=lambda: (selected_work_configs.clear(), work_listbox.delete(0, tk.END)), style='Secondary.TButton')
    work_remove_btn.pack(side=tk.LEFT)
    
    # Content area with preview and selected list
    work_content = ttk.Frame(work_section, style='Card.TFrame')
    work_content.pack(fill=tk.BOTH, expand=True)
    
    # YAML preview area for workload - with preview dropdown
    work_yaml_frame = ttk.Frame(work_content, style='Card.TFrame')
    work_yaml_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))
    ttk.Label(work_yaml_frame, text='Config Preview', style='Header.TLabel').pack(anchor='w', pady=(0, 8))
    # Dropdown for preview selection
    work_preview_var = tk.StringVar(value=work_options[0] if work_options else '')
    work_preview_dropdown = ttk.Combobox(work_yaml_frame, textvariable=work_preview_var, values=work_options, width=12, state='readonly', style='Dark.TCombobox')
    work_preview_dropdown.pack(anchor='w', pady=(0, 8))
    # Create Treeview with scrollbar
    work_tree_frame = ttk.Frame(work_yaml_frame, style='Card.TFrame')
    work_tree_frame.pack(fill=tk.BOTH, expand=True)
    work_tree = ttk.Treeview(work_tree_frame, columns=('value',), show='tree headings', height=8, style='Dark.Treeview')
    work_tree.heading('#0', text='Key', anchor='w')
    work_tree.heading('value', text='Value', anchor='w')
    work_tree.column('#0', width=180, stretch=True)
    work_tree.column('value', width=120, stretch=True)
    work_tree_scroll = ttk.Scrollbar(work_tree_frame, orient='vertical', command=work_tree.yview, style='Dark.Vertical.TScrollbar')
    work_tree.configure(yscrollcommand=work_tree_scroll.set)
    work_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    work_tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    
    # Listbox to show selected work configs
    work_selected_frame = ttk.Frame(work_content, style='Card.TFrame')
    work_selected_frame.pack(side=tk.RIGHT, fill=tk.Y)
    
    ttk.Label(work_selected_frame, text='Selected', style='Header.TLabel').pack(anchor='w', pady=(0, 8))
    work_listbox = tk.Listbox(work_selected_frame, width=12, height=8,
                                font=('Consolas', 10), bg=BG_TERTIARY, fg=FG_PRIMARY,
                                selectbackground=ACCENT, selectforeground='white',
                                borderwidth=0, highlightthickness=0, relief='flat')
    work_listbox.pack(fill=tk.Y, expand=True)
    
    # No longer needed: add_work_config and remove_work_config
    
    def update_work_preview(*args):
        # Show preview for the selected config in the dropdown
        for item in work_tree.get_children():
            work_tree.delete(item)
        cfg = work_preview_var.get()
        if cfg:
            row = df[df['work_config'] == int(cfg)].iloc[0]
            work_path = row['work_path']
            try:
                import yaml
                with open(work_path, 'r') as f:
                    data = yaml.safe_load(f)
                if data:
                    populate_tree(work_tree, '', data)
            except Exception as e:
                work_tree.insert('', 'end', text='Error', values=(str(e),))
    work_preview_var.trace('w', update_work_preview)
    update_work_preview()
    
    # ==================== BUTTONS ====================
    button_frame = ttk.Frame(root, padding="20", style='Dark.TFrame')
    button_frame.pack(fill=tk.X)
    
    # Center the buttons
    button_container = ttk.Frame(button_frame, style='Dark.TFrame')
    button_container.pack()
    
    def on_confirm():
        # Convert selected configs to integers for comparison
        selected_arch_ints = [int(x) for x in selected_arch_configs]
        selected_work_ints = [int(x) for x in selected_work_configs]
        
        # Filter df to get rows with both arch and work config in selected lists
        filtered_df = df[
            (df['arch_config'].isin(selected_arch_ints)) & 
            (df['work_config'].isin(selected_work_ints))
        ]
        
        from tkinter import messagebox
        if len(filtered_df) == 0:
            # Show confirmation dialog for no valid configs
            result = messagebox.askyesno(
                "No Valid Configurations",
                "Are you sure you want to exit?\nNo valid configurations found with the selected options."
            )
            if not result:
                return  # Don't close if user clicks "No"
        else:
            # Show confirmation dialog for valid configs
            result = messagebox.askyesno(
                "Confirm Selection",
                f"{len(filtered_df)} valid configuration(s) selected.\nAre you sure you want to exit?"
            )
            if not result:
                return
            
        _generate_runs(filtered_df, path)
        root.destroy()
    
    confirm_button = ttk.Button(button_container, text='Confirm Selection', command=on_confirm, style='Accent.TButton')
    confirm_button.pack(side=tk.LEFT, padx=10)
    
    close_button = ttk.Button(button_container, text='Cancel', command=root.destroy, style='Secondary.TButton')
    close_button.pack(side=tk.LEFT, padx=10)
    

    # Run the main loop
    root.mainloop()