class Parameter():
    def __init__(self, value, sweep):
        self.value = value
        self.sweep = sweep

    def set_vertex(self, vertex):
        self.vertex = vertex

    def get_vertex(self):
        return self.vertex

    def get_value(self):
        return self.value
    
    def get_sweep(self):
        return self.sweep
    
class SweepParameter():
    def __init__(self, d: dict):
        self.d = d

    def get_dict(self):
        return self.d
    
class ArchitectureParameter():
    def __init__(self, d: dict):
        self.d = d

    def get_dict(self):
        return self.d
    
class WorkloadParameter():
    def __init__(self, d: dict):
        self.d = d

    def get_dict(self):
        return self.d
    
class ArchitectureWorkloadParameter():
    def __init__(self, d: dict):
        self.d = d

    def get_dict(self):
        return self.d