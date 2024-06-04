
from Workshop import Base
import numpy as np
import math

class constructive(Base.CodeBase):
    def __init__(self, m: int = 1, n: int = 2, L: int = 0, node_info: np.ndarray = None, name: str = None, verbose:bool = True, **kwargs) -> None:
        super().__init__(m, n, L, node_info, algo_type = 'constructivo', name = name, verbose=verbose, **kwargs)
        
    @Base.CodeBase.execution_time
    def solve_aux(self):
        init_visited = {0, self.n-1}
        metric = lambda S_aux, S: self.total_weight(S_aux)/(self.total_dist(S_aux)-self.total_dist(S))
        self.search(init_visited, metric)
            
class random1(Base.CodeBase):
    def __init__(self, m: int = 1, n: int = 2, L: int = 0, node_info: np.ndarray = None,
                 name: str = None, range_: (float, float) = (-1,1), num_sim : int = 10, verbose:bool = True, **kwargs) -> None:
        super().__init__(m, n, L, node_info, algo_type = 'ruido', name = name, verbose=verbose, **kwargs)
        self.num_sim = num_sim
        self.alt_node_info: np.ndarray
        self.range_ = range_
        self.random()
        self.total_weight_alt = lambda rute: np.sum(self.alt_node_info[:,2][[rute]])
         
    def random(self):
        self.alt_node_info = self.node_info.copy()
        self.alt_node_info[:,2] = self.node_info[:,2] + np.random.uniform(low=self.range_[0], high=self.range_[1], size=self.n)
        
        
    @Base.CodeBase.execution_time
    def solve_aux(self):
        max_final_weight = -1
        for i in range(self.num_sim):
            self.random()
            self.init_sol()
            
            init_visited = {0, self.n-1}
            metric = lambda S_aux, S: self.total_weight_alt(S_aux)/(self.total_dist(S_aux)-self.total_dist(S))
            
            self.search(init_visited, metric)
            
            final_weight =  sum(map(lambda i: self.total_weight(self.sol[i]), range(self.m)))
            if max_final_weight < final_weight:
                aux = self.sol.copy()
                max_final_weight = final_weight
        self.sol = aux.copy()
                
class random2(Base.CodeBase):
    def __init__(self, m: int = 1, n: int = 2, L: int = 0, node_info: np.ndarray = None, alpha:float=0.1,
                 name: str = None, num_sim : int = 10, verbose:bool = True, **kwargs) -> None:
        super().__init__(m, n, L, node_info, algo_type = 'eliminacion', name = name, verbose=verbose, **kwargs)
        self.alpha = alpha
        self.num_sim = num_sim
        
    @Base.CodeBase.execution_time
    def solve_aux(self):
        max_final_weight = -1
        
        for i in range(self.num_sim):
            self.init_sol()
            
            init_visited = {0, self.n-1}
            num_del_nodes = int(np.ceil(self.alpha * self.n))
            del_nodes = np.random.choice(self.n-1, size=num_del_nodes)+1
            [init_visited.add(del_node) for del_node in del_nodes]
            
            metric = lambda S_aux, S: self.total_weight(S_aux)/(self.total_dist(S_aux)-self.total_dist(S))
            
            self.search(init_visited, metric)
            
            final_weight = sum(map(lambda i: self.total_weight(self.sol[i]), range(self.m)))
            if max_final_weight < final_weight:
                aux = self.sol
                max_final_weight = final_weight
        self.sol = aux
        
