from Workshop import Base, Metaheuristics
import concurrent.futures
import pandas as pd
import numpy as np
import threading
import random
import time

class genetic(Base.CodeBase):
    ALGO_TYPE = 'Genetic'
    def __init__(self, m: int = 1, n: int = 2, L: int = 0, node_info: np.ndarray = None, name: str = None, **kwargs) -> None:
        super().__init__(m, n, L, node_info, algo_type = self.ALGO_TYPE, name = name, **kwargs)
        
        if 'p_mutations' in kwargs.keys():
            self.p_mutations = kwargs['p_mutations']
        else:
            self.p_mutations = 0
        if 'k_elite' in kwargs.keys():
            self.k_elite = kwargs['k_elite']
        else:
            self.k_elite = 0

        self.margin = kwargs['margin']
        self.max_time = kwargs['max_time'] - self.margin

        self.rng = np.random.default_rng()
        self.decode_sols = dict()
        self.nc = 100
        self.vnd = Metaheuristics.vnd

    def init_encoded_sol(self):
        self.init_sol = self.sol.copy() 
        self.encoded_sols = self.rng.permuted(np.tile(np.arange(1,self.n-1), (self.nc,1)), axis=1)
    
    def decoder_aux(self, i):
        sol = self.init_sol.copy()
        metric = lambda S_aux, S: self.total_weight(S_aux)/(self.total_dist(S_aux)-self.total_dist(S))
        visited = {0,self.n-1}
        
        if time.time() > self.time_limit:
            return
        posible_nodes = self.encoded_sols[i]
        visited_size = 0
        while visited_size < len(visited):
            visited_size = len(visited)
            f_max = 0
            f_obj = 0
            index_max = None
            team_max = None
            for node in posible_nodes:
                for team in sol.keys():
                    S = sol[team].copy()
                    for index in range(1, len(S)):
                        if time.time() > self.time_limit:
                            return
                        
                        S_aux = np.insert(S, index, node)
                        if self.total_dist_mat(S_aux) <= self.L:
                            try:
                                f_obj = metric(S_aux, S)
                            except ZeroDivisionError:
                                f_obj = np.inf

                            if f_obj > f_max:
                                f_max = f_obj
                                index_max = index
                                team_max = team
                if team_max != None:
                    sol[team_max] = np.insert(sol[team_max], index_max, node)
                    posible_nodes = np.delete(posible_nodes, np.argwhere(posible_nodes==node))
                    visited.add(node)
                    f_max = 0
                    f_obj = 0
                    index_max = None
                    team_max = None
                    
                    if node in posible_nodes:
                        raise Exception(f'Synchronization error: {index}: {node}, {posible_nodes}')
        for team in sol.keys():
            sol[team] = pd.unique(sol[team])
        
        self.decode_sols[i] = (i, self.total_weight(np.concatenate([*sol.values()])), sol)
        return

    def decoder(self):
        self.counter_decoder += 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.nc) as executor:
            futures = [executor.submit(self.decoder_aux, i) for i in range(self.nc)]
            for future in concurrent.futures.as_completed(futures):
                future.result()

    def selecction(self):
        self.counter_selecction += 1
        candiates_info = np.array([*self.decode_sols.values()])
        candiates = np.random.choice(len(self.encoded_sols), size=(self.nc//2,2), replace=False)
        mask = candiates_info[:,1][candiates[:,0]] >= candiates_info[:,1][candiates[:,1]]
        select_aux = candiates_info[np.unique((np.r_[candiates[:,0][mask], candiates[:,1][~mask]]))]
        indices = np.argsort(select_aux[:,1])[::-1]
        self.selected = select_aux[indices][:,0].astype(int)

    def crossover(self):
        self.counter_crossover +=1
        self.children = np.empty((50,self.encoded_sols.shape[1]))
        for idx, (i, j) in enumerate(np.c_[self.selected[::2], self.selected[1::2]]):
            try:
                idx_c1, idx_c2 = np.argsort(self.dist_mat[self.encoded_sols[[i,j]].T[:,0].astype(int), self.encoded_sols[[i,j]].T[:,1].astype(int)])[:2]
            except Exception as e:
                print(e)
                print(i, j)
                print(self.encoded_sols[[i,j]].T[:,0], self.encoded_sols[[i,j]].T[:,1])
            self.children[idx*2] = np.r_[self.encoded_sols[i][:idx_c1], self.encoded_sols[j][idx_c1:]]
            self.children[idx*2+1] = np.r_[self.encoded_sols[i][:idx_c1], self.encoded_sols[j][idx_c1:]]

    def mutation(self):
        self.counter_mutation +=1
        num_mutants = int(self.p_mutations*len(self.children))+1
        mutants_idxs = np.random.choice(self.children.shape[0], size=num_mutants, replace=False)
        swap_indices = np.random.choice(self.children.shape[1], size=(num_mutants,2))
        self.children[mutants_idxs][np.arange(num_mutants), swap_indices[:,0]], self.children[mutants_idxs][np.arange(num_mutants), swap_indices[:,1]] = self.children[mutants_idxs][np.arange(num_mutants), swap_indices[:,1]], self.children[mutants_idxs][np.arange(num_mutants), swap_indices[:,0]]
        self.encoded_sols = np.r_[self.encoded_sols[self.selected], self.children]

    def best_solution(self):
        candidates_info = np.array([*self.decode_sols.values()])
        sorted_candidates = np.argsort(candidates_info[:,1])[::-1]
        
        if self.k_elite > 0:
            max_weight = 0
            elite_info = candidates_info[sorted_candidates][:self.k_elite][:,2]
            for sol_aux in elite_info:
                sol = self.fun_vnd(sol_aux)
                if self.total_weight(np.concatenate([*sol.values()])) > max_weight:
                    max_weight = self.total_weight(np.concatenate([*sol.values()]))
                    self.sol = sol.copy()
            
        else:
            self.sol = candidates_info[sorted_candidates][0][2].copy()

    def fun_vnd(self, sol_aux, verbose:bool = False):
        if time.time() > self.time_limit:
            return sol_aux
            
        return self.vnd(self.m, self.n, self.L, self.node_info, self.name,
                        init_sol=sol_aux,
                        time_limit=self.time_limit,
                        verbose=verbose).solve().sol.copy()
    
    def time_keeper(self, t):
        while time.time() < self.time_limit+self.margin:
            print(f'time: {time.time() - (self.time_limit - self.max_time):.2f}, decoder: {self.counter_decoder}, selecction: {self.counter_selecction}, crossover: {self.counter_crossover}, mutation: {self.counter_mutation}', end='\r')
            time.sleep(2)
        print(f'time: {time.time() - (self.time_limit - self.max_time):.2f}, decoder: {self.counter_decoder}, selecction: {self.counter_selecction}, crossover: {self.counter_crossover}, mutation: {self.counter_mutation}')

    @Base.CodeBase.execution_time
    def solve_aux(self):
        self.time_limit = time.time() + self.max_time
        self.counter_decoder = 0
        self.counter_selecction = 0
        self.counter_crossover = 0
        self.counter_mutation = 0
        threading.Thread(target=self.time_keeper, args=[1]).start()
        self.init_encoded_sol()
        while time.time() < self.time_limit:
            self.decoder()
            self.selecction()
            self.crossover()
            self.mutation()
        self.decoder()
        self.best_solution()
