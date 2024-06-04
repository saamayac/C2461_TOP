import os
import time
import numpy as np
import pandas as pd
import time
from openpyxl import load_workbook
from itertools import product, pairwise

class CodeBase:
    def __init__(self, m: int = 1, n: int = 2, L: int = 0, node_info: np.ndarray = None, algo_type: str = '',
                 name: str = None, verbose:bool = True, **kwargs) -> None:
        '''
        m: number of teams
        n: number of nodes
        L: time limit
        node_info: node information (x_coord, y_coord, weight)
        '''
        if 'time_limit' in kwargs.keys():
            self.time_limit = kwargs['time_limit']
        else:
            self.time_limit = np.inf

        
        self.m = int(m)
        self.n = int(n)
        self.L = float(L)
        self.node_info = node_info
        self.dist = lambda coor: np.linalg.norm(self.node_info[coor[0],:2]-self.node_info[coor[1],:2])
        self.total_weight = lambda rute: np.sum(self.node_info[:,2][[rute]])
        self.total_dist = lambda rute: sum(map(lambda coor: self.dist((coor[0], coor[1])), pairwise(rute)))

        # avoids unnecessary loops
        z = np.array(list(map(lambda i: complex(self.node_info[i,0],self.node_info[i,1]), range(n)))).reshape(-1,1)
        self.dist_mat = np.abs(z.T-z)
        self.total_dist_mat = lambda route: np.sum(self.dist_mat[route[:-1],route[1:]])
        
        self.sol_format: pd.DataFrame
        self.algo_type = algo_type
        self.name = name
        self.verbose = verbose
        try: 
            self.UB = dict(zip(np.loadtxt('UB.txt',  dtype=str, usecols=0), np.loadtxt('UB.txt', usecols = 1)))
        except Exception as e:
            print('UB not found:', e)
        self.init_sol()
        
        
    def init_sol(self):
        self.sol = dict()
        for i in range(self.m):
            self.sol[i] = np.array([0,self.n-1])
            
        self.max_rute_nodes = 2
        self.compute_time = 0

    @staticmethod
    def execution_time(func): 
        '''Decorator that reports the execution time.'''
        def wrap(self, *args, **kwargs): 
            start = time.time() 
            result = func(self, *args, **kwargs) 
            end = time.time() 
            self.compute_time = round(end-start,4)
            return result 
        return wrap
    
    def xlsx_format(self):
        self.max_rute_nodes = max(map(lambda i: self.sol[i].shape[0], range(self.m)))
        sol_aux = np.empty((self.m+1, self.max_rute_nodes+3))
        sol_aux[:,:] = np.nan
        final_weight = 0
        for i in range(self.m): 
            sol_aux[i, 0] = self.sol[i].shape[0]
            sol_aux[i, 1:self.sol[i].shape[0]+1] = self.sol[i]+1
            sol_aux[i, self.sol[i].shape[0]+1] = self.total_dist(self.sol[i])
            sol_aux[i, self.sol[i].shape[0]+2] = self.total_weight(self.sol[i])
            final_weight += int(sol_aux[i,self.sol[i].shape[0]+2])
        sol_aux[-1, 0] = final_weight
        sol_aux[-1, 1] = self.compute_time
        
        return sol_aux
        

    def solve(self, logs: bool = False):
        self.solve_aux()
        
        self.sol_format = self.xlsx_format()
        
        if self.verbose:
            if logs:
                self.resolve_logs()
            print('-'*70)
            if self.name:
                print(f'Solution found ({self.algo_type}): {self.name}')
            else:
                print(f'Solution found ({self.algo_type})')
            print('total weight:', self.sol_format[-1, 0])
            print('compute time:', self.sol_format[-1, 1])
            print('-'*70)
        
        return self

    def resolve_logs(self):
        f = open(f'logs_{self.algo_type}.txt', 'a')
        f.write('-'*50+'\n');
        if self.name:
            f.write(f'Solution found: {self.name}'+'\n');
            try:
                GAP = 100*(self.UB[self.name[:-4]]-self.sol_format[-1, 0])/self.sol_format[-1, 0]
                f.write(f'GAP: {GAP:.2f}%'+'\n')
            except Exception as e:
                print('Error ocurre while calculating the GAP:', e)
            finally: 
                f.write(f'total weight: {self.sol_format[-1, 0]}'+'\n')
                f.write(f'compute time: {self.sol_format[-1, 1]}'+'\n')
        else:
            f.write('Solution found'+'\n')
            f.write(f'total weight: {self.sol_format[-1, 0]}'+'\n')
            f.write(f'compute time: {self.sol_format[-1, 1]}'+'\n')
        f.write('-'*50+'\n')

    def solve_aux(self):
        pass
    
    def search(self, visited, metric):
        if time.time() > self.time_limit:
            return
        posible_nodes = set(range(self.n))
        visited_size = 0
        while visited_size < len(visited):
            visited_size = len(visited)
            f_max = 0
            f_obj = 0
            index_max = None
            team_max = None
            node_max = None
            for node in (posible_nodes-visited):
                for team in self.sol.keys():
                    S = self.sol[team].copy()
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
                                node_max = node
            if node_max != None:
                self.sol[team_max] = np.insert(self.sol[team_max], index_max, node_max)
                visited.add(node_max) 
    
    def dump(self, run_id:str='', dir_path = ''):
        try:
            sol_aux = pd.DataFrame(np.round(self.sol_format,7))
            fille_name = f'TOP_sebastian_{self.algo_type}_' + run_id + '.xlsx'
            path = os.path.join(dir_path, fille_name)
            print('Saving at', fille_name, '/', end = ' ')

            if os.path.exists(path):
                with pd.ExcelWriter(path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer: 
                    sol_aux.to_excel(writer, index=False, header=False, sheet_name=self.name[:-4])
            else:
                sol_aux.to_excel(path, index=False, header=False, sheet_name=self.name[:-4])
            print('done')
        except Exception as e:
            print(e)
            
    @staticmethod
    def run(dir_path_TOP:str = 'TOP2', algo = None, name:str = 'summary.xlsx', path_summary:str = '', 
            max_times:dict = None, dump:bool = True, run_id:str='', **kwargs):
        TOP_data_dict = Read(dir_path_TOP).read_all()
        file_names = list(map(lambda x: x, TOP_data_dict.keys()))
        file_names.sort(key=lambda str_: int(str_.split('.')[0][3:]))
        
        aux = np.zeros((len(file_names),5))
        
        for idx, file_name in enumerate(file_names):
            m, n, L, data = TOP_data_dict[file_name]
            
            if isinstance(max_times, dict):
                max_time = max_times[file_name.split('.')[0]]
            else:
                max_time = None
                
            obj = algo(m, n, L, data, file_name, max_time=max_time, **kwargs).solve()
            
            aux[idx, 0] = obj.UB[file_name[:-4]]
            aux[idx, 1] = (obj.UB[file_name[:-4]]-obj.sol_format[-1,0])/obj.UB[file_name[:-4]]
            aux[idx, 2] = obj.sol_format[-1,0]
            aux[idx, 3] = obj.sol_format[-1,1]
            aux[idx, 4] = max_time
            
            
            if dump:
                obj.dump(run_id)
        
        if dump:
            data_frame = pd.DataFrame(aux, index=file_names, columns=['UB', 'GAP', 'Obj fun', 'time', 'max time'])
            path = os.path.join(path_summary, name)
            sheet_name = f'{algo.ALGO_TYPE} ' + ''.join(f' {param}' for param_name, param in kwargs.items())
            print(f'Saving {algo.ALGO_TYPE} run summary at', f'{name} sheet: {sheet_name}', '/', end = ' ')
            
            if os.path.exists(path):
                with pd.ExcelWriter(path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer: 
                    data_frame.to_excel(writer, sheet_name=sheet_name)
            else:
                data_frame.to_excel(path, sheet_name=sheet_name)
            print('done')

class Read:
    def __init__(self, dir_path: str = '') -> None:
        self.dir_path = dir_path
        
    def read_fille(self, file_name: str = '') -> tuple:
        if file_name:
            path = os.path.join(self.dir_path, file_name)
            n,m,L = np.loadtxt(path, max_rows= 3)
            node_info = np.loadtxt(path, skiprows= 3)
            return (int(m), int(n), L, node_info)
            
            
    def read_all(self) -> dict:
        dict1 = dict()

        for file_name in os.listdir(self.dir_path):
            dict1[file_name] = self.read_fille(file_name)
        
        return dict1
