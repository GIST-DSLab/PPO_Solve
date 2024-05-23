import json
import numpy as np
import os
from typing import List
from numpy.typing import NDArray

from arcle.loaders import ARCLoader
from arcle.loaders import MiniARCLoader

from arcle.envs.arcenv import AbstractARCEnv
from arcle.loaders import ARCLoader
from gymnasium.core import ObsType, ActType
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os
from io import BytesIO
import gym
import copy
import pdb
from PIL import Image
import random
import torch
from numpy.typing import NDArray
from typing import Dict,Optional,Union,Callable,List, Tuple, SupportsFloat, SupportsInt, SupportsIndex, Any
from functools import wraps
from numpy import ma
import json
import pickle


class SizeConstrainedLoader(ARCLoader):
    def __init__(self, size, train=True) -> None:
        self.size = size
        super().__init__(train=train)
    
    def parse(self, **kwargs):
        
        dat = []

        for p in self._pathlist:
            with open(p) as fp:
                problem = json.load(fp)

                ti: List[NDArray] = []
                to: List[NDArray] = []
                ei: List[NDArray] = []
                eo: List[NDArray] = []


                for d in problem['train']:
                    inp = np.array(d['input'],dtype=np.uint8)
                    oup = np.array(d['output'],dtype=np.uint8)
                    if inp.shape[0] > self.size or inp.shape[1] > self.size or oup.shape[0] > self.size or oup.shape[1] > self.size:
                        continue
                    ti.append(inp)
                    to.append(oup)

                for d in problem['test']:
                    inp = np.array(d['input'],dtype=np.uint8)
                    oup = np.array(d['output'],dtype=np.uint8)
                    if inp.shape[0] > self.size or inp.shape[1] > self.size or oup.shape[0] > self.size or oup.shape[1] > self.size:
                        continue
                    ei.append(inp)
                    eo.append(oup)

                if len(ti) == 0:
                    continue

                desc = {'id': os.path.basename(fp.name).split('.')[0]}
                dat.append((ti,to,ei,eo,desc))
                
        return dat
    
class MiniARCLoader(MiniARCLoader):
    def __init__(self) -> None:
        super().__init__()
    
    def parse(self, **kwargs):
        
        dat = []

        for p in self._pathlist:
            with open(p) as fp:
                fpdata = fp.read().replace('null', '"0"')
                problem = json.loads(fpdata)

                ti: List[NDArray] = []
                to: List[NDArray] = []
                ei: List[NDArray] = []
                eo: List[NDArray] = []

                for d in problem['train']:
                    ti.append(np.array(d['input'],dtype=np.uint8))
                    to.append(np.array(d['output'],dtype=np.uint8))
                
                for d in problem['test']:
                    ei.append(np.array(d['input'],dtype=np.uint8))
                    eo.append(np.array(d['output'],dtype=np.uint8))

                fns = os.path.basename(fp.name).split('_')
                desc = {'id': fns[-1].split('.')[-2], 'description': ' '.join(fns[0:-1]).strip() }

                dat.append((ti,to,ei,eo,desc))
                
        return dat

class EntireSelectionLoader(ARCLoader):
    def __init__(self) -> None:
        super().__init__()
    
    def parse(self, **kwargs):
        dat = []

        ti: List[NDArray] = []
        to: List[NDArray] = []
        ei: List[NDArray] = []
        eo: List[NDArray] = []

        if not os.path.exists('/home/jovyan/ppo_cat/dataset/train_150.pkl'):
            for d in problem['train']:
                    inp = np.array(d['input'],dtype=np.uint8)
                    oup = np.array(d['output'],dtype=np.uint8)
                    ti.append(inp)
                    to.append(oup)
        else:
            with open('/home/jovyan/ppo_cat/dataset/train_150.pkl', 'rb') as f:
                full_list = pickle.load(f)                    
            ti = full_list[0] # list type으로 바꾸기
            to = full_list[1]

        if not os.path.exists('/home/jovyan/ppo_cat/dataset/eval_150.pkl'):
            while len(ei) < 100:
                grid = np.random.randint(0, 10, size=(3, 3))
                if not any((grid == x).all() for x in ti):
                    ei.append(grid)
                    transformed_grid = horizontal_flip(rotate_right(grid))
                    eo.append(transformed_grid)
                else:
                    continue
            ei = np.array(ei)
            eo = np.array(eo)
            full_list = np.stack((ei, eo))
            np.save(f'/home/dslab/arcle-trajectory/augmented_task/Task179/eval_diagonal.npy', full_list)
        else:
            with open('/home/jovyan/ppo_cat/dataset/eval_150.pkl', 'rb') as f:
                full_list = pickle.load(f)                    
            ei = full_list[0] # list type으로 바꾸기
            eo = full_list[1]

        #desc = {'id': os.path.basename(fp.name).split('.')[0]}
        desc = {'id': 'aug150'}
        dat.append((ti,to,ei,eo,desc))
            


    
        print(len(ti), len(to), len(ei), len(eo))
        return dat



def rotate_left(state):
    temp_state = copy.deepcopy(state['grid'] if 'grid' in state else state)
    rotate_state = []
    for  i in range(3):
        temp = []
        for j in range(3):
            temp.append(temp_state[j][2-i])
        rotate_state.append(temp)
    return rotate_state

# rotate_right function is a clockwise rotation about the given state.
def rotate_right(state):
    temp_state = copy.deepcopy(state['grid'] if 'grid' in state else state)
    rotate_state = []
    for  i in range(3):
        temp = []
        for j in range(3):
            temp.append(temp_state[2-j][i])
        rotate_state.append(temp)
    return rotate_state

# vertical_flip function is a flip by y-axis about the given state
def vertical_flip(state): 
    temp_state = copy.deepcopy(state['grid'] if 'grid' in state else state)
    rotate_state = []
    for  i in range(3):
        temp = []
        for j in range(3):
            temp.append(temp_state[2-i][j])
        rotate_state.append(temp)
    return rotate_state

# horizontal_flip function is a flip by x-axis about the given state.
def horizontal_flip(state): 
    temp_state = copy.deepcopy(state['grid'] if 'grid' in state else state)
    rotate_state = []
    for  i in range(3):
        temp = []
        for j in range(3):
            temp.append(temp_state[i][2-j])
        rotate_state.append(temp)
    return rotate_state
    

# if not os.path.exists(f'/home/dslab/arcle-trajectory/augmented_task/train_diagonal.pkl'):
# ti.append(np.array([np.array(np.random.randint(0, 10, size=(3, 3)).tolist()) for _ in range(1000)])) # 타입 체크
# to.append(np.array([np.array(horizontal_flip(rotate_right(target))) for target in ti[0]]))
# full_list = np.stack((ti[0], to[0]))
# np.save(f'/home/dslab/arcle-trajectory/augmented_task/Task179/train_diagonal.npy', full_list)
# ti = ti[0]
# to = to[0]

        # if mode == 'eval':
        #     input_list = []
        #     for  in range(self.aug_eval_num):
        #         while True:
        #             temp = np.random.randint(0, 10, size=(3, 3)).tolist()
        #             if train_set == None or str(temp) not in train_set:
        #                 input_list.append(np.array(temp))
        #                 break
        #     output_list = [np.array(horizontal_flip(rotate_right(target))) for target in input_list]
                # else:                    
                #     with open(f'/home/dslab/arcle-trajectory/augmented_task/train_diagonal.pkl', 'rb') as f:
                #         full_list = pickle.load(f)      
                #     ti = full_list[0] # list type으로 바꾸기
                #     to = full_list[1]
                    