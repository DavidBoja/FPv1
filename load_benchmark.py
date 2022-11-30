

import yaml
import open3d as o3d
import numpy as np
import pandas as pd
import os.path as osp
import pickle

from utils import *

T_COLS = ['T00','T01','T02','T03',
        'T10','T11','T12','T13',
        'T20','T21','T22','T23',
        'T30','T31','T32','T33']

def process_single_example(ind,benchmark,faust_data_path,indices_path):
    
    scan_name = benchmark[ind,'Scan']
    viewpoint_i = benchmark[ind,'Scan']
    viewpoint_j = benchmark[ind,'Scan']
    overlap = benchmark[ind,'Scan']
    T_gt = np.array(benchmark[ind,T_COLS]).astype('float64').reshape(4,4)

    indices_path = osp.join(indices_path,
                        f'indices_{scan_name}.pickle')
    with open(indices_path,'rb') as f:
        indices = pickle.load(f)

    pc_o3d = o3d.io.read_point_cloud(osp.join(faust_data_path,scan_name+'.ply'))
    pc = np.asarray(pc_o3d.points)   

    pci = pc[indices[viewpoint_i]]
    pci = homo_matmul(pci,T_gt)
    pcj = pc[indices[viewpoint_j]]

    return_dict = {'scan_name':scan_name,
                   'viewpoint_i':viewpoint_i,
                   'viewpoint_j':viewpoint_j,
                   'overlap':overlap,
                   'transformation_GT':T_gt,
                   'pci':pci,
                   'pcj':pcj}

    return return_dict


def load_benchmark(config):

    faust_data_path = config['FAUST-DATA-PATH']
    partial_view_indices_path = osp.join(config['BENCHMARK-CSV-PATH'],'indices')
    benchmark_path = osp.join(config['BENCHMARK-CSV-PATH'],config['BENCHMARK-CSV-NAME'])
    
    benchmark = pd.read_csv(benchmark_path)
    N = benchmark.shape[0]

    for ind in range(N):

        # the goal is to register pcj to pci
        return_dict = process_single_example(ind,benchmark,faust_data_path,partial_view_indices_path)



if __name__ == '__main__':
    
    with open('config.yaml','r') as f:
        config = yaml.safe_load(f)

    config = config['LOAD-BENCHMARK']

    load_benchmark(config)