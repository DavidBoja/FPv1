

import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import *

T_COLS = ['T00','T01','T02','T03',
        'T10','T11','T12','T13',
        'T20','T21','T22','T23',
        'T30','T31','T32','T33']

def RRE(R_gt,R_estim):
    '''
    R_gt: numpy array dim (3,3)
    R_estim: np array dim (3,3)
    Returns: angle measurement in degrees
    '''

    # tnp = np.matmul(R_estim.T,R_gt)
    tnp = np.matmul(np.linalg.inv(R_estim),R_gt)
    tnp = (np.trace(tnp) -1) /2
    tnp = np.clip(tnp, -1, 1)
    tnp = np.arccos(tnp) * (180/np.pi)
    return tnp

def RTE(t_gt,t_estim):
    '''
    t_gt: np array dim (3,)
    t_estim: np array dim (3,)
    '''

    return np.linalg.norm(t_gt - t_estim,ord=2)

def process_single_gt_example(ind,benchmark):
    
    scan_name = benchmark.loc[ind,'Scan']
    viewpoint_i = benchmark.loc[ind,'Viewpoint_i']
    viewpoint_j = benchmark.loc[ind,'Viewpoint_j']
    overlap = benchmark.loc[ind,'overlap']
    T_gt = np.array(benchmark.loc[ind,T_COLS]).astype('float64').reshape(4,4)

    return_dict = {'scan_name':scan_name,
                   'viewpoint_i':viewpoint_i,
                   'viewpoint_j':viewpoint_j,
                   'overlap':overlap,
                   'transformation_GT':T_gt,
                   }

    return return_dict

def find_same_pred_example(gt_dict,results):

    scan_name = gt_dict['scan_name']
    vi = gt_dict['viewpoint_i']
    vj = gt_dict['viewpoint_j']

    filter_ = (results['Scan'] == scan_name) & \
              (results['Viewpoint_i'] == vi) & \
              (results['Viewpoint_j'] == vj)  

    results = results[filter_]

    T_pred = np.array(results[T_COLS]).astype('float64').reshape(4,4)

    return {'transformation_PRED': T_pred}



def evaluate(config):

    THR_ROT = config['THRESHOLD-ROTATION']
    THR_TRANS = config['THRESHOLD-TRANSLATION']

    rres = []
    rre_all = []
    rtes = []
    rte_all = []
    TP = 0 # true positives for recall
    tp_indices = []

    benchmark_path = config['BENCHMARK-CSV-FULL-PATH']
    results_path = config['RESULTS-CSV-FULL-PATH']
    
    benchmark = pd.read_csv(benchmark_path)
    N = benchmark.shape[0]
    results = pd.read_csv(results_path)

    for ind in tqdm(range(N)):

        # the goal is to register pcj to pci
        gt_dict = process_single_gt_example(ind,benchmark)
        pred_dict = find_same_pred_example(gt_dict,results)

        R_gt = gt_dict['transformation_GT'][:3,:3]
        t_gt = gt_dict['transformation_GT'][:3,3]

        R_pred = pred_dict['transformation_PRED'][:3,:3]
        t_pred = pred_dict['transformation_PRED'][:3,3]

        rre = RRE(R_gt,R_pred)
        rte = RTE(t_gt,t_pred)

        rre_all.append(rre)
        rte_all.append(rte)

        if (rre < THR_ROT) and (rte < THR_TRANS):
            TP += 1
            tp_indices.append(ind)

            rres.append(rre)
            rtes.append(rte)

    print('RESULTS:')
    print(f'RR(%): {TP/N}')

    print(f'RRE (only true positive) (degrees): {np.mean(rres)}')
    print(f'RTE (only true positie) (cm): {np.mean(rtes)}')

    print(f'RRE (all) (degrees): {np.mean(rre_all)}')
    print(f'RTE (all) (cm): {np.mean(rte_all)}')


if __name__ == '__main__':
    
    with open('config.yaml','r') as f:
        config = yaml.safe_load(f)

    config = config['EVALUATE']

    evaluate(config)