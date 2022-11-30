
import yaml
import open3d as o3d
import numpy as np
import pandas as pd
import os.path as osp
import pickle
import sys

from utils import *

T_COLS = ['T00','T01','T02','T03',
        'T10','T11','T12','T13',
        'T20','T21','T22','T23',
        'T30','T31','T32','T33']

BLUE = list((np.array([153,191,237])-50)/255)
RED = list(np.array([255,0,0])/255)
GRAY = list(np.array([120,120,120])/255)
YELLOW = list(np.array([250,212,8])/255)

def visualize_benchmark(config):

    faust_data_path = config['FAUST-DATA-PATH']
    faust_scan_name = config['FAUST-SCAN-NAME']
    partial_view_indices_path = osp.join(config['BENCHMARK-CSV-PATH'],'indices')
    partial_viewpoints_path = osp.join(config['BENCHMARK-CSV-PATH'],'viewpoints')
    benchmark_path = osp.join(config['BENCHMARK-CSV-PATH'],config['BENCHMARK-CSV-NAME'])
    vi = config['VIEWPOINT1']
    vj = config['VIEWPOINT2']
    
    benchmark = pd.read_csv(benchmark_path)
    N = benchmark.shape[0]

    bench_filtered = benchmark[benchmark['Scan'] == faust_scan_name]
    bench_filtered.reset_index(inplace=True,drop=True)
    N_filtered = bench_filtered.shape[0]

    # print(bench_filtered.loc[4,'Viewpoint_i'])
    # input()

    vis = [int(bench_filtered.loc[i,'Viewpoint_i'].split('viewpoint')[1]) for i in range(N_filtered)]
    vjs = [int(bench_filtered.loc[i,'Viewpoint_j'].split('viewpoint')[1]) for i in range(N_filtered)]

    possible_viewpoints = [(vis[i],vjs[i]) for i in range(len(vis))]

    if (vi,vj) in possible_viewpoints:
        index = possible_viewpoints.index((vi,vj))
    elif (vj,vi) in possible_viewpoints:
        index = possible_viewpoints.index((vi,vj))
    else:
        print('There is no such pair of viewpoints in the benchmark.')
        print(f'For scan {faust_scan_name} the options are:')
        print(possible_viewpoints + [(j,i) for (i,j) in possible_viewpoints])
        sys.exit('Stopping visualization.')

    # benchmark_example = bench_filtered.iloc[index]

        
    scan = bench_filtered.loc[index,'Scan']
    vi = bench_filtered.loc[index,'Viewpoint_i']
    vj = bench_filtered.loc[index,'Viewpoint_j']
    T_gt = np.array(bench_filtered.loc[index,T_COLS]).astype('float64').reshape(4,4)
        
    # load pc
    pc_o3d = o3d.io.read_point_cloud(osp.join(faust_data_path,scan+'.ply'))
    pc = np.asarray(pc_o3d.points)    
    
    # load partial view indices
    indices_path = osp.join(partial_view_indices_path,
                            f'indices_{faust_scan_name}.pickle')
    with open(indices_path,'rb') as f:
        indices = pickle.load(f)
    viewpoints_path = osp.join(partial_viewpoints_path,
                               f'viewpoints_{faust_scan_name}.pickle')
    with open(viewpoints_path,'rb') as f:
        viewpoints = pickle.load(f)
    
        
    # create partial views and viewpoints
    pci = pc[indices[vi]]
    pcj = pc[indices[vj]]

    # vi_xyz = viewpoints[vi]
    # vj_xyz = viewpoints[vj]
    

    pcj_o3d = o3d.geometry.PointCloud()
    pcj_o3d.points = o3d.utility.Vector3dVector(pcj)
    pcj_o3d.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcj_o3d.paint_uniform_color(BLUE)

            
    pci_GT = homo_matmul(pci,T_gt)
    pci_GT_o3d = o3d.geometry.PointCloud()
    pci_GT_o3d.points = o3d.utility.Vector3dVector(pci_GT)
    pci_GT_o3d.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pci_GT_o3d.paint_uniform_color(YELLOW)
    
    pcj_GT = homo_matmul(pcj,T_gt)
    pcj_GT_o3d = o3d.geometry.PointCloud()
    pcj_GT_o3d.points = o3d.utility.Vector3dVector(pcj_GT)
    pcj_GT_o3d.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcj_GT_o3d.paint_uniform_color(BLUE)
    
    
    # visualize initial case
    o3d.visualization.draw_geometries([pci_GT_o3d,pcj_o3d],
                                      window_name=f'Benchmark examples to register')
    
    # visualize GT case
    o3d.visualization.draw_geometries([pci_GT_o3d,pcj_GT_o3d],
                                      window_name=f'Ground truth alignment')


if __name__ == '__main__':
    
    with open('config.yaml','r') as f:
        config = yaml.safe_load(f)

    config = config['VISUALIZE-BENCHMARK']

    visualize_benchmark(config)