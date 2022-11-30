
import yaml
import os
import os.path as osp
from tqdm import tqdm
import open3d as o3d
import numpy as np
import pickle
import plotly.graph_objects as go
import pandas as pd
from scipy.spatial.transform import Rotation as ScipyRot
import random

from icosahaedron import create_ico, scale_ico, split_icosahaedron
from visualization import circle_pts_under_floor, draw_icosahaedron, draw_partial_pointclouds, draw_spheres, draw_viewpoints, draw_xyz_axs


def create_faust_partial_indices(config):
    '''
    1. Create icosahaedron with "ICOSAHAEDRON-NR-DIVISIONS" divisions around each faust scan.
        # https://en.wikipedia.org/wiki/List_of_geodesic_polyhedra_and_Goldberg_polyhedra#Icosahedral
        # number of resulting points
    2. Each point from the icosahaedron acts as a viewpoint
    3. Create partial scan from desired viewpoint and save it as indices from the original complete scan
    Input:   config: config.yaml file with all the parameters
    Returns: None -- saves to disk the indices and viewpoints in the folder config['SAVE-TO']
    '''

    indices_path = osp.join(config['SAVE-TO'],'indices')
    viewpoints_path = osp.join(config['SAVE-TO'],'viewpoints')

    if not osp.exists(indices_path):
        os.makedirs(indices_path)
    if not osp.exists(viewpoints_path):
        os.makedirs(viewpoints_path)

    scan_names = sorted([x for x in os.listdir(config['FAUST-DATA-PATH']) if '.ply' in x])

    # create icosahaedron
    ico_v, ico_f = create_ico() # vertices, faces
    ico_v = scale_ico(ico_v,1)

    for division in range(config['ICOSAHAEDRON-NR-DIVISIONS']):
        ico_v, ico_f = split_icosahaedron(ico_v,ico_f)
        ico_v = scale_ico(ico_v,1)
        # print(np.sqrt(np.sum((ico_v)**2,axis=1)))
        
    ico_v_sc_orig = scale_ico(ico_v, config['ICOSAHEDRON-SCALE'])

    for full_name in tqdm(scan_names):
        indices = {}
        viewpoints = {}
        ico_v_sc = ico_v_sc_orig.copy()
        
        scan_path = osp.join(config['FAUST-DATA-PATH'],full_name)
        pcd = o3d.io.read_point_cloud(scan_path)
        
        # variables
        diameter = np.linalg.norm(
            np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
        radius = diameter * 1000

        # translate scan and icosahaedron points so xz is floor
        min_point = pcd.get_min_bound()
        pcd.translate(-min_point)
        ico_v_sc_trans = ico_v_sc - min_point
        # remove icosahaedron points with negative y ax -- imposible viewing point (below floor)
        ico_v_sc_trans_pos = ico_v_sc_trans[ico_v_sc_trans[:,1] > 0]
        
        # iterate over viewpoints and get partial viewpoints -- save indices of partial viewpoint
        diameter = np.linalg.norm(
            np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
        radius = diameter * 1000
        for ci,camera in enumerate(ico_v_sc_trans_pos):
            _, pt_map = pcd.hidden_point_removal(camera, radius)
            
            viewpoints[f'viewpoint{ci}'] = camera
            indices[f'viewpoint{ci}'] = pt_map
            
        # save partial indices and camera viewpoints for current pc
        name = full_name.split('.ply')[0]
        
        saving_name = osp.join(indices_path, f'indices_{name}.pickle')
        with open(saving_name, 'wb') as handle:
            pickle.dump(indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        saving_name = osp.join(viewpoints_path,f'viewpoints_{name}.pickle')
        with open(saving_name, 'wb') as handle:
            pickle.dump(indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

        if config['VISUALIZE']:
            fig = go.Figure()
            pts = np.asarray(pcd.points)
            colors_pts = np.arange(pts.shape[0])

            fig = draw_viewpoints(fig,ico_v_sc_trans_pos)
            fig = draw_partial_pointclouds(fig,
                                            ico_v_sc_trans_pos,
                                            pts,
                                            colors_pts,
                                            indices)
            fig = draw_icosahaedron(fig,ico_v_sc_trans,ico_f)
            
            # circle viewpoints which were discarded
            fig = circle_pts_under_floor(fig,ico_v_sc_trans)
            

            # add sphere
            fig = draw_spheres(config['ICOSAHEDRON-SCALE'], fig, clr='#ffff00', 
                               dist=-min_point,name='Sphere')

            fig = draw_xyz_axs(fig,k=2)

            fig.update_layout(scene_aspectmode='data',
                                width=900, height=700,
                                title=f'FAUST EXAMPLE {name}',
                                showlegend=True
                                )
            fig.show()
            input('Waiting for you to look at the pretty graphs!')
    
def calc_overlap(config):
    '''
    For each scan of the FAUSt dataset, find overlaps between all of its partial scans created
    and save it as csv
    Input:   config.yaml file with desired parameters
    Returns: overlap: pandas dataframe with columns ['Scan','Viewpoint_i','Viewpoint_j','overlap']
                      denoting the overlap in percentages between all partial views
                      saving the csv to disk in config['SAVE-TO']
    '''

    scan_names = sorted([x for x in os.listdir(config['FAUST-DATA-PATH']) if '.ply' in x])
    all_indices_path = osp.join(config['SAVE-TO'],'indices')

    overlap_df_columns = ['Scan','Viewpoint_i','Viewpoint_j','overlap']
    overlap_df = pd.DataFrame(columns=overlap_df_columns)

    for full_name in tqdm(scan_names):
        name = full_name.split('.ply')[0]
        indices_path = osp.join(all_indices_path, f'indices_{name}.pickle')
        
        with open(indices_path,'rb') as f:
            indices = pickle.load(f)
        
        # scan_path = os.path.join(config['FAUST-DATA-PATH'],full_name)
        # pcd = o3d.io.read_point_cloud(scan_path)
        
        all_viewpoints = list(indices.keys())
        
        for ci in range(len(all_viewpoints)-1):
            for cj in range(ci+1, len(all_viewpoints)):

                indices1 = indices[all_viewpoints[ci]]
                indices2 = indices[all_viewpoints[cj]]
                
                M = len(indices1)
                m = len(set.intersection(set(indices1),set(indices2)))
                overlap = (m / M) * 100
                
                # save overlaps
                overlap_ij = pd.Series([name,
                                        all_viewpoints[ci],
                                        all_viewpoints[cj],
                                        overlap
                                        ],
                            index=overlap_df_columns)
                overlap_df = overlap_df.append(overlap_ij,ignore_index=True)
                                                        
    overlap_df.to_csv(osp.join(config['SAVE-TO'], 'overlaps.csv'), 
                        index=False)

def filter_overlap(config):
    '''
    For computed overlap, filter only those that meet the requiremets
    Input:   config.yaml file with desired parameters
    Returns: overlap-filtered: pandas dataframe with columns ['Scan','Viewpoint_i','Viewpoint_j','overlap']
                      denoting the overlap in percentages between all partial views
                      the rows are filtered by taking only those where overlap is bigger than
                      config['DESIRED-OVERLAP-LOWER-BOUND'] and smaller than config['DESIRED-OVERLAP-UPPER-BOUND']
                      + saving the overlap-filtered.csv to disk in config['SAVE-TO']
    '''

    overlaps = pd.read_csv(osp.join(config['SAVE-TO'], 'overlaps.csv'),
                            index_col=False)

    UB = config['DESIRED-OVERLAP-UPPER-BOUND']
    LB = config['DESIRED-OVERLAP-LOWER-BOUND']

    mask = (overlaps['overlap'] >= LB) & (overlaps['overlap'] <= UB)
        
    overlaps = overlaps[mask]
    overlaps.reset_index(inplace=True,drop=True)

    saving_name = osp.join(config['SAVE-TO'],'overlaps-filtered.csv')
    overlaps.to_csv(saving_name,index=False)

    return overlaps

def create_benchmark_from_overlaps(config):
    '''
    Given the overlap-filtered.csv, create a random rotation and translation to use as 
    benchmark and save it along each example in the csv
    Input:    config: config.yaml file with the desired parametrs
    Returns:  None - saves to disk the benchmark csv under the name config['BENCHMARK-CSV-NAME']
    '''

    overlaps_filtered = pd.read_csv(osp.join(config['SAVE-TO'], 'overlaps-filtered.csv'),
                                    index_col=False)

    T_names = ['T00','T01','T02','T03',
                'T10','T11','T12','T13',
                'T20','T21','T22','T23',
                'T30','T31','T32','T33']

    generated_transformations = pd.DataFrame(columns=T_names)

    LB_ROT = config['DESIRED-ROTATION-LOWER-BOUND-DEGREES']
    UB_ROT = config['DESIRED-ROTATION-UPPER-BOUND-DEGREES']
    LB_TRANS = config['DESIRED-TRANSLATION-LOWER-BOUND-METERS']
    UB_TRANS = config['DESIRED-TRANSLATION-UPPER-BOUND-METERS']

    for rp in tqdm(range(overlaps_filtered.shape[0])):
                
        # random rotation between LB_ROT to UB_ROT degrees
        rand_euler_angles = np.random.randint(LB_ROT,UB_ROT,3)
        R_rand = ScipyRot.from_euler('xyz',
                    rand_euler_angles,
                    degrees=True).as_matrix()
        
        # random translation from -50 to 50cm
        tx = random.uniform(LB_TRANS, UB_TRANS)
        ty = random.uniform(LB_TRANS, UB_TRANS)
        tz = random.uniform(LB_TRANS, UB_TRANS)
        t_rand = np.array([tx,ty,tz])
        
        T_rand = np.eye(4)
        T_rand[:3,:3] = R_rand
        T_rand[:3,3] = t_rand
        
        T_rand_series = pd.Series(T_rand.ravel(),index=T_names)
        
        generated_transformations.loc[rp,T_names] = T_rand_series
    
    benchmark = pd.concat([overlaps_filtered, generated_transformations], 
                            axis=1, join="inner")

    benchmark.to_csv(osp.join(config['SAVE-TO'],config['BENCHMARK-CSV-NAME']),
                     index=False)


if __name__ == '__main__':

    with open('config.yaml','r') as f:
        config = yaml.safe_load(f)

    config = config['CREATE-BENCHMARK']



    print('Creating partial view indices')
    if osp.exists(config['SAVE-TO']):
        examples = os.listdir(config['SAVE-TO'])
    else:
        examples = []
    if len(examples) != 0:
        print('The partial view indices have already been processed.')
        user_input = input('overwrite (O) or skip (S) this process? [O/S]?: ')
        
        if user_input.lower() == 'o':
            create_faust_partial_indices(config)

    else:
        create_faust_partial_indices(config)
        

    print('Finding partial views overlap.')
    overlap_df_path = osp.join(config['SAVE-TO'],'overlaps.csv')
    if osp.exists(overlap_df_path):
        print('Overlap csv already created.')
        user_input = input('overwrite (O) or skip (S) this process? [O/S]?: ')
        if user_input.lower() == 'o':
            calc_overlap(config)
    else:
        calc_overlap(config)

    print('Filter overlaps to desired requirements.')
    overlap_filtered_df_path = osp.join(config['SAVE-TO'],
                    'overlaps-filtered.csv')
    if osp.exists(overlap_filtered_df_path):
        print('Overlap-filtered csv already created.')
        user_input = input('overwrite (O) or skip (S) this process? [O/S]?: ')
        if user_input.lower() == 'o':
            _ = filter_overlap(config)
    else:
        _ = filter_overlap(config)

    print('Create benchmark dataset.')
    benchmark_path = osp.join(config['SAVE-TO'],config['BENCHMARK-CSV-NAME'])
    if osp.exists(benchmark_path):
        print('Benchmark csv already created.')
        user_input = input('overwrite (O) or skip (S) this process? [O/S]?: ')
        if user_input.lower() == 'o':
            create_benchmark_from_overlaps(config)
    else:
        create_benchmark_from_overlaps(config)
