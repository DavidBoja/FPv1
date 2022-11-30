
import numpy as np

def pts2homo(pts):
    '''
    input pts: np array dim N x 3
    return pts: np array dim N x 4
    '''
    return np.concatenate((pts, np.ones(pts.shape[0]).reshape(-1,1)), axis=1)

def homo_matmul(pts,T): 
    '''
    inputs Nx3 pts and 4x4 transformation matrix
    '''
    pts_T = np.matmul(pts2homo(pts),T.T)
    return (pts_T / pts_T[:,3].reshape(-1,1))[:,:3]