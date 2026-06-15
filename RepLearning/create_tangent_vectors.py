# author: Pedro
import os
import numpy as np

def compute_local_axes(p_pos):
    
        aa_vec = p_pos[1:] - p_pos[0:-1]
        ops_aa_vec = -1.0*aa_vec
        axes_1 = np.cross(ops_aa_vec[0:-1], aa_vec[1:])

        norm_axis_1 = np.linalg.norm(axes_1, 2, -1, True)
        mask_zero = norm_axis_1.reshape((-1)) < 1e-5
        if np.any(mask_zero):
            indices = np.where(mask_zero)[0]
            max_index = axes_1.shape[0]
            for cur_index in indices:
                aux_index = (cur_index + 1)%max_index
                while np.sum(axes_1[cur_index]) < 1e-5:
                    axes_1[cur_index] = axes_1[aux_index]
                    aux_index = (aux_index + 1)%max_index
            norm_axis_1 = np.linalg.norm(axes_1, 2, -1, True)

        axes_1 = axes_1/norm_axis_1
        axes_2 = np.cross(aa_vec[1:], axes_1)

        norm_axis_2 = np.linalg.norm(axes_2, 2, -1, True)
        mask_zero = norm_axis_2.reshape((-1)) < 1e-5
        if np.any(mask_zero):
            indices = np.where(mask_zero)[0]
            max_index = axes_2.shape[0]
            for cur_index in indices:
                aux_index = (cur_index + 1)%max_index
                while np.sum(axes_2[cur_index]) < 1e-5:
                    axes_2[cur_index] = axes_2[aux_index]
                    aux_index = (aux_index + 1)%max_index
            norm_axis_2 = np.linalg.norm(axes_2, 2, -1, True)

        axes_2 = axes_2/norm_axis_2
        axes_3 = np.cross(axes_1, axes_2)

        norm_axis_3 = np.linalg.norm(axes_3, 2, -1, True)
        mask_zero = norm_axis_3.reshape((-1)) < 1e-5
        if np.any(mask_zero):
            indices = np.where(mask_zero)[0]
            max_index = axes_3.shape[0]
            for cur_index in indices:
                aux_index = (cur_index + 1)%max_index
                while np.sum(axes_3[cur_index]) < 1e-5:
                    axes_3[cur_index] = axes_3[aux_index]
                    aux_index = (aux_index + 1)%max_index
            norm_axis_3 = np.linalg.norm(axes_3, 2, -1, True)

        axes_3 = axes_3/norm_axis_3

        axes_list = np.concatenate([
                np.reshape(axes_1, (-1, 1, 3)),
                np.reshape(axes_2, (-1, 1, 3)),
                np.reshape(axes_3, (-1, 1, 3))
            ], axis=1)

        return np.pad(axes_list,
            ((1,1), (0,0), (0,0)), 'edge')


#pdb_path = "./data_chains/pdb_chains/"
pdb_path = "./data_chains/af_chains/"
list_pdbs = [f for f in os.listdir(pdb_path) if os.path.isfile(os.path.join(pdb_path, f)) and not f.endswith("_axes.npy")]

for cur_iter, cur_pdb in enumerate(list_pdbs):
    cur_path = os.path.join(pdb_path, cur_pdb)
    cur_pos = np.load(cur_path)
    cur_axes = compute_local_axes(cur_pos)
    cur_axes_path = os.path.join(pdb_path, cur_pdb[:-4]+"_axes.npy")
    np.save(cur_axes_path, cur_axes)
    if cur_iter%1000==0:
        print("\r Creating {:6d}/{:6d}".format(cur_iter, len(list_pdbs)), end ="")
print()