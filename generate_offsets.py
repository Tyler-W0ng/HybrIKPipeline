import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

def generate_bone_length_file():
    '''
    bone_scales index mapping for SMPL 24-joint skeleton:
    Each entry corresponds to a bone from parent[i] -> i

    0  -> root (pelvis) — no parent
    1  -> pelvis -> left_hip
    2  -> pelvis -> right_hip
    3  -> pelvis -> spine1
    4  -> left_hip -> left_knee
    5  -> right_hip -> right_knee
    6  -> spine1 -> spine2
    7  -> left_knee -> left_ankle
    8  -> right_knee -> right_ankle
    9  -> spine2 -> spine3 (chest)
    10  -> left_ankle -> left_foot
    11  -> right_ankle -> right_foot
    12  -> spine3 -> neck
    13  -> neck -> head
    14  -> spine3 -> left_shoulder
    15  -> spine3 -> right_shoulder
    16  -> left_shoulder -> left_elbow
    17  -> right_shoulder -> right_elbow
    18  -> left_elbow -> left_wrist
    19  -> right_elbow -> right_wrist
    20  -> left_wrist -> left_hand (optional)
    21  -> right_wrist -> right_hand (optional)
    22  -> left_foot -> left_toe (optional)
    23  -> right_foot -> right_toe (optional)
    '''
    bone_lengths = torch.tensor([[
        0.00000000, 0.11599894, 0.11463822, 0.11174528, 0.39043379, 0.39638457,
        0.13928591, 0.41141477, 0.41061604, 0.06138568, 0.13807219, 0.13793969,
        0.22496200, 0.15318780, 0.15186137, 0.09165876, 0.09931105, 0.10259350,
        0.26978287, 0.26014253, 0.25837988, 0.26133198, 0.10922580, 1.53871775
    ]])

    # Convert to numpy (remove batch dim if desired)
    bone_lengths_np = bone_lengths.squeeze(0).numpy()

    os.makedirs('./bone_scale', exist_ok=True)

    # Save to .npz
    np.savez('./bone_scale/bones.npz', bone_lengths=bone_lengths_np)

def visualize_joints(joints, parents):
    joints = joints.cpu().numpy()  
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Skip the last joint
    num_joints_to_plot = joints.shape[0] - 1

    ax.scatter(joints[:num_joints_to_plot, 0],
               joints[:num_joints_to_plot, 1],
               joints[:num_joints_to_plot, 2],
               c='r')

    for i in range(num_joints_to_plot):
        x, y, z = joints[i]
        ax.text(x, y, z, str(i))

    for i in range(num_joints_to_plot):
        p = parents[i]
        if p == -1 or p >= num_joints_to_plot:
            continue
        ax.plot(
            [joints[i, 0], joints[p, 0]],
            [joints[i, 1], joints[p, 1]],
            [joints[i, 2], joints[p, 2]],
            c='b'
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Skeleton")
    plt.show()

def get_bone_scales(parents):
    bone_length_file = './bone_scale/bones.npz'

    bone_lengths = np.load(bone_length_file)['bone_lengths']
    smpl_lengths = compute_bone_lengths_smpl(parents)

    bone_scales = bone_lengths / (smpl_lengths + 1e-8)  # avoid divide-by-zero

    return bone_scales


def compute_bone_lengths_smpl(parents):
    body_model_file = './body_model/smpl_skeleton.npz'
    parent_info = np.load(body_model_file)
    offsets = torch.from_numpy(parent_info['p3d0'])
    offsets = offsets.squeeze(0)[:24]

    bone_lengths = np.zeros(len(parents))

    for i in range(len(parents)):
        parent = parents[i]
        if parent == -1:
            bone_lengths[i] = 0.0 
        else:
            vec = offsets[i] - offsets[parent]
            bone_lengths[i] = np.linalg.norm(vec)

    return bone_lengths

def compute_scaled_offsets_from_scales(bone_scales, save_path='./body_model/scaled_offsets.npz'):
    """
    Compute and save new joint offsets using bone scales applied to SMPL standard offsets.

    Parameters:
    - bone_scales: array-like of shape (24,) — bone length scale factors for each bone.
    - save_path: where to save the resulting .npz file containing scaled offsets.

    Output:
    - Saves scaled offsets under the key 'p3d0' to the specified file path.
    """
    # Load SMPL standard offsets
    smpl_offsets = np.load('./body_model/smpl_skeleton.npz')['p3d0'][0][:24]  # [24, 3]
    # Define SMPL 24-joint parent list
    parents = [
        -1, 0, 0, 0, 1, 2, 3, 4, 5, 6,
        7, 8, 9, 12, 9, 9, 14, 15, 16, 17, 18, 19, 10, 11
    ]

    new_offsets = smpl_offsets.copy()

    for i in range(len(parents)):
        parent = parents[i]
        if parent == -1:
            continue
        vec = smpl_offsets[i] - smpl_offsets[parent]
        scaled_vec = vec * bone_scales[i]
        new_offsets[i] = new_offsets[parent] + scaled_vec

    # Save to .npz file with consistent name
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, p3d0=new_offsets)
    print(f"[INFO] Scaled offsets saved to: {save_path}")


pklFile = "C:/Users/Avinash/Tyler/HybrIK test/HybrIK/model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
with open(pklFile, 'rb') as f:
    model_data = pickle.load(f, encoding='latin1')

kintree_table = model_data['kintree_table']  

parents_raw = kintree_table[0]            

parents = parents_raw.astype(np.int64)  

root_mask = parents_raw == np.iinfo(np.uint32).max 
parents[root_mask] = -1

rot_file = torch.load("C:/Users/Avinash/Tyler/HybrIK test/HybrIK/rotation_matrices_fencing.pt", weights_only=True)

frame_num = 300
start = frame_num * 24
end = 24 + frame_num * 24
rotmats = rot_file[start:end].unsqueeze(0)

generate_bone_length_file()
bone_scales = get_bone_scales(parents)
bone_scales = torch.tensor(bone_scales, dtype=torch.float32).unsqueeze(0)
compute_scaled_offsets_from_scales(bone_scales.squeeze(0).numpy())

### Vizualization example
# from skeleton import AMASSSkeleton
# skeleton = AMASSSkeleton()
# joint_positions = skeleton.ang2joint(rotmats)

#visualize_joints(joint_positions[0], parents)
