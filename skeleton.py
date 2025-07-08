import torch
import numpy as np
import pickle

class BaseSkeleton():
    """
    Class to model skeleton
    """
    joint_dict = None
    joint_connection_dict = None
    def __init__(self) -> None:
        self.num_joints = None
        self.inward = None
        self.outward = None
        self.neighbor = None
        self.self_link = None

    def init(self):
        raise NotImplementedError

    @classmethod
    def expmap2rotmat(cls, r):
        """
        Converts an exponential map angle to a rotation matrix
        Matlab port to python for evaluation purposes
        I believe this is also called Rodrigues' formula
        https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/expmap2rotmat.m
        Args
        r: 1x3 exponential map
        Returns
        R: 3x3 rotation matrix
        """
        theta = np.linalg.norm(r)
        r0 = np.divide(r, theta + np.finfo(np.float32).eps)
        r0x = np.array([0, -r0[2], r0[1], 0, 0, -r0[0], 0, 0, 0]).reshape(3, 3)
        r0x = r0x - r0x.T
        R = np.eye(3, 3) + np.sin(theta) * r0x + (1 - np.cos(theta)) * (r0x).dot(r0x);
        return R

    @classmethod
    def expmap2rotmat_torch(cls, r : torch.Tensor):
        """
        Converts expmap matrix to rotation
        batch pytorch version ported from the corresponding method above
        :param r: N*3
        :return: N*3*3
        """
        device = r.device
        theta = torch.norm(r, 2, 1)
        r0 = torch.div(r, theta.unsqueeze(1).repeat(1, 3) + 0.0000001)
        r1 = torch.zeros_like(r0).repeat(1, 3)
        r1[:, 1] = -r0[:, 2]
        r1[:, 2] = r0[:, 1]
        r1[:, 5] = -r0[:, 0]
        r1 = r1.view(-1, 3, 3)
        r1 = r1 - r1.transpose(1, 2)
        n = r1.data.shape[0]
        R = torch.eye(3, 3).repeat(n, 1, 1).float().to(device) + torch.mul(
            torch.sin(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3), r1) + torch.mul(
            (1 - torch.cos(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3)), torch.matmul(r1, r1))
        return R    

    @classmethod
    def rotmat2euler_torch(cls, R : torch.Tensor):
        """
        Converts a rotation matrix to euler angles
        batch pytorch version ported from the corresponding numpy method above
        :param R:N*3*3
        :return: N*3
        """
        n = R.data.shape[0]
        eul = torch.zeros(n, 3).float().to(R)
        idx_spec1 = (R[:, 0, 2] == 1).nonzero().cpu().data.numpy().reshape(-1).tolist()
        idx_spec2 = (R[:, 0, 2] == -1).nonzero().cpu().data.numpy().reshape(-1).tolist()
        if len(idx_spec1) > 0:
            R_spec1 = R[idx_spec1, :, :]
            eul_spec1 = torch.zeros(len(idx_spec1), 3).float().cuda()
            eul_spec1[:, 2] = 0
            eul_spec1[:, 1] = -np.pi / 2
            delta = torch.atan2(R_spec1[:, 0, 1], R_spec1[:, 0, 2])
            eul_spec1[:, 0] = delta
            eul[idx_spec1, :] = eul_spec1

        if len(idx_spec2) > 0:
            R_spec2 = R[idx_spec2, :, :]
            eul_spec2 = torch.zeros(len(idx_spec2), 3).float().cuda()
            eul_spec2[:, 2] = 0
            eul_spec2[:, 1] = np.pi / 2
            delta = torch.atan2(R_spec2[:, 0, 1], R_spec2[:, 0, 2])
            eul_spec2[:, 0] = delta
            eul[idx_spec2] = eul_spec2

        idx_remain = np.arange(0, n)
        idx_remain = np.setdiff1d(np.setdiff1d(idx_remain, idx_spec1), idx_spec2).tolist()
        if len(idx_remain) > 0:
            R_remain = R[idx_remain, :, :]
            eul_remain = torch.zeros(len(idx_remain), 3).float().cuda()
            eul_remain[:, 1] = -torch.asin(R_remain[:, 0, 2])
            eul_remain[:, 0] = torch.atan2(R_remain[:, 1, 2] / torch.cos(eul_remain[:, 1]),
                                           R_remain[:, 2, 2] / torch.cos(eul_remain[:, 1]))
            eul_remain[:, 2] = torch.atan2(R_remain[:, 0, 1] / torch.cos(eul_remain[:, 1]),
                                           R_remain[:, 0, 0] / torch.cos(eul_remain[:, 1]))
            eul[idx_remain, :] = eul_remain

        return eul
    
class AMASSSkeleton(BaseSkeleton):
    joint_dict = None
    joint_connection_dict = {}
    body_model_file = './body_model/scaled_offsets.npz'

    pklFile = "C:/Users/Avinash/Tyler/HybrIK test/HybrIK/model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
    with open(pklFile, 'rb') as f:
        model_data = pickle.load(f, encoding='latin1')

    kintree_table = model_data['kintree_table']  

    parents_raw = kintree_table[0]            

    parents = parents_raw.astype(np.int64)      

    root_mask = parents_raw == np.iinfo(np.uint32).max 
    parents[root_mask] = -1

    parent_info = np.load(body_model_file)
    offsets = torch.from_numpy(parent_info['p3d0'])

    # has connection of all joints
    parent_connect_dict = { i : p for i, p in enumerate(parents) }

    joints_to_use = np.arange(0,24)
    # total of 52 joints but the first 22 are the ones that are needed
    joints_to_ignore = np.setdiff1d(np.arange(24), joints_to_use)
    num_joints = len(joints_to_use)

    connect = {
        0: 1, 0: 2, 
        1: 4, 5: 2, 
        7: 4, 8: 5, 
        7: 10, 8: 11, 
        12: 15,
        12: 16, 12: 17,
        16: 18, 19: 17, 20: 18, 21: 19,
        1: 16, 2: 17} 

    remap_jidx_j = {i:j for i, j in enumerate(joints_to_use)}
    remap_j_jidx = {j:i for i, j in enumerate(joints_to_use)}
    
    def __init__(self) -> None:
        super().__init__()
        self.num_joints = len(self.joints_to_use)
        self.inward = []
        for i in range(self.num_joints):
            old_jidx = self.remap_jidx_j[i]
            if old_jidx in self.connect:
                end = self.connect[old_jidx]
                if end in self.joints_to_use:
                    self.inward.append((i, self.remap_j_jidx[end]))

        self.outward = [(j,i) for (i, j) in self.inward]
        self.self_link = [(i,i) for i in range(self.num_joints)] 
        self.neighbor = self.inward + self.outward
    
    def ang2joint(self, poses):
        """
        Picked entirely from https://github.com/wei-mao-2019/HisRepItself
        :param poses:[batch_size, joint_num, 3]
        :return:
        """
        def rodrigues(r):
            """
            Rodrigues' rotation formula that turns axis-angle tensor into rotation
            matrix in a batch-ed manner.
            Parameter:
            ----------
            r: Axis-angle rotation tensor of shape [batch_size * angle_num, 1, 3].
            Return:
            -------
            Rotation matrix of shape [batch_size * angle_num, 3, 3].
            """
            eps = r.clone().normal_(std=1e-8)
            theta = torch.norm(r + eps, dim=(1, 2), keepdim=True)
            # theta = torch.norm(r, dim=(1, 2), keepdim=True)  # dim cannot be tuple
            theta_dim = theta.shape[0]
            r_hat = r / theta
            cos = torch.cos(theta)
            z_stick = torch.zeros(theta_dim, dtype=torch.float).to(r.device)
            m = torch.stack(
                (z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1], r_hat[:, 0, 2], z_stick,
                 -r_hat[:, 0, 0], -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick), dim=1)
            m = torch.reshape(m, (-1, 3, 3))
            i_cube = (torch.eye(3, dtype=torch.float).unsqueeze(dim=0) \
                      + torch.zeros((theta_dim, 3, 3), dtype=torch.float)).to(r.device)
            A = r_hat.permute(0, 2, 1)
            dot = torch.matmul(A, r_hat)
            R = cos * i_cube + (1 - cos) * dot + torch.sin(theta) * m
            return R
        
        def with_zeros(x):
            """
            Append a [0, 0, 0, 1] tensor to a [3, 4] tensor.
            Parameter:
            ---------
            x: Tensor to be appended.
            Return:
            ------
            Tensor after appending of shape [4,4]
            """
            ones = torch.tensor(
                [[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float
            ).expand(x.shape[0], -1, -1).to(x.device)
            ret = torch.cat((x, ones), dim=1)
            return ret
        
        offsets = self.offsets.repeat([poses.shape[0], 1, 1])
        offsets = offsets.to(poses)
        batch_num = offsets.shape[0]

        jnum = len(self.parent_connect_dict.keys())

        # TODO (Tyler) : since you already have the rotation matrices, you need to extract the code segment below and use only that
        R_cube_big = poses
        results = []
        results.append(
            with_zeros(torch.cat((R_cube_big[:, 0], torch.reshape(offsets[:, 0, :], (-1, 3, 1))), dim=2))
        )
        # for i in range(1, kintree_table.shape[1]):
        for i in range(1, jnum):
            results.append( 
                torch.matmul(
                    results[self.parent_connect_dict[i]],
                    with_zeros(
                        torch.cat(
                            (R_cube_big[:, i], torch.reshape(offsets[:, i, :] - offsets[:, self.parent_connect_dict[i], :], (-1, 3, 1))),
                            dim=2
                        )
                    )
                )
            )

        stacked = torch.stack(results, dim=1)
        J_transformed = stacked[:, :, :3, 3]
        return J_transformed
