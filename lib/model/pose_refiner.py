import torch.nn as nn
from .network_utils import initseq, RodriguesModule

class PoseRefiner(nn.Module):
    def __init__(self, opt):
        super(PoseRefiner, self).__init__()
        self.pose_dim = opt.d_in
        self.total_bones = opt.n_bone
        embedding_size = opt.d_in * opt.n_bone
        mlp_width = opt.d_hid
        mlp_depth = opt.n_dims
        block_mlps = [nn.Linear(embedding_size, mlp_width), nn.ReLU()]
        for _ in range(0, mlp_depth - 1):
            block_mlps += [nn.Linear(mlp_width, mlp_width), nn.ReLU()]

        block_mlps += [nn.Linear(mlp_width, embedding_size)]

        self.block_mlps = nn.Sequential(*block_mlps)
        initseq(self.block_mlps)

        # init the weights of the last layer as very small value
        # -- at the beginning, we hope the rotation matrix can be identity
        init_val = 1e-5
        last_layer = self.block_mlps[-1]
        last_layer.weight.data.uniform_(-init_val, init_val)
        last_layer.bias.data.zero_()

        self.rodriguez = RodriguesModule()


    def forward(self, pose_input):
        rvec = self.block_mlps(pose_input).view(-1, self.pose_dim)
        Rs = self.rodriguez(rvec).view(-1, self.total_bones, 3, 3)
        return Rs
