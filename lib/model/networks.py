import torch.nn as nn
import torch
import numpy as np
from .embedders import get_embedder

class ImplicitNet(nn.Module):
    def __init__(self, opt):
        super().__init__()

        dims_hidden = []
        for i in range(opt.n_layers):
            dims_hidden.append(opt.d_hid)

        dims = [opt.d_in] + list(dims_hidden) + [opt.d_out + opt.feature_vector_size]
        self.num_layers = len(dims)
        self.skip_in = opt.skip_in
        self.embed_fn = None
        self.opt = opt

        if opt.multires > 0:
            embed_fn, input_ch = get_embedder(opt.multires, input_dims=opt.d_in, mode=opt.embedder_mode)
            self.embed_fn = embed_fn
            dims[0] = input_ch
        self.cond = opt.cond   
        if self.cond == 'smpl':
            self.cond_layer = [0]
            self.cond_dim = 23*3
        elif self.cond == 'frame':
            self.cond_layer = [0]
            self.cond_dim = opt.dim_frame_encoding
        self.dim_pose_embed = 0
        if self.dim_pose_embed > 0:
            self.lin_p0 = nn.Linear(self.cond_dim, self.dim_pose_embed)
            self.cond_dim = self.dim_pose_embed
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                dim_out = dims[l + 1] - dims[0]
            else:
                dim_out = dims[l + 1]
            
            if self.cond != 'none' and l in self.cond_layer:
                lin = nn.Linear(dims[l] + self.cond_dim, dim_out)
            else:
                lin = nn.Linear(dims[l], dim_out)
            if opt.init == 'geometry':
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight,
                                          mean=np.sqrt(np.pi) /
                                          np.sqrt(dims[l]),
                                          std=0.0001)
                    torch.nn.init.constant_(lin.bias, -opt.bias)
                elif opt.multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0,
                                          np.sqrt(2) / np.sqrt(dim_out))
                elif opt.multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                          np.sqrt(2) / np.sqrt(dim_out))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):],
                                            0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                          np.sqrt(2) / np.sqrt(dim_out))
            if opt.init == 'zero':
                init_val = 1e-5
                if l == self.num_layers - 2:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.uniform_(lin.weight, -init_val, init_val)
            if opt.weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        self.softplus = nn.Softplus(beta=100)

    def forward(self, input, cond, current_epoch=None):
        if input.ndim == 2: input = input.unsqueeze(0)

        num_batch, num_point, num_dim = input.shape

        if num_batch * num_point == 0: return input

        input = input.reshape(num_batch * num_point, num_dim)

        if self.cond != 'none':
            num_batch, num_cond = cond[self.cond].shape

            input_cond = cond[self.cond].unsqueeze(1).expand(num_batch, num_point, num_cond)

            input_cond = input_cond.reshape(num_batch * num_point, num_cond)

            if self.dim_pose_embed:
                input_cond = self.lin_p0(input_cond)

        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if self.cond != 'none' and l in self.cond_layer:
                x = torch.cat([x, input_cond], dim=-1)
            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.softplus(x)
        x = torch.tanh(x)
        x = x.reshape(num_batch, num_point, -1)

        return x

    def gradient(self, x, cond):
        x.requires_grad_(True)
        y = self.forward(x, cond)[:, :1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(outputs=y,
                                        inputs=x,
                                        grad_outputs=d_output,
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]
        return gradients.unsqueeze(1)


class RenderingNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        dims_hidden = []
        for i in range(opt.n_layers):
            dims_hidden.append(opt.d_hid)

        self.mode = opt.mode
        dims = [opt.d_in + opt.feature_vector_size] + list(dims_hidden) + [opt.d_out]

        self.embedview_fn = None
        if opt.multires_view > 0:
            embedview_fn, input_ch = get_embedder(opt.multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        if self.mode == 'nerf_frame_encoding':
            dims[0] += opt.dim_frame_encoding

        if self.mode == 'pose':
            self.dim_cond_embed = 8 
            self.cond_dim = 23*3 # dimension of the body pose, global orientation excluded.
            # lower the condition dimension
            self.lin_pose = torch.nn.Linear(self.cond_dim, self.dim_cond_embed)
            dims[0] += 8

        self.num_layers = len(dims)
        for l in range(0, self.num_layers - 1):
            dim_out = dims[l + 1]
            lin = nn.Linear(dims[l], dim_out)
            nn.init.kaiming_normal_(lin.weight, mode='fan_out', nonlinearity='relu')

            if opt.weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
        
    def forward(self, points, normals, view_dirs, cond, feature_vectors, frame_latent_code=None):
        if self.embedview_fn is not None:
            if self.mode == 'nerf_frame_encoding':
                view_dirs = self.embedview_fn(view_dirs)

        if self.mode == 'nerf_frame_encoding':
            frame_latent_code = frame_latent_code.expand(view_dirs.shape[0], -1)
            rendering_input = torch.cat([view_dirs, frame_latent_code, feature_vectors], dim=-1)
        elif self.mode == 'pose':
            body_pose = self.lin_pose(cond['smpl'])
            num_points = points.shape[0]
            body_pose = body_pose.unsqueeze(1).expand(-1, num_points, -1).reshape(num_points, -1)
            rendering_input = torch.cat([points, normals, body_pose, feature_vectors], dim=-1)
        else:
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
            # raise NotImplementedError

        x = rendering_input
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.relu(x)
                feat = x
        return x, feat


class HF_ImplicitNet(nn.Module):
    def __init__(self, opt):
        super().__init__()

        dims_hidden = []
        for i in range(opt.n_layers):
            dims_hidden.append(opt.d_hid)

        dims = [opt.d_in] + list(dims_hidden) + [opt.d_out + opt.feature_vector_size]
        self.num_layers = len(dims)
        self.skip_in = opt.skip_in
        self.embed_fn = None
        self.opt = opt
        self.d_in = opt.d_in

        if opt.multires > 0:
            embed_fn, input_ch = get_embedder(opt.multires,
                                              input_dims=opt.d_in,
                                              mode=opt.embedder_mode,
                                              min_freq=opt.min_freq,
                                              include_input=opt.include_input)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.dim_feat_lf = opt.feature_vector_size

        self.cond = opt.cond
        if self.cond == 'smpl':
            self.cond_layer = [0]
            self.cond_dim = 23*3

        elif self.cond == 'frame':
            self.cond_layer = [0]
            self.cond_dim = opt.dim_frame_encoding

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                dim_out = dims[l + 1] - dims[0]
            else:
                dim_out = dims[l + 1]

            dim_in = dims[l]

            if l == 0:
                dim_in += self.dim_feat_lf

            if self.cond != 'none' and l in self.cond_layer:
                lin = nn.Linear(dim_in + self.cond_dim, dim_out)
            else:
                lin = nn.Linear(dim_in, dim_out)
            if opt.init == 'geometry':
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight,
                                          mean=np.sqrt(np.pi) /
                                               np.sqrt(dims[l]),
                                          std=0.0001)
                    torch.nn.init.constant_(lin.bias, -opt.bias)
                elif opt.multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, opt.d_in:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :opt.d_in], 0.0,
                                          np.sqrt(2) / np.sqrt(dim_out))
                elif opt.multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                          np.sqrt(2) / np.sqrt(dim_out))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - opt.d_in):],
                                            0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                          np.sqrt(2) / np.sqrt(dim_out))
            if opt.init == 'zero':
                init_val = 1e-5
                if l == self.num_layers - 2:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.uniform_(lin.weight, -init_val, init_val)
            if opt.weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        self.softplus = nn.Softplus(beta=100)

    def forward(self, input, lf_feat, cond):
        if input.ndim == 2: input = input.unsqueeze(0)

        num_batch, num_point, num_dim = input.shape

        if num_batch * num_point == 0: return input

        input = input.reshape(num_batch * num_point, num_dim)
        lf_feat = lf_feat.reshape(num_batch * num_point, -1)

        if self.cond != 'none':
            input_cond = cond[self.cond]
            num_batch, num_cond = input_cond.shape
            input_cond = input_cond.unsqueeze(1).expand(num_batch, num_point, num_cond)
            input_cond = input_cond.reshape(num_batch * num_point, num_cond)

        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = torch.cat([input, lf_feat], dim=-1)

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if self.cond != 'none' and l in self.cond_layer:
                x = torch.cat([x, input_cond], dim=-1)
            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.softplus(x)
        x = torch.tanh(x)
        x = x.reshape(num_batch, num_point, -1)
        return x

    def gradient(self, x, cond):
        x.requires_grad_(True)
        y = self.forward(x, cond)[:, :1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(outputs=y,
                                        inputs=x,
                                        grad_outputs=d_output,
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]
        return gradients.unsqueeze(1)


class HF_RenderingNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        dims_hidden = []
        for i in range(opt.n_layers):
            dims_hidden.append(opt.d_hid)

        self.mode = opt.mode

        dims = [opt.d_in + opt.feature_vector_size] + list(dims_hidden) + [opt.d_out]

        self.lin_lf2hf = torch.nn.Linear(opt.feature_vector_size, 32)
        dims[0] += 32
        self.embedview_fn = None
        if opt.multires_view > 0:
            embedview_fn, input_ch = get_embedder(opt.multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        if self.mode == 'nerf_frame_encoding':
            dims[0] += opt.dim_frame_encoding

        if self.mode == 'pose':
            self.dim_cond_embed = 8
            self.cond_dim = 23*3  # dimension of the body pose, global orientation excluded.
            # lower the condition dimension
            self.lin_pose = torch.nn.Linear(self.cond_dim, self.dim_cond_embed)
            dims[0] += 8

        self.num_layers = len(dims)
        for l in range(0, self.num_layers - 1):
            dim_out = dims[l + 1]
            lin = nn.Linear(dims[l], dim_out)
            nn.init.kaiming_normal_(lin.weight, mode='fan_out', nonlinearity='relu')

            if opt.weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        self.relu = nn.ReLU()

    def forward(self, lf_feats, points, normals, view_dirs, cond, feature_vectors, frame_latent_code=None):
        lf_feats = self.lin_lf2hf(lf_feats)
        if self.embedview_fn is not None:
            normals = self.embedview_fn(normals)
        if self.mode == 'pose':
            body_pose = self.lin_pose(cond['smpl'])
            num_points = points.shape[0]
            body_pose = body_pose.unsqueeze(1).expand(-1, num_points, -1).reshape(num_points, -1)
            rendering_input = torch.cat([points, normals, body_pose, lf_feats, feature_vectors], dim=-1)
        else:
            rendering_input = torch.cat([points, normals, lf_feats, feature_vectors], dim=-1)

        x = rendering_input
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.relu(x)
        return x

