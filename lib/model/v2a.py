from .networks import ImplicitNet, RenderingNet, HF_ImplicitNet, HF_RenderingNet
from .density import LaplaceDensity
from .ray_sampler import ErrorBoundSampler
from .deformer import SMPLDeformer
from .smpl import SMPLServer
from .sampler import PointInSpace
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
import h5py


class V2A(nn.Module):
    def __init__(self, opt, betas_path, gender):
        super().__init__()
        self.lf_implicit_network = ImplicitNet(opt.lf_implicit_network)
        self.lf_rendering_network = RenderingNet(opt.lf_rendering_network)
        self.density = LaplaceDensity(**opt.density)
        self.sigmoid = nn.Sigmoid()
        self.hf_implicit_network = HF_ImplicitNet(opt.hf_implicit_network)
        self.hf_rendering_network = HF_RenderingNet(opt.hf_rendering_network)

        dataset_h5py = h5py.File(betas_path, 'r')
        betas = dataset_h5py['mean_shape'][:][None]
        dataset_h5py.close()

        self.use_smpl_deformer = opt.use_smpl_deformer
        self.gender = gender
        if self.use_smpl_deformer:
            self.deformer = SMPLDeformer(betas=betas, gender=self.gender)

        self.sdf_bounding_sphere = 3.0

        self.ray_sampler = ErrorBoundSampler(self.sdf_bounding_sphere, inverse_sphere_bg=False, **opt.ray_sampler)
        self.smpl_server = SMPLServer(gender=self.gender, betas=betas)
        self.sampler = PointInSpace()
        self.n_sdf_samp = opt.n_sdf_samp

        self.step_start_cond_pose = opt.step_start_cond_pose
        self.epoch_inter_cond_pose = opt.epoch_inter_cond_pose


    def sdf_func_with_smpl_deformer(self, x, cond, smpl_tfs, smpl_verts):
        if hasattr(self, "deformer"):
            x_c, outlier_mask = self.deformer.forward(x, smpl_tfs, return_weights=False, inverse=True, smpl_verts=smpl_verts)
            lf_output = self.lf_implicit_network(x_c, cond)[0]
            lf_sdf = lf_output[:, :1]
            lf_feature = lf_output[:, 1:]
            hf_sdf = self.hf_implicit_network(x_c, lf_feature, cond)[0, :, :1]
            w_sdf = 1.0 / (1 + (lf_sdf.detach() / 0.05) ** 4)
            sdf = w_sdf * hf_sdf + lf_sdf
            if not self.training:
                sdf[outlier_mask] = 4.
        return sdf, x_c

    def rendering(self, batch_size, cond, dirs,
                     pts_c,
                     smpl_tfs,
                     N_samples, z_vals, z_max, input):

        grad_theta = None
        lf_grad_theta = None
        if self.training:
            # sample canonical SMPL surface pnts for the eikonal loss
            smpl_verts_c = self.smpl_server.verts_c.repeat(batch_size, 1,1)
            num_pixels = self.n_sdf_samp
            indices = torch.randperm(smpl_verts_c.shape[1], device=pts_c.device)[:num_pixels]
            verts_c = torch.index_select(smpl_verts_c, 1, indices)
            sample = self.sampler.get_points(verts_c, global_ratio=0.)

            sample.requires_grad_()
            lf_im_out = self.lf_implicit_network(sample, cond)
            lf_im_feat = lf_im_out[..., 1:]
            lf_sdf = lf_im_out[..., :1]

            hf_sdf = self.hf_implicit_network(sample, lf_im_feat, cond)[..., 0:1]

            w_sdf = 1.0 / (1 + (lf_sdf.detach() / 0.05) ** 4)
            sdf = w_sdf * hf_sdf + lf_sdf

            grad_theta = gradient(sample, sdf)
            lf_grad_theta = gradient(sample, lf_sdf)

        view = -dirs.reshape(-1, 3)

        results = self.get_rbg_value(pts_c,
                                    view,
                                    cond, smpl_tfs,
                                    is_training=self.training)
        fg_rgb = results['rgb']
        normal_values = results['normals']
        sdf = results['sdf']
        density_values = self.density(sdf)

        fg_rgb = fg_rgb.reshape(-1, N_samples, 3)
        normal_values = normal_values.reshape(-1, N_samples, 3)

        weights, bg_transmittance = self.volume_rendering(z_vals, z_max, density_values)
        fg_rgb_values = torch.sum(weights.unsqueeze(-1) * fg_rgb, 1)
        bg_rgb_values = torch.ones_like(fg_rgb_values, device=fg_rgb_values.device) * input['bgcolor'][0]
        # Composite foreground and background
        bg_rgb_values = bg_transmittance.unsqueeze(-1) * bg_rgb_values
        rgb_values = fg_rgb_values + bg_rgb_values

        bg_rgb_values_normal = torch.ones_like(fg_rgb_values, device=fg_rgb_values.device)
        bg_rgb_values_normal = bg_transmittance.unsqueeze(-1) * bg_rgb_values_normal

        normal_values = (torch.sum(weights.unsqueeze(-1) * normal_values, 1) + 1) / 2 + bg_rgb_values_normal

        lf_sdf = results['lf_sdf']
        lf_fg_rgb = results['lf_rgb']
        lf_normal_values = results['lf_normals']
        lf_fg_rgb = lf_fg_rgb.reshape(-1, N_samples, 3)
        lf_normal_values = lf_normal_values.reshape(-1, N_samples, 3)
        lf_density_values = self.density(lf_sdf)
        lf_weights, lf_bg_transmittance = self.volume_rendering(z_vals, z_max, lf_density_values)
        lf_fg_rgb_values = torch.sum(lf_weights.unsqueeze(-1) * lf_fg_rgb, 1)
        bg_rgb_values = torch.ones_like(fg_rgb_values, device=fg_rgb_values.device) * input['bgcolor'][0]
        lf_bg_rgb_values = lf_bg_transmittance.unsqueeze(-1) * bg_rgb_values
        lf_rgb_values = lf_fg_rgb_values + lf_bg_rgb_values

        lf_bg_rgb_values_normal = torch.ones_like(fg_rgb_values, device=fg_rgb_values.device)
        lf_bg_rgb_values_normal = bg_transmittance.unsqueeze(-1) * lf_bg_rgb_values_normal
        lf_normal_values = (torch.sum(lf_weights.unsqueeze(-1) * lf_normal_values, 1) + 1) / 2 + lf_bg_rgb_values_normal

        if self.training:
            output = {
                'rgb_values': rgb_values,
                'lf_rgb_values': lf_rgb_values,
                'normal_values': normal_values,
                'lf_normal_values': lf_normal_values,
                'lf_acc_map': torch.sum(lf_weights, -1),
                'acc_map': torch.sum(weights, -1),
                'sdf_output': sdf,
                'grad_theta': grad_theta,
                'lf_grad_theta': lf_grad_theta,
                'epoch': input['current_epoch'],
                'bgcolor': input['bgcolor'],
            }
        else:
            fg_output_rgb = fg_rgb_values + bg_transmittance.unsqueeze(-1) * torch.ones_like(fg_rgb_values,
                                                                                             device=fg_rgb_values.device)
            output = {
                'acc_map': torch.sum(weights, -1),
                'rgb_values': rgb_values,
                'lf_rgb_values': lf_rgb_values,
                'fg_rgb_values': fg_output_rgb,
                'normal_values': normal_values,
                'lf_normal_values': lf_normal_values,
                'sdf_output': sdf,
            }
        return output


    def forward(self, input):

        # Parse model input
        torch.set_grad_enabled(True)
        cam_loc = input["cam_loc"].reshape(-1, 3)
        ray_dirs = input["ray_dirs"].reshape(-1, 3)
        near = input["near"].reshape(-1, 1)
        far = input["far"].reshape(-1, 1)
        batch_size, num_pixels, _ = input["ray_dirs"].shape

        scale = input['smpl_params'][:, 0]
        smpl_pose = input["smpl_pose"]
        smpl_shape = input["smpl_shape"]
        smpl_trans = input["smpl_trans"]
        smpl_output = self.smpl_server(scale, smpl_trans, smpl_pose, smpl_shape)

        smpl_tfs = smpl_output['smpl_tfs']
        cond = {'smpl': smpl_pose[:, 3:]/np.pi}
        if self.training:
            if input['global_step'] < self.step_start_cond_pose or input['current_epoch'] % self.epoch_inter_cond_pose == 0:
                cond = {'smpl': smpl_pose[:, 3:] * 0.}

        cond['global_step'] = input['global_step']
        cond['current_epoch'] = input['current_epoch']

        z_vals, _ = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, near, far, self, cond, smpl_tfs, eval_mode=True,
                                                smpl_verts=smpl_output['smpl_verts'])

        z_max = z_vals[:,-1]
        z_vals = z_vals[:,:-1]
        N_samples = z_vals.shape[1]

        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points.reshape(-1, 3)

        dirs = ray_dirs.unsqueeze(1).repeat(1,N_samples,1)
        _, pts_c = self.sdf_func_with_smpl_deformer(points_flat, cond, smpl_tfs, smpl_output['smpl_verts'])

        output = self.rendering(batch_size, cond, dirs,
                                       pts_c,
                                       smpl_tfs,
                                       N_samples, z_vals, z_max, input)

        return output

    def get_rbg_value(self, pnts_c, view_dirs, cond, tfs, is_training=True):
        outputs = {}

        results = self.forward_gradient(pnts_c, cond, tfs, create_graph=is_training, retain_graph=is_training)

        lf_rgb, lf_rendering_feature = self.lf_rendering_network(pnts_c, results['lf_gradients'], view_dirs, cond, results['lf_feature'])
        hf_rgb = self.hf_rendering_network(lf_rendering_feature, pnts_c, results['gradients'], view_dirs, cond, results['hf_feature'])
        lf_rgb = self.sigmoid(lf_rgb)
        rgb_vals = results['w_sdf'] * hf_rgb + lf_rgb

        outputs['rgb'] = rgb_vals
        outputs['lf_rgb'] = lf_rgb
        outputs['normals'] = results['gradients']
        outputs['lf_normals'] = results['lf_gradients']
        outputs['sdf'] = results['sdf']
        outputs['lf_sdf'] = results['lf_sdf']
        return outputs

    def forward_gradient(self, pnts_c, cond, tfs, create_graph=True, retain_graph=True):
        if pnts_c.shape[0] == 0:
            return pnts_c.detach()
        pnts_c.requires_grad_(True)

        pnts_d = self.deformer.forward_skinning(pnts_c.unsqueeze(0), None, tfs).squeeze(0)
        num_dim = pnts_d.shape[-1]
        grads = []
        for i in range(num_dim):
            d_out = torch.zeros_like(pnts_d, requires_grad=False, device=pnts_d.device)
            d_out[:, i] = 1
            grad = torch.autograd.grad(
                outputs=pnts_d,
                inputs=pnts_c,
                grad_outputs=d_out,
                create_graph=create_graph,
                retain_graph=True if i < num_dim - 1 else retain_graph,
                only_inputs=True)[0]
            grads.append(grad)
        grads = torch.stack(grads, dim=-2)
        grads_inv = grads.inverse()

        lf_output = self.lf_implicit_network(pnts_c, cond)[0]
        lf_sdf = lf_output[:, :1]
        lf_feature = lf_output[:, 1:]
        d_output = torch.ones_like(lf_sdf, requires_grad=False, device=lf_sdf.device)
        lf_gradients = torch.autograd.grad(
            outputs=lf_sdf,
            inputs=pnts_c,
            grad_outputs=d_output,
            create_graph=create_graph,
            retain_graph=True,
            only_inputs=True)[0]
        lf_gradients = lf_gradients.reshape(lf_gradients.shape[0], -1)
        lf_gradients_xo = torch.nn.functional.normalize(torch.einsum('bi,bij->bj', lf_gradients, grads_inv), dim=1)

        hf_output = self.hf_implicit_network(pnts_c, lf_feature, cond)[0]
        hf_sdf = hf_output[:, :1]
        hf_feature = hf_output[:, 1:]

        w_sdf = 1.0 / (1 + (lf_sdf.detach() / 0.05) ** 4)
        sdf = w_sdf * hf_sdf + lf_sdf

        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=pnts_c,
            grad_outputs=d_output,
            create_graph=create_graph,
            retain_graph=retain_graph,
            only_inputs=True)[0]
        gradients = gradients.reshape(gradients.shape[0], -1)
        gradients_xo = torch.nn.functional.normalize(torch.einsum('bi,bij->bj', gradients, grads_inv), dim=1)

        results = {'gradients':gradients_xo,
                   'lf_gradients':lf_gradients_xo,
                   'lf_feature':lf_feature,
                   'hf_feature':hf_feature,
                   'lf_sdf':lf_sdf,
                   'sdf':sdf,
                   'w_sdf':w_sdf,
                   }

        return results

    def volume_rendering(self, z_vals, z_max, density):
        density = density.reshape(-1, z_vals.shape[1]) # (batch_size * num_pixels) x N_samples
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat([dists, z_max.unsqueeze(-1) - z_vals[:, -1:]], -1)

        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1, device=density.device), free_energy], dim=-1)  # add 0 for transperancy 1 at t_0
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        fg_transmittance = transmittance[:, :-1]
        weights = alpha * fg_transmittance  # probability of the ray hits something here
        bg_transmittance = transmittance[:, -1]  # factor to be multiplied with the bg volume rendering

        return weights, bg_transmittance

def gradient(inputs, outputs):

    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0][:, :, -3:]
    return points_grad