import pytorch_lightning as pl
import torch.optim as optim
from lib.utils.meshing import generate_mesh
from lib.model.v2a import V2A
from lib.model.body_model_params import BodyModelParams
from lib.model.deformer import SMPLDeformer
import cv2
import torch
from lib.model.loss import Loss
import hydra
import os
import numpy as np
import trimesh
from lib.model.deformer import skinning
from lib.utils import utils
import h5py


class V2AModel(pl.LightningModule):
    def __init__(self, opt) -> None:
        super().__init__()

        self.opt = opt
        self.path_h5py = opt.dataset.metainfo.data_dir
        self.gender = opt.dataset.metainfo.gender

        dataset_h5py = h5py.File(self.path_h5py, 'r')
        n_images = dataset_h5py['img_shape'][0]
        training_indices = list(range(0, n_images, 1))
        shape = dataset_h5py['mean_shape'][:]
        poses = dataset_h5py['poses'][training_indices]
        trans = dataset_h5py['normalize_trans'][training_indices]
        dataset_h5py.close()

        num_training_frames = n_images
        self.model = V2A(opt.model, self.path_h5py, self.gender)
        self.start_frame = 0
        self.end_frame = n_images
        self.training_modules = ["model"]
        self.training_indices = list(range(self.start_frame, self.end_frame))
        self.body_model_params = BodyModelParams(num_training_frames, model_type='smpl')
        self.load_body_model_params(shape, poses, trans)
        optim_params = self.body_model_params.param_names
        for param_name in optim_params:
            self.body_model_params.set_requires_grad(param_name, requires_grad=True)
        self.training_modules += ['body_model_params']
        self.loss = Loss(opt.model.loss)

    def load_body_model_params(self, shape, poses, trans):
        body_model_params = {param_name: [] for param_name in self.body_model_params.param_names}
        body_model_params['betas'] = torch.tensor(shape[None], dtype=torch.float32)
        body_model_params['global_orient'] = torch.tensor(poses[:, :3], dtype=torch.float32)
        body_model_params['body_pose'] = torch.tensor(poses[:, 3:], dtype=torch.float32)
        body_model_params['transl'] = torch.tensor(trans, dtype=torch.float32)


        for param_name in body_model_params.keys():
            self.body_model_params.init_parameters(param_name, body_model_params[param_name], requires_grad=False)

    def set_init_lr(self, module, lr, params):
        for m in module:
            params_m = list(filter(lambda kv: m in kv[0], self.model.named_parameters()))
            params_m = [d[1] for d in params_m]
            if len(params) == 0:
                params = [{'params': params_m, 'lr': lr}]
            else:
                params.append({'params': params_m, 'lr': lr})

        return params

    def configure_optimizers(self):
        params = []
        module_lf = ['lf_implicit_network', 'lf_rendering_network']
        module_hf = ['hf_implicit_network', 'hf_rendering_network']
        module_density = ['density']
        params = self.set_init_lr(module_lf, self.opt.model.learning_rate, params)
        params = self.set_init_lr(module_hf, self.opt.model.hf_learning_rate, params)
        params = self.set_init_lr(module_density, self.opt.model.density_learning_rate, params)
        params.append({'params': self.body_model_params.parameters(), 'lr':self.opt.model.learning_rate*0.1})
        self.optimizer = optim.Adam(params, lr=self.opt.model.learning_rate, eps=1e-8)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.opt.model.sched_milestones, gamma=self.opt.model.sched_factor)
        return [self.optimizer], [self.scheduler]


    def training_step(self, batch):
        inputs, targets = batch
        inputs['global_step'] = self.global_step
        inputs['cam_idxs'] = inputs["idx"]
        batch_idx = inputs["idx"]

        body_model_params = self.body_model_params(batch_idx)
        inputs['smpl_pose'] = torch.cat((body_model_params['global_orient'], body_model_params['body_pose']), dim=1)
        inputs['smpl_shape'] = body_model_params['betas']
        inputs['smpl_trans'] = body_model_params['transl']

        inputs['current_epoch'] = self.current_epoch

        model_outputs = self.model(inputs)

        loss_output = self.loss(model_outputs, targets, self.current_epoch, self.global_step)

        for k, v in loss_output.items():
            if self.opt.model.mode == 'sockeye':
                self.log(k, v.item())
            else:
                self.log(k, v.item(), prog_bar=True, on_step=True)

        if self.global_step > 0 and self.global_step % 100 == 0:
            for i in range(len(self.optimizer.param_groups)):
                self.log(f'lr/lr_{i}', self.optimizer.param_groups[i]['lr'])

            for name, cur_para in self.named_parameters():
                if cur_para.grad is not None and cur_para.requires_grad:
                    para_norm = torch.norm(cur_para.grad.detach(), 2)
                    self.log('Grad/%s_norm' % name, para_norm)

        return loss_output["loss"]

    def training_epoch_end(self, outputs) -> None:
        return super().training_epoch_end(outputs)

    def query_oc(self, x, cond):
        x = x.reshape(-1, 3)
        lf_im_out = self.model.lf_implicit_network(x, cond)
        lf_sdf = lf_im_out[:,:,0].reshape(-1,1)
        lf_im_feat = lf_im_out[..., 1:]
        hf_sdf = self.model.hf_implicit_network(x, lf_im_feat, cond)[:,:,0].reshape(-1,1)
        w_sdf = 1.0 / (1 + (lf_sdf.detach() / 0.05) ** 4)
        sdf = w_sdf * hf_sdf + lf_sdf
        return {'sdf': sdf, 'lf_sdf': lf_sdf}

    def get_deformed_mesh_fast_mode(self, verts, smpl_tfs):
        verts = torch.tensor(verts, device=self.device).float()
        weights = self.model.deformer.query_weights(verts)
        verts_deformed = skinning(verts.unsqueeze(0),  weights, smpl_tfs).data.cpu().numpy()[0]
        return verts_deformed

    def validation_step(self, batch, *args, **kwargs):
        output = {}
        inputs, targets = batch
        inputs['current_epoch'] = self.current_epoch
        inputs['global_step'] = self.global_step
        inputs['cam_idxs'] = inputs["idx"]
        self.model.eval()

        body_model_params = self.body_model_params(inputs['image_id'])
        inputs['smpl_pose'] = torch.cat((body_model_params['global_orient'], body_model_params['body_pose']), dim=1)
        inputs['smpl_shape'] = body_model_params['betas']
        inputs['smpl_trans'] = body_model_params['transl']

        cond = {'smpl': inputs["smpl_pose"][:, 3:]/np.pi}
        cond['global_step'] = self.global_step
        cond['current_epoch'] = self.current_epoch
        mesh_canonical = generate_mesh(lambda x: self.query_oc(x, cond), self.model.smpl_server.verts_c[0], point_batch=10000, res_up=3, device=self.device)
        if mesh_canonical is not None:
            mesh_canonical = trimesh.Trimesh(mesh_canonical.vertices, mesh_canonical.faces)
            output.update({'canonical_mesh': mesh_canonical})

            scale, _, _, _ = torch.split(inputs["smpl_params"], [1, 3, 72, 10], dim=1)
            smpl_outputs = self.model.smpl_server(scale, inputs['smpl_trans'], inputs['smpl_pose'],
                                                  inputs['smpl_shape'])
            smpl_tfs = smpl_outputs['smpl_tfs']
            self.model.deformer = SMPLDeformer(betas=body_model_params['betas'], gender=self.gender, K=7).to(
                self.device)
            verts_deformed = self.get_deformed_mesh_fast_mode(mesh_canonical.vertices, smpl_tfs)
            mesh_deformed = trimesh.Trimesh(vertices=verts_deformed, faces=mesh_canonical.faces, process=False)
            output.update({'deformed_mesh': mesh_deformed})

        lf_mesh_canonical = generate_mesh(lambda x: self.query_oc(x, cond), self.model.smpl_server.verts_c[0], type='lf_sdf',
                                       point_batch=10000, res_up=3, device=self.device)
        if lf_mesh_canonical is not None:
            lf_mesh_canonical = trimesh.Trimesh(lf_mesh_canonical.vertices, lf_mesh_canonical.faces)
            output.update({'lf_canonical_mesh': lf_mesh_canonical})

        split = utils.split_input(inputs, targets["total_pixels"][0], device=self.device, n_pixels=min(targets['pixel_per_batch'], targets["img_size"][0] * targets["img_size"][1]))

        res = []
        for s in split:

            out = self.model(s)

            for k, v in out.items():
                try:
                    out[k] = v.detach()
                except:
                    out[k] = v

            if 'lf_rgb_values' in out:
                res.append({
                    'lf_rgb_values': out['lf_rgb_values'].detach(),
                    'rgb_values': out['rgb_values'].detach(),
                    'normal_values': out['normal_values'].detach(),
                    'fg_rgb_values': out['fg_rgb_values'].detach(),
                    'acc_map': out['acc_map'].detach(),
                })
            else:
                res.append({
                    'rgb_values': out['rgb_values'].detach(),
                    'normal_values': out['normal_values'].detach(),
                    'fg_rgb_values': out['fg_rgb_values'].detach(),
                    'acc_map': out['acc_map'].detach(),
                })


        batch_size = targets['rgb'].shape[0]

        model_outputs = utils.merge_output(res, targets["total_pixels"][0], batch_size)

        if 'lf_rgb_values' in model_outputs:
            output.update({
                "lf_rgb_values": model_outputs["lf_rgb_values"].detach().clone(),
                "rgb_values": model_outputs["rgb_values"].detach().clone(),
                "normal_values": model_outputs["normal_values"].detach().clone(),
                "fg_rgb_values": model_outputs["fg_rgb_values"].detach().clone(),
                "acc_map": model_outputs["acc_map"].detach().clone(),
                **targets,
            })
        else:
            output.update({
                "rgb_values": model_outputs["rgb_values"].detach().clone(),
                "normal_values": model_outputs["normal_values"].detach().clone(),
                "fg_rgb_values": model_outputs["fg_rgb_values"].detach().clone(),
                "acc_map": model_outputs["acc_map"].detach().clone(),
                **targets,
            })


        return output

    def validation_step_end(self, batch_parts):
        return batch_parts

    def validation_epoch_end(self, outputs) -> None:
        img_size = outputs[0]["img_size"]
        rgb_bg = (torch.ones_like(outputs[0]["rgb"]) * outputs[0]["bgcolor"][0]).reshape(-1,3)
        rgb_gt = torch.cat([output["rgb"] for output in outputs], dim=1).squeeze(0)
        rgb_gt = rgb_gt.reshape(*img_size, -1).cpu().numpy()
        ray_mask = outputs[0]["ray_mask"][0].reshape(-1)
        rgb_pred_ray = torch.cat([output["rgb_values"] for output in outputs], dim=0)
        rgb_pred = rgb_bg.clone()
        rgb_pred[ray_mask] = rgb_pred_ray
        rgb_pred = rgb_pred.reshape(*img_size, -1)

        if 'lf_rgb_values' in outputs[0]:
            lf_rgb_pred_ray = torch.cat([output["lf_rgb_values"] for output in outputs], dim=0)
            lf_rgb_pred = rgb_bg.clone()
            lf_rgb_pred[ray_mask] = lf_rgb_pred_ray
            lf_rgb_pred = lf_rgb_pred.reshape(*img_size, -1)
            rgb_pred = torch.cat([rgb_pred, lf_rgb_pred], dim=1).cpu().numpy()

        normal_pred_ray = torch.cat([output["normal_values"] for output in outputs], dim=0)
        normal_pred = rgb_bg.clone()
        normal_pred[ray_mask] = (normal_pred_ray + 1) / 2
        normal_pred = normal_pred.reshape(*img_size, -1).cpu().numpy()

        rgb_gt = (255. * np.clip(rgb_gt, 0., 1.)).astype(np.uint8)
        rgb_pred = (255. * np.clip(rgb_pred, 0., 1.)).astype(np.uint8)
        normal_pred = (255. * np.clip(normal_pred, 0., 1.)).astype(np.uint8)

        os.makedirs("rendering", exist_ok=True)

        img_out = np.concatenate([rgb_gt, rgb_pred, normal_pred], axis=1)
        cv2.imwrite(f"rendering/{self.current_epoch}.png", img_out[:, :, ::-1])

    def test_step(self, batch, *args, **kwargs):
        inputs, targets, pixel_per_batch, total_pixels, idx = batch
        idx_pose = inputs['idx_pose']
        num_splits = (total_pixels + pixel_per_batch - 1) // pixel_per_batch
        results = []

        scale, smpl_trans, smpl_pose, smpl_shape = torch.split(inputs["smpl_params"], [1, 3, 72, 10], dim=1)
        if idx_pose > -1:
            body_model_params = self.body_model_params(inputs['idx_pose'])
            smpl_shape = body_model_params['betas'] if body_model_params['betas'].dim() == 2 else body_model_params[
                'betas'].unsqueeze(0)
            smpl_pose = torch.cat((smpl_pose[..., :3], body_model_params['body_pose']), dim=1)

        smpl_outputs = self.model.smpl_server(scale, smpl_trans, smpl_pose, smpl_shape)
        smpl_tfs = smpl_outputs['smpl_tfs']
        cond = {'smpl': smpl_pose[:, 3:]/np.pi}
        cond['global_step'] = 1e10
        cond['current_epoch'] = 1e10
        mesh_canonical = generate_mesh(lambda x: self.query_oc(x, cond), self.model.smpl_server.verts_c[0], point_batch=10000, res_up=4, device=self.device)
        self.model.deformer = SMPLDeformer(betas=inputs['shape'], gender=self.gender, K=7).to(self.device)
        verts_deformed = self.get_deformed_mesh_fast_mode(mesh_canonical.vertices, smpl_tfs)
        mesh_deformed = trimesh.Trimesh(vertices=verts_deformed, faces=mesh_canonical.faces, process=False)

        os.makedirs(f"{self.opt.dataset.testing.type}/test_rendering", exist_ok=True)
        os.makedirs(f"{self.opt.dataset.testing.type}/test_mesh", exist_ok=True)

        mesh_deformed.export(f"{self.opt.dataset.testing.type}/test_mesh/{int(idx.cpu().numpy()):04d}_deformed.ply")

        self.model.deformer = SMPLDeformer(betas=inputs['shape'], gender=self.gender).to(self.device)

        for i in range(num_splits):
            indices = list(range(i * pixel_per_batch,
                                min((i + 1) * pixel_per_batch, total_pixels)))
            batch_inputs = {"cam_loc": inputs["cam_loc"][:, indices],
                            "ray_dirs": inputs['ray_dirs'][:, indices],
                            "near": inputs['near'][:, indices],
                            "far": inputs['far'][:, indices],
                            "bgcolor": inputs['bgcolor'],
                            "smpl_params": inputs["smpl_params"],
                            "smpl_pose": inputs["smpl_params"][:, 4:76],
                            "smpl_shape": inputs["smpl_params"][:, 76:],
                            "smpl_trans": inputs["smpl_params"][:, 1:4],
                            "cam_idxs": torch.ones_like(inputs["idx"]) * -1,
                            "global_step": 1e10,
                            "current_epoch": 1e10,
                            "idx": inputs["idx"] if 'idx' in inputs.keys() else None}

            if idx_pose > -1:
                body_model_params = self.body_model_params(idx_pose)

                batch_inputs.update(
                    {'smpl_pose': torch.cat((batch_inputs['smpl_pose'][..., :3], body_model_params['body_pose']), dim=1)})
                batch_inputs.update({'smpl_shape': body_model_params['betas']})

            batch_targets = {"rgb": targets["rgb"][:, indices].detach().clone() if 'rgb' in targets.keys() else None,
                             "img_size": targets["img_size"]}

            with torch.no_grad():
                model_outputs = self.model(batch_inputs)
            results.append({"rgb_values":model_outputs["rgb_values"].detach().clone(),
                            "lf_rgb_values":model_outputs["lf_rgb_values"].detach().clone(),
                            "fg_rgb_values":model_outputs["fg_rgb_values"].detach().clone(),
                            "normal_values": model_outputs["normal_values"].detach().clone(),
                            "lf_normal_values": model_outputs["lf_normal_values"].detach().clone(),
                            "acc_map": model_outputs["acc_map"].detach().clone(),
                            **batch_targets})

        img_size = results[0]["img_size"]
        rgb_gt = targets["rgb"][0]
        rgb_gt = rgb_gt.reshape(*img_size, -1).cpu().numpy()

        rgb_bg = (torch.ones((img_size[0], img_size[1], 3), device=model_outputs["rgb_values"].device) * targets['bgcolor']).reshape(-1, 3)
        ray_mask = targets["ray_mask"].reshape(-1)
        rgb_pred_ray = torch.cat([result["rgb_values"] for result in results], dim=0)
        rgb_pred = rgb_bg.clone()
        rgb_pred[ray_mask] = rgb_pred_ray
        rgb_pred = rgb_pred.reshape(*img_size, -1)

        if 'lf_rgb_values' in results[0]:
            lf_rgb_pred_ray = torch.cat([result["lf_rgb_values"] for result in results], dim=0)
            lf_rgb_pred = rgb_bg.clone()
            lf_rgb_pred[ray_mask] = lf_rgb_pred_ray
            lf_rgb_pred = lf_rgb_pred.reshape(*img_size, -1)
            rgb_pred = torch.cat([rgb_pred, lf_rgb_pred], dim=1).cpu().numpy()

        normal_pred_ray = torch.cat([result["normal_values"] for result in results], dim=0)
        normal_pred = torch.ones_like(rgb_bg)
        normal_pred[ray_mask] = normal_pred_ray
        normal_pred = normal_pred.reshape(*img_size, -1)

        if 'lf_normal_values' in results[0]:
            lf_normal_pred_ray = torch.cat([result["lf_normal_values"] for result in results], dim=0)
            lf_normal_pred = torch.ones_like(rgb_bg)
            lf_normal_pred[ray_mask] = lf_normal_pred_ray
            lf_normal_pred = lf_normal_pred.reshape(*img_size, -1)
            normal_pred = torch.cat([normal_pred, lf_normal_pred], dim=1).cpu().numpy()

        pred_mask_ray = torch.cat([result["acc_map"] for result in results], dim=0)
        pred_mask = rgb_bg[..., 0].clone()
        pred_mask[ray_mask] = pred_mask_ray
        pred_mask = pred_mask.reshape(*img_size, -1)

        rgb_gt = (255. * np.clip(rgb_gt, 0., 1.)).astype(np.uint8)
        rgb_pred = (255. * np.clip(rgb_pred, 0., 1.)).astype(np.uint8)
        normal_pred = (255. * np.clip(normal_pred, 0., 1.)).astype(np.uint8)

        img_msk = pred_mask.cpu().numpy() * 255
        img_msk = np.concatenate([img_msk, img_msk, img_msk], axis=-1)

        img_out1 = np.concatenate([rgb_gt, rgb_pred], axis=1)
        img_out2 = np.concatenate([img_msk, normal_pred], axis=1)
        img_out = np.concatenate([img_out1, img_out2], axis=0)

        cv2.imwrite(f"{self.opt.dataset.testing.type}/test_rendering/{int(idx.cpu().numpy()):04d}.png", img_out[:, :, ::-1])
