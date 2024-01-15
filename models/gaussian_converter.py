import torch
import torch.nn as nn
import numpy as np
from .deformer import get_deformer
from .pose_correction import get_pose_correction
from .texture import get_texture

class GaussianConverter(nn.Module):
    def __init__(self, cfg, metadata):
        super().__init__()
        self.cfg = cfg
        self.metadata = metadata

        self.pose_correction = get_pose_correction(cfg.model.pose_correction, metadata)
        self.deformer = get_deformer(cfg.model.deformer, metadata)
        self.texture = get_texture(cfg.model.texture, metadata)

        self.optimizer, self.scheduler = None, None
        self.set_optimizer()

    def set_optimizer(self):
        opt_params = [
            {'params': self.deformer.rigid.parameters(), 'lr': self.cfg.opt.get('rigid_lr', 0.)},
            # {'params': self.deformer.non_rigid.parameters(), 'lr': self.cfg.opt.get('non_rigid_lr', 0.)},
            {'params': [p for n, p in self.deformer.non_rigid.named_parameters() if 'latent' not in n],
             'lr': self.cfg.opt.get('non_rigid_lr', 0.)},
            {'params': [p for n, p in self.deformer.non_rigid.named_parameters() if 'latent' in n],
             'lr': self.cfg.opt.get('nr_latent_lr', 0.), 'weight_decay': self.cfg.opt.get('latent_weight_decay', 0.05)},
            {'params': self.pose_correction.parameters(), 'lr': self.cfg.opt.get('pose_correction_lr', 0.)},
            {'params': [p for n, p in self.texture.named_parameters() if 'latent' not in n],
             'lr': self.cfg.opt.get('texture_lr', 0.)},
            {'params': [p for n, p in self.texture.named_parameters() if 'latent' in n],
             'lr': self.cfg.opt.get('tex_latent_lr', 0.), 'weight_decay': self.cfg.opt.get('latent_weight_decay', 0.05)},
        ]
        self.optimizer = torch.optim.Adam(params=opt_params, lr=0.001, eps=1e-15)

        gamma = self.cfg.opt.lr_ratio ** (1. / self.cfg.opt.iterations)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)

    def forward(self, gaussians, camera, iteration, compute_loss=True):
        loss_reg = {}
        # loss_reg.update(gaussians.get_opacity_loss())
        camera, loss_reg_pose = self.pose_correction(camera, iteration)

        # pose augmentation
        pose_noise = self.cfg.pipeline.get('pose_noise', 0.)
        if self.training and pose_noise > 0 and np.random.uniform() <= 0.5:
            camera = camera.copy()
            camera.rots = camera.rots + torch.randn(camera.rots.shape, device=camera.rots.device) * pose_noise

        deformed_gaussians, loss_reg_deformer = self.deformer(gaussians, camera, iteration, compute_loss)

        loss_reg.update(loss_reg_pose)
        loss_reg.update(loss_reg_deformer)

        color_precompute = self.texture(deformed_gaussians, camera)

        return deformed_gaussians, loss_reg, color_precompute

    def optimize(self):
        grad_clip = self.cfg.opt.get('grad_clip', 0.)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()