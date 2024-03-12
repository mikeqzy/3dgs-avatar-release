#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from models import GaussianConverter
from scene.gaussian_model import GaussianModel
from dataset import load_dataset


class Scene:

    gaussians : GaussianModel

    def __init__(self, cfg, gaussians : GaussianModel, save_dir : str):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.cfg = cfg

        self.save_dir = save_dir
        self.gaussians = gaussians

        self.train_dataset = load_dataset(cfg.dataset, split='train')
        self.metadata = self.train_dataset.metadata
        if cfg.mode == 'train':
            self.test_dataset = load_dataset(cfg.dataset, split='val')
        elif cfg.mode == 'test':
            self.test_dataset = load_dataset(cfg.dataset, split='test')
        elif cfg.mode == 'predict':
            self.test_dataset = load_dataset(cfg.dataset, split='predict')
        else:
            raise ValueError

        self.cameras_extent = self.metadata['cameras_extent']

        self.gaussians.create_from_pcd(self.test_dataset.readPointCloud(), spatial_lr_scale=self.cameras_extent)

        self.converter = GaussianConverter(cfg, self.metadata).cuda()

    def train(self):
        self.converter.train()

    def eval(self):
        self.converter.eval()

    def optimize(self, iteration):
        gaussians_delay = self.cfg.model.gaussian.get('delay', 0)
        if iteration >= gaussians_delay:
            self.gaussians.optimizer.step()
        self.gaussians.optimizer.zero_grad(set_to_none=True)
        self.converter.optimize()

    def convert_gaussians(self, viewpoint_camera, iteration, compute_loss=True):
        return self.converter(self.gaussians, viewpoint_camera, iteration, compute_loss)

    def get_skinning_loss(self):
        loss_reg = self.converter.deformer.rigid.regularization()
        loss_skinning = loss_reg.get('loss_skinning', torch.tensor(0.).cuda())
        return loss_skinning

    def save(self, iteration):
        point_cloud_path = os.path.join(self.save_dir, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def save_checkpoint(self, iteration):
        print("\n[ITER {}] Saving Checkpoint".format(iteration))
        torch.save((self.gaussians.capture(),
                    self.converter.state_dict(),
                    self.converter.optimizer.state_dict(),
                    self.converter.scheduler.state_dict(),
                    iteration), self.save_dir + "/ckpt" + str(iteration) + ".pth")

    def load_checkpoint(self, path):
        (gaussian_params, converter_sd, converter_opt_sd, converter_scd_sd, first_iter) = torch.load(path)
        self.gaussians.restore(gaussian_params, self.cfg.opt)
        self.converter.load_state_dict(converter_sd)
        # self.converter.optimizer.load_state_dict(converter_opt_sd)
        # self.converter.scheduler.load_state_dict(converter_scd_sd)