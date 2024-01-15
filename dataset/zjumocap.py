import os
import sys
import glob
import cv2
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from utils.dataset_utils import get_02v_bone_transforms, fetchPly, storePly, AABB
from scene.cameras import Camera
from utils.camera_utils import freeview_camera


import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
import trimesh

class ZJUMoCapDataset(Dataset):
    def __init__(self, cfg, split='train'):
        super().__init__()
        self.cfg = cfg
        self.split = split

        self.root_dir = cfg.root_dir
        self.refine = cfg.refine
        if self.refine:
            self.root_dir = "../../data/refined_ZJUMoCap_arah_format"

        self.subject = cfg.subject
        self.train_frames = cfg.train_frames
        self.train_cams = cfg.train_views
        self.val_frames = cfg.val_frames
        self.val_cams = cfg.val_views
        self.white_bg = cfg.white_background
        self.H, self.W = 1024, 1024 # hardcoded original size
        self.h, self.w = cfg.img_hw

        self.faces = np.load('body_models/misc/faces.npz')['faces']
        self.skinning_weights = dict(np.load('body_models/misc/skinning_weights_all.npz'))
        self.posedirs = dict(np.load('body_models/misc/posedirs_all.npz'))
        self.J_regressor = dict(np.load('body_models/misc/J_regressors.npz'))

        if split == 'train':
            cam_names = self.train_cams
            frames = self.train_frames
        elif split == 'val':
            cam_names = self.val_cams
            frames = self.val_frames
        elif split == 'test':
            cam_names = self.cfg.test_views[self.cfg.test_mode]
            frames = self.cfg.test_frames[self.cfg.test_mode]
        elif split == 'predict':
            cam_names = self.cfg.predict_views
            frames = self.cfg.predict_frames
        else:
            raise ValueError

        with open(os.path.join(self.root_dir, self.subject, 'cam_params.json'), 'r') as f:
            self.cameras = json.load(f)

        if len(cam_names) == 0:
            cam_names = self.cameras['all_cam_names']
        elif self.refine:
            cam_names = [f'{int(cam_name) - 1:02d}' for cam_name in cam_names]


        start_frame, end_frame, sampling_rate = frames

        subject_dir = os.path.join(self.root_dir, self.subject)
        if split == 'predict':
            predict_seqs = ['gBR_sBM_cAll_d04_mBR1_ch05_view1',
                            'gBR_sBM_cAll_d04_mBR1_ch06_view1',
                            'MPI_Limits-03099-op8_poses_view1',
                            'canonical_pose_view1',]
            predict_seq = self.cfg.get('predict_seq', 0)
            predict_seq = predict_seqs[predict_seq]
            model_files = sorted(glob.glob(os.path.join(subject_dir, predict_seq, '*.npz')))
            self.model_files = model_files
            frames = list(reversed(range(-len(model_files), 0)))
            if end_frame == 0:
                end_frame = len(model_files)
            frame_slice = slice(start_frame, end_frame, sampling_rate)
            model_files = model_files[frame_slice]
            frames = frames[frame_slice]
        else:
            if self.cfg.get('arah_opt', False):
                model_files = sorted(glob.glob(os.path.join(subject_dir, 'opt_models/*.npz')))
            else:
                model_files = sorted(glob.glob(os.path.join(subject_dir, 'models/*.npz')))
            self.model_files = model_files
            frames = list(range(len(model_files)))
            if end_frame == 0:
                end_frame = len(model_files)
            frame_slice = slice(start_frame, end_frame, sampling_rate)
            model_files = model_files[frame_slice]
            frames = frames[frame_slice]

        # add freeview rendering
        if cfg.freeview:
            # with open(os.path.join(self.root_dir, self.subject, 'freeview_cam_params.json'), 'r') as f:
            #     self.cameras = json.load(f)
            model_dict = np.load(model_files[0])
            trans = model_dict['trans'].astype(np.float32)
            self.cameras = freeview_camera(self.cameras[cam_names[0]], trans)
            cam_names = self.cameras['all_cam_names']

        self.data = []
        if split == 'predict' or cfg.freeview:
            for cam_idx, cam_name in enumerate(cam_names):
                cam_dir = os.path.join(subject_dir, cam_name)

                for d_idx, f_idx in enumerate(frames):
                    model_file = model_files[d_idx]
                    # get dummy gt...
                    # img_file = glob.glob(os.path.join(cam_dir, '*.jpg'))[0]
                    img_file = os.path.join(subject_dir, '1', '000000.jpg')
                    # mask_file = glob.glob(os.path.join(cam_dir, '*.png'))[0]
                    mask_file = os.path.join(subject_dir, '1', '000000.png')

                    self.data.append({
                        'cam_idx': cam_idx,
                        'cam_name': cam_name,
                        'data_idx': d_idx,
                        'frame_idx': f_idx,
                        'img_file': img_file,
                        'mask_file': mask_file,
                        'model_file': model_file,
                    })
        else:
            for cam_idx, cam_name in enumerate(cam_names):
                cam_dir = os.path.join(subject_dir, cam_name)
                img_files = sorted(glob.glob(os.path.join(cam_dir, '*.jpg')))[frame_slice]
                mask_files = sorted(glob.glob(os.path.join(cam_dir, '*.png')))[frame_slice]

                for d_idx, f_idx in enumerate(frames):
                    img_file = img_files[d_idx]
                    mask_file = mask_files[d_idx]
                    model_file = model_files[d_idx]

                    self.data.append({
                        'cam_idx': cam_idx,
                        'cam_name': cam_name,
                        'data_idx': d_idx,
                        'frame_idx': f_idx,
                        'img_file': img_file,
                        'mask_file': mask_file,
                        'model_file': model_file,
                    })

        self.frames = frames
        self.model_files_list = model_files

        self.get_metadata()

        self.preload = cfg.get('preload', True)
        if self.preload:
            self.cameras = [self.getitem(idx) for idx in range(len(self))]


    def get_metadata(self):
        data_paths = self.model_files
        data_path = data_paths[0]

        cano_data = self.get_cano_smpl_verts(data_path)
        if self.split != 'train':
            self.metadata = cano_data
            return

        start, end, step = self.train_frames
        frames = list(range(len(data_paths)))
        if end == 0:
            end = len(frames)
        frame_slice = slice(start, end, step)
        frames = frames[frame_slice]

        frame_dict = {
            frame: i for i, frame in enumerate(frames)
        }

        self.metadata = {
            'faces': self.faces,
            'posedirs': self.posedirs,
            'J_regressor': self.J_regressor,
            'cameras_extent': 3.469298553466797, # hardcoded, used to scale the threshold for scaling/image-space gradient
            'frame_dict': frame_dict,
        }
        self.metadata.update(cano_data)
        if self.cfg.train_smpl:
            self.metadata.update(self.get_smpl_data())


    def get_cano_smpl_verts(self, data_path):
        '''
            Compute star-posed SMPL body vertices.
            To get a consistent canonical space,
            we do not add pose blend shape
        '''
        # compute scale from SMPL body
        model_dict = np.load(data_path)
        gender = 'neutral'

        # 3D models and points
        minimal_shape = model_dict['minimal_shape']
        # Break symmetry if given in float16:
        if minimal_shape.dtype == np.float16:
            minimal_shape = minimal_shape.astype(np.float32)
            minimal_shape += 1e-4 * np.random.randn(*minimal_shape.shape)
        else:
            minimal_shape = minimal_shape.astype(np.float32)

        # Minimally clothed shape
        J_regressor = self.J_regressor[gender]
        Jtr = np.dot(J_regressor, minimal_shape)

        skinning_weights = self.skinning_weights[gender]
        # Get bone transformations that transform a SMPL A-pose mesh
        # to a star-shaped A-pose (i.e. Vitruvian A-pose)
        bone_transforms_02v = get_02v_bone_transforms(Jtr)

        T = np.matmul(skinning_weights, bone_transforms_02v.reshape([-1, 16])).reshape([-1, 4, 4])
        vertices = np.matmul(T[:, :3, :3], minimal_shape[..., np.newaxis]).squeeze(-1) + T[:, :3, -1]

        coord_max = np.max(vertices, axis=0)
        coord_min = np.min(vertices, axis=0)
        padding_ratio = self.cfg.padding
        padding_ratio = np.array(padding_ratio, dtype=np.float)
        padding = (coord_max - coord_min) * padding_ratio
        coord_max += padding
        coord_min -= padding

        cano_mesh = trimesh.Trimesh(vertices=vertices.astype(np.float32), faces=self.faces)

        return {
            'gender': gender,
            'smpl_verts': vertices.astype(np.float32),
            'minimal_shape': minimal_shape,
            'Jtr': Jtr,
            'skinning_weights': skinning_weights.astype(np.float32),
            'bone_transforms_02v': bone_transforms_02v,
            'cano_mesh': cano_mesh,

            'coord_min': coord_min,
            'coord_max': coord_max,
            'aabb': AABB(coord_max, coord_min),
        }

    def get_smpl_data(self):
        # load all smpl fitting of the training sequence
        if self.split != 'train':
            return {}

        from collections import defaultdict
        smpl_data = defaultdict(list)

        for idx, (frame, model_file) in enumerate(zip(self.frames, self.model_files_list)):
            model_dict = np.load(model_file)

            if idx == 0:
                smpl_data['betas'] = model_dict['betas'].astype(np.float32)

            smpl_data['frames'].append(frame)
            smpl_data['root_orient'].append(model_dict['root_orient'].astype(np.float32))
            smpl_data['pose_body'].append(model_dict['pose_body'].astype(np.float32))
            smpl_data['pose_hand'].append(model_dict['pose_hand'].astype(np.float32))
            smpl_data['trans'].append(model_dict['trans'].astype(np.float32))

        return smpl_data

    def __len__(self):
        return len(self.data)

    def getitem(self, idx, data_dict=None):
        if data_dict is None:
            data_dict = self.data[idx]
        cam_idx = data_dict['cam_idx']
        cam_name = data_dict['cam_name']
        data_idx = data_dict['data_idx']
        frame_idx = data_dict['frame_idx']
        img_file = data_dict['img_file']
        mask_file = data_dict['mask_file']
        model_file = data_dict['model_file']

        K = np.array(self.cameras[cam_name]['K'], dtype=np.float32).copy()
        dist = np.array(self.cameras[cam_name]['D'], dtype=np.float32).ravel()
        R = np.array(self.cameras[cam_name]['R'], np.float32)
        T = np.array(self.cameras[cam_name]['T'], np.float32)

        # note that in ZJUMoCap the camera center does not align perfectly
        # here we try to offset it by modifying the extrinsic...
        M = np.eye(3)
        M[0, 2] = (K[0, 2] - self.W / 2) / K[0, 0]
        M[1, 2] = (K[1, 2] - self.H / 2) / K[1, 1]
        K[0, 2] = self.W / 2
        K[1, 2] = self.H / 2
        R = M @ R
        T = M @ T

        R = np.transpose(R)
        T = T[:, 0]

        image = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)

        if self.refine:
            mask = cv2.imread(mask_file)
            mask = mask.sum(-1)
            mask[mask != 0] = 100
            mask = mask.astype(np.uint8)
        else:
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        image = cv2.undistort(image, K, dist, None)
        mask = cv2.undistort(mask, K, dist, None)

        lanczos = self.cfg.get('lanczos', False)
        interpolation = cv2.INTER_LANCZOS4 if lanczos else cv2.INTER_LINEAR

        image = cv2.resize(image, (self.w, self.h), interpolation=interpolation)
        mask = cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)

        mask = mask != 0
        image[~mask] = 255. if self.white_bg else 0.
        image = image / 255.

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        # update camera parameters
        K[0, :] *= self.w / self.W
        K[1, :] *= self.h / self.H

        focal_length_x = K[0, 0]
        focal_length_y = K[1, 1]
        FovY = focal2fov(focal_length_y, self.h)
        FovX = focal2fov(focal_length_x, self.w)

        # Compute posed SMPL body
        minimal_shape = self.metadata['minimal_shape']
        gender = self.metadata['gender']

        model_dict = np.load(model_file)
        n_smpl_points = minimal_shape.shape[0]
        trans = model_dict['trans'].astype(np.float32)
        bone_transforms = model_dict['bone_transforms'].astype(np.float32)
        # Also get GT SMPL poses
        root_orient = model_dict['root_orient'].astype(np.float32)
        pose_body = model_dict['pose_body'].astype(np.float32)
        pose_hand = model_dict['pose_hand'].astype(np.float32)
        # Jtr_posed = model_dict['Jtr_posed'].astype(np.float32)
        pose = np.concatenate([root_orient, pose_body, pose_hand], axis=-1)
        pose = Rotation.from_rotvec(pose.reshape([-1, 3]))

        pose_mat_full = pose.as_matrix()  # 24 x 3 x 3
        pose_mat = pose_mat_full[1:, ...].copy()  # 23 x 3 x 3
        pose_rot = np.concatenate([np.expand_dims(np.eye(3), axis=0), pose_mat], axis=0).reshape(
            [-1, 9])  # 24 x 9, root rotation is set to identity
        pose_rot_full = pose_mat_full.reshape([-1, 9])  # 24 x 9, including root rotation

        # Minimally clothed shape
        posedir = self.posedirs[gender]
        Jtr = self.metadata['Jtr']

        # canonical SMPL vertices without pose correction, to normalize joints
        center = np.mean(minimal_shape, axis=0)
        minimal_shape_centered = minimal_shape - center
        cano_max = minimal_shape_centered.max()
        cano_min = minimal_shape_centered.min()
        padding = (cano_max - cano_min) * 0.05

        # compute pose condition
        Jtr_norm = Jtr - center
        Jtr_norm = (Jtr_norm - cano_min + padding) / (cano_max - cano_min) / 1.1
        Jtr_norm -= 0.5
        Jtr_norm *= 2.

        # final bone transforms that transforms the canonical Vitruvian-pose mesh to the posed mesh
        # without global translation
        bone_transforms_02v = self.metadata['bone_transforms_02v']
        bone_transforms = bone_transforms @ np.linalg.inv(bone_transforms_02v)
        bone_transforms = bone_transforms.astype(np.float32)
        bone_transforms[:, :3, 3] += trans  # add global offset

        return Camera(
            frame_id=frame_idx,
            cam_id=int(cam_name),
            K=K, R=R, T=T,
            FoVx=FovX,
            FoVy=FovY,
            image=image,
            mask=mask,
            gt_alpha_mask=None,
            image_name=f"c{int(cam_name):02d}_f{frame_idx if frame_idx >= 0 else -frame_idx - 1:06d}",
            data_device=self.cfg.data_device,
            # human params
            rots=torch.from_numpy(pose_rot).float().unsqueeze(0),
            Jtrs=torch.from_numpy(Jtr_norm).float().unsqueeze(0),
            bone_transforms=torch.from_numpy(bone_transforms),
        )

    def __getitem__(self, idx):
        if self.preload:
            return self.cameras[idx]
        else:
            return self.getitem(idx)

    def readPointCloud(self,):
        if self.cfg.get('random_init', False):
            ply_path = os.path.join(self.root_dir, self.subject, 'random_pc.ply')

            aabb = self.metadata['aabb']
            coord_min = aabb.coord_min.unsqueeze(0).numpy()
            coord_max = aabb.coord_max.unsqueeze(0).numpy()
            n_points = 50_000

            xyz_norm = np.random.rand(n_points, 3)
            xyz = xyz_norm * coord_min + (1. - xyz_norm) * coord_max
            rgb = np.ones_like(xyz) * 255
            storePly(ply_path, xyz, rgb)

            pcd = fetchPly(ply_path)
        else:
            ply_path = os.path.join(self.root_dir, self.subject, 'cano_smpl.ply')
            try:
                pcd = fetchPly(ply_path)
            except:
                verts = self.metadata['smpl_verts']
                faces = self.faces
                mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                n_points = 50_000

                xyz = mesh.sample(n_points)
                rgb = np.ones_like(xyz) * 255
                storePly(ply_path, xyz, rgb)

                pcd = fetchPly(ply_path)

        return pcd
