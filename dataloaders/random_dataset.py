import os
from os.path import join
import cv2
import torch
import numpy as np
import tempfile, shutil
import glob
import logging
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize
from PIL import Image
import torch.distributed as dist
from .depthanything_preprocess import _load_and_process_image, _load_and_process_depth

class RandomDataset(Dataset):
    def __init__(self, root_dir, resolution=None, crop_type=None, large_dir=None):
        self.root_dir = root_dir
        self.resolution = resolution
        self.crop_type = crop_type
        self.large_dir = large_dir

        # Check if input is video files or npy directory
        if self.root_dir.endswith('.mp4'):
            self.input_type = 'video'
            self.seq_paths = [self.root_dir]
        elif os.path.isdir(self.root_dir):
            # Check if directory contains .npy files or .mp4 files
            npy_files = glob.glob(join(self.root_dir, '*.npy'))
            mp4_files = glob.glob(join(self.root_dir, '*.mp4'))
            
            if npy_files:
                self.input_type = 'npy_folder'
                self.seq_paths = [self.root_dir]  # Single directory path
            elif mp4_files:
                self.input_type = 'video'
                self.seq_paths = sorted(mp4_files)
            else:
                raise ValueError(f"No .mp4 or .npy files found in {self.root_dir}")
        else:
            raise ValueError(f"provide an mp4 file or a directory of mp4/npy files")

    def __len__(self):
        return len(self.seq_paths)
        
    def __getitem__(self, idx):
        if self.input_type == 'video':
            return self._process_video(idx)
        else:
            return self._process_npy_folder(idx)

    def _process_video(self, idx):
        """Original video processing logic"""
        img_paths, tmpdirname = self.parse_seq_path(self.seq_paths[idx])
        img_paths = sorted(img_paths, key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
        imgs = []

        first_img = cv2.imread(img_paths[0])
        h, w = first_img.shape[:2]
        if max(h, w) > 2044: # set max long side to 2044
            logging.info("resizing long side of video to 2044")
            scale = 2044 / max(h, w)
            res = (int(w * scale), int(h * scale))
            logging.info(f"new resolution: {res}")
        else:
            res = (w, h)

        for img_path in img_paths:
            img, _ = _load_and_process_image(img_path, resolution=res, crop_type=None)
            imgs.append(img)
        
        if tmpdirname is not None:
            shutil.rmtree(tmpdirname)

        return dict(batch=torch.stack(imgs).float(), 
                    scene_name=os.path.basename(self.seq_paths[idx].split('.')[0]))

    def _process_npy_folder(self, idx):
        """New npy folder processing logic"""
        npy_dir = self.seq_paths[idx]
        npy_files = sorted(glob.glob(join(npy_dir, '*.npy')), 
                          key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
        
        if not npy_files:
            raise ValueError(f"No .npy files found in {npy_dir}")
            
        logging.info(f"Loading {len(npy_files)} .npy image files from {npy_dir}")
        
        imgs = []
        
        # Create temp directory to save npy files as images for processing
        tmpdirname = tempfile.mkdtemp()
        temp_img_paths = []
        
        try:
            for i, npy_file in enumerate(npy_files):
                # Load numpy array
                img_array = np.load(npy_file)
                
                # Convert to proper image format for cv2
                if img_array.dtype != np.uint8:
                    # Normalize to 0-255 if needed
                    if img_array.max() <= 1.0:
                        img_array = (img_array * 255).astype(np.uint8)
                    else:
                        img_array = img_array.astype(np.uint8)
                
                # Ensure proper shape (H, W, C) for cv2
                if len(img_array.shape) == 3 and img_array.shape[0] == 3:  # (C, H, W)
                    img_array = np.transpose(img_array, (1, 2, 0))  # Convert to (H, W, C)
                elif len(img_array.shape) == 2:  # Grayscale (H, W)
                    img_array = np.expand_dims(img_array, axis=2)  # (H, W, 1)
                
                # Save as temporary image file
                temp_img_path = os.path.join(tmpdirname, f"frame_{i:06d}.jpg")
                cv2.imwrite(temp_img_path, img_array)
                temp_img_paths.append(temp_img_path)
            
            # Get resolution from first image
            first_img = cv2.imread(temp_img_paths[0])
            h, w = first_img.shape[:2]
            if max(h, w) > 2044:
                logging.info("resizing long side to 2044")
                scale = 2044 / max(h, w)
                res = (int(w * scale), int(h * scale))
                logging.info(f"new resolution: {res}")
            else:
                res = (w, h)
            
            # Process images using existing pipeline
            for img_path in temp_img_paths:
                img, _ = _load_and_process_image(img_path, resolution=res, crop_type=None)
                imgs.append(img)
                
        finally:
            # Clean up temp directory
            shutil.rmtree(tmpdirname)

        scene_name = os.path.basename(npy_dir.rstrip('/'))
        return dict(batch=torch.stack(imgs).float(), scene_name=scene_name)

    def parse_seq_path(self, p):
        """Original video parsing logic - unchanged"""
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file {p}")
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if video_fps == 0:
            cap.release()
            raise ValueError(f"Error: Video FPS is 0 for {p}")
        frame_interval = 1
        frame_indices = list(range(0, total_frames, frame_interval))
        print(
            f" - Video FPS: {video_fps}, Frame Interval: {frame_interval}, Total Frames to Read: {len(frame_indices)}"
        )
        img_paths = []
        tmpdirname = tempfile.mkdtemp()
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(tmpdirname, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            img_paths.append(frame_path)
        cap.release()
        return img_paths, tmpdirname