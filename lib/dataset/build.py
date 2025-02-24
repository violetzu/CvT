import logging
import os

from timm.data import create_loader
import torch
import torch.utils.data
import torchvision.datasets as datasets
from PIL import Image
import numpy as np

import torchvision.transforms as T

from .transformas import build_transforms
from .samplers import RASampler


def build_dataset(cfg, is_train):
    """
    判斷 dataset 種類，建構並回傳對應的 Dataset。
    """
    dataset_name = cfg.DATASET.DATASET.lower()

    if 'imagenet' in dataset_name and 'multidim' not in dataset_name:
        # 原先的 ImagetNet (單圖) 讀取方式
        dataset = _build_imagenet_dataset(cfg, is_train)

    elif 'multidim_imagenet' in dataset_name:
        # 新增：多維度資料夾結構
        dataset = _build_multidim_imagenet_dataset(cfg, is_train)

    else:
        raise ValueError('Unknown dataset: {}'.format(cfg.DATASET.DATASET))

    logging.info(
        '=> [build_dataset] dataset={}, is_train={}, #samples={}'
        .format(cfg.DATASET.DATASET, is_train, len(dataset))
    )
    return dataset


def _build_imagenet_dataset(cfg, is_train):
    """
    標準 ImageNet 格式：/train/classX 與 /val/classX
    """
    transforms = build_transforms(cfg, is_train)
    dataset_name = cfg.DATASET.TRAIN_SET if is_train else cfg.DATASET.TEST_SET

    dataset = datasets.ImageFolder(
        os.path.join(cfg.DATASET.ROOT, dataset_name), transforms
    )
    return dataset


def _build_multidim_imagenet_dataset(cfg, is_train):
    """
    多維度 (dim1..dimN) ImageNet-like 結構。
    `cfg.DATASET.NUM_DIMS` 控制維度數量。
    """
    transforms = build_transforms(cfg, is_train)
    dataset_name = cfg.DATASET.TRAIN_SET if is_train else cfg.DATASET.TEST_SET

    root_path = os.path.join(cfg.DATASET.ROOT, dataset_name)
    dataset = MultiDimImageDataset(
        root=root_path, 
        transform=transforms, 
        num_dims=cfg.DATASET.NUM_DIMS  # 讀取設定中的數量
    )
    return dataset



def build_dataloader(cfg, is_train=True, distributed=False):
    """
    建立 DataLoader，兼容單機或分散式。
    """
    if is_train:
        batch_size_per_gpu = cfg.TRAIN.BATCH_SIZE_PER_GPU
        shuffle = True
    else:
        batch_size_per_gpu = cfg.TEST.BATCH_SIZE_PER_GPU
        shuffle = False

    dataset = build_dataset(cfg, is_train)

    # 分散式情況
    if distributed:
        if is_train and cfg.DATASET.SAMPLER == 'repeated_aug':
            logging.info('=> use repeated aug sampler')
            sampler = RASampler(dataset, shuffle=shuffle)
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=shuffle
            )
        shuffle = False
    else:
        sampler = None

    # 如果需要使用 timm 的 create_loader (針對單圖增強)
    # 注意多圖情況可能需要客製化
    if cfg.AUG.TIMM_AUG.USE_LOADER and is_train:
        logging.info('=> use timm loader for training')
        timm_cfg = cfg.AUG.TIMM_AUG
        data_loader = create_loader(
            dataset,
            input_size=cfg.TRAIN.IMAGE_SIZE[0],
            batch_size=batch_size_per_gpu,
            is_training=True,
            use_prefetcher=True,
            no_aug=False,
            re_prob=timm_cfg.RE_PROB,
            re_mode=timm_cfg.RE_MODE,
            re_count=timm_cfg.RE_COUNT,
            re_split=timm_cfg.RE_SPLIT,
            scale=cfg.AUG.SCALE,
            ratio=cfg.AUG.RATIO,
            hflip=timm_cfg.HFLIP,
            vflip=timm_cfg.VFLIP,
            color_jitter=timm_cfg.COLOR_JITTER,
            auto_augment=timm_cfg.AUTO_AUGMENT,
            num_aug_splits=0,
            interpolation=timm_cfg.INTERPOLATION,
            mean=cfg.INPUT.MEAN,
            std=cfg.INPUT.STD,
            num_workers=cfg.WORKERS,
            distributed=distributed,
            collate_fn=None,
            pin_memory=cfg.PIN_MEMORY,
            use_multi_epochs_loader=True
        )
    else:
        # 一般的 PyTorch DataLoader
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size_per_gpu,
            shuffle=shuffle,
            num_workers=cfg.WORKERS,
            pin_memory=cfg.PIN_MEMORY,
            sampler=sampler,
            drop_last=True if is_train else False,
        )

    for batch in data_loader:
        imgs, labels = batch
        logging.info(f"[DEBUG] Batch shape: {imgs.shape}")  # 應該要是 [B, 6, C, H, W]
        break

    return data_loader


# ----------------------------------------------------
# 自訂多維度 Dataset
# ----------------------------------------------------
class MultiDimImageDataset(torch.utils.data.Dataset):
    """
    適用於 /train/classX/dim1..dim6/ 與 /val/classX/dim1..dim6/ 結構。
    假設各個 dim 目錄下的檔案名稱一一對應。
    例如：
      root/
        class1/
          dim1/
            0001.jpg, 0002.jpg, ...
          dim2/
            0001.jpg, 0002.jpg, ...
          ...
          dim6/
        class2/
          dim1/...
          ...
          dim6/
    每個樣本對應到同一檔名在 dim1~dim6 的路徑。
    回傳 shape: [6, C, H, W] 的圖 + label
    """

    def __init__(self, root, transform=None, num_dims=6):
        super().__init__()
        self.root = root
        self.transform = transform
        self.num_dims = num_dims  # 讀取 cfg.DATASET.NUM_DIMS

        # 掃描所有 class 資料夾
        self.classes = sorted([
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []  # 儲存 (paths, label) 的列表

        for cls_name in self.classes:
            cls_path = os.path.join(root, cls_name)
            # dim 目錄
            dim_folders = sorted([
                d for d in os.listdir(cls_path)
                if os.path.isdir(os.path.join(cls_path, d))
            ])
            if len(dim_folders) != self.num_dims:
                logging.warning(f"Class '{cls_name}' under '{root}' doesn't have {self.num_dims} dim folders. Found: {dim_folders}")
                continue  # 跳過這個 class
                # 你也可以 raise Error 或其它處理

            # 取得 dim1 中所有檔案，作為「基準檔名清單」
            dim1_path = os.path.join(cls_path, dim_folders[0])
            dim1_files = sorted(os.listdir(dim1_path))

            for fname in dim1_files:
                # 確認 dim2..dimN 同樣有這個檔名
                all_dim_paths = []
                valid_sample = True
                for dimf in dim_folders:
                    f_path = os.path.join(cls_path, dimf, fname)
                    if not os.path.isfile(f_path):
                        valid_sample = False
                        break
                    all_dim_paths.append(f_path)

                if not valid_sample:
                    continue

                label = self.class_to_idx[cls_name]
                self.samples.append((all_dim_paths, label))

        logging.info(f"=> MultiDimImageDataset from '{root}' loaded. total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        all_dim_paths, label = self.samples[index]
        imgs = []
        for img_path in all_dim_paths:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)  # 這裡應該要產生 shape=(C, H, W)

            if not isinstance(img, torch.Tensor):
                img = T.ToTensor()(img)  # 確保轉為 Tensor 並維持正確 shape

            imgs.append(img)

        imgs = torch.stack(imgs, dim=0)  # [num_dims, C, H, W]
        return imgs, label
