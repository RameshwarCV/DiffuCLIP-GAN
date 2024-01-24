import argparse
import copy
import gc
import hashlib
import itertools
import logging
import math
import os
import shutil
import warnings
from pathlib import Path
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, model_info, upload_folder
from packaging import version
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from PIL import Image
import requests
from transformers import AutoProcessor, CLIPModel
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from omegaconf import OmegaConf
import PIL
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import random
from torch.nn.functional import interpolate
import json
import os
import h5py
import tempfile
from einops import rearrange
import os
import numpy as np
import PIL

import random
import json
import pycocotools.mask as mask_utils
from collections import defaultdict
from skimage.transform import resize as imresize
import math
if is_wandb_available():
    import wandb
import numpy as np
import torch.nn.functional as F
from torch import nn
import torch
import math
from typing import Optional
from ldm.modules.cgip.tools import create_tensor_by_assign_samples_to_img
import torch.optim as optim
from torch.optim import lr_scheduler
def build_loaders_coco(batch_size):
    dset_kwargs = {
        'image_dir': './datasets/coco/images/train2017',
        'instances_json': './datasets/coco/annotations/instances_train2017.json',
        'stuff_json': './datasets/coco/annotations/stuff_train2017.json',
        'caption_json':'./datasets/coco/annotations/captions_train2017.json',
        'stuff_only': True,
        'image_size': (256,256),
        'normalize_images': True,
        'max_samples': None,
        'include_relationships': True,
        'min_object_size':0.02,
        'min_objects_per_image':3,
        'max_objects_per_image':8,
        'include_other':False,
        'instance_whitelist':None,
        'stuff_whitelist':None,
    }
    dset = CocoDatabase(**dset_kwargs)
    collate_fn = coco_collate_fn

    loader = DataLoader(dset, batch_size=batch_size, num_workers=4, shuffle=False, collate_fn=collate_fn)
    return loader,dset

def main():
    config = OmegaConf.load("./config_coco.yaml")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.cuda()
    clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    gcn=instantiate_from_config(config.model.params.cond_stage_config)
    gcn.cuda()
    discriminator = Discriminator().cuda()
    optimizer_G = optim.Adam(gcn.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    num_epochs = 25
    train_dataloader,dset= build_loaders_coco(16)
    global_step=0
    num_update_steps_per_epoch = len(train_dataloader)
    max_train_steps=num_epochs * num_update_steps_per_epoch
    progress_bar = tqdm(range(global_step, max_train_steps))
    progress_bar.set_description("Steps")
    checkpointing_steps=math.ceil(max_train_steps/5)
    loss_per_epoch=[]
    for epoch in range(num_epochs):
        for step,batch in enumerate(train_dataloader):
            global_step+=1
            imgs, objs, boxes, triples, obj_to_img, triple_to_img,img_ids = [x.cuda() for x in batch]
            graph_info=[imgs, objs, boxes, triples, obj_to_img, triple_to_img]
            c_local, c_global=gcn.encode_graph_local_global(graph_info)
            c_global.cuda()
            with torch.no_grad():
                input_clip=clip_processor(images=imgs, return_tensors="pt")
                clip_feature=clip_model.get_image_features(input_clip['pixel_values'].cuda()).cuda()
            transform_layer_real=nn.Linear(512, 768).cuda()
            real_data_transformed = transform_layer_real(clip_feature)
            optimizer_D.zero_grad()

            real_labels = torch.ones(imgs.size(0), 1).to("cuda")
            output = discriminator(real_data_transformed)
            loss_real = criterion(output, real_labels)
            loss_real.backward()

            
            generated_data = c_global
            transform_layer_generated=nn.Linear(512, 768).cuda()
            generated_data_transformed = transform_layer_generated(generated_data)
            fake_labels = torch.zeros(generated_data.size(0), 1).to("cuda")
            output = discriminator(generated_data_transformed.detach())
            loss_fake = criterion(output, fake_labels)
            loss_fake.backward()

            optimizer_D.step()

            optimizer_G.zero_grad()
            output = discriminator(generated_data_transformed)
            loss_G = criterion(output, real_labels)
            loss_G.backward()
            optimizer_G.step()
            progress_bar.update(1)
            logs = {"loss": (loss_real + loss_fake).detach().item()}
            progress_bar.set_postfix(**logs)
            if(global_step%checkpointing_steps==0):
                gcn_path=os.path.join("/ssd-storage/home/rameshwarm/SG_diffusion/SGDiff/Output_clip_gan","gcn"+str(global_step)+".pt")
                torch.save(gcn.state_dict(),gcn_path)
            
        print(
            f"Epoch [{epoch}/{num_epochs}] Loss D: {loss_real + loss_fake}, Loss G: {loss_G}"
        )
        loss_per_epoch.append([loss_real.detach().item(),loss_fake.detach().item()])
    gcn_path=os.path.join("/ssd-storage/home/rameshwarm/SG_diffusion/SGDiff/Output_clip_gan","gcn.pt")
    torch.save(gcn.state_dict(),gcn_path)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        x = self.main(x)
        return x
class CocoDatabase(Dataset):
    def __init__(self, image_dir, instances_json, stuff_json, caption_json, stuff_only=True,
                 image_size=256,
                 mask_size=16,
                 normalize_images=True, max_samples=None,
                 include_relationships=True, min_object_size=0.02,
                 min_objects_per_image=3, max_objects_per_image=8,
                 include_other=False, instance_whitelist=None, stuff_whitelist=None):
        """
        A PyTorch Dataset for loading Coco and Coco-Stuff annotations and converting
        them to scene graphs on the fly.

        Inputs:
        - image_dir: Path to a directory where images are held
        - instances_json: Path to a JSON file giving COCO annotations
        - stuff_json: (optional) Path to a JSON file giving COCO-Stuff annotations
        - stuff_only: (optional, default True) If True then only iterate over
          images which appear in stuff_json; if False then iterate over all images
          in instances_json.
        - image_size: Size (H, W) at which to load images. Default (64, 64).
        - mask_size: Size M for object segmentation masks; default 16.
        - normalize_image: If True then normalize images by subtracting ImageNet
          mean pixel and dividing by ImageNet std pixel.
        - max_samples: If None use all images. Other wise only use images in the
          range [0, max_samples). Default None.
        - include_relationships: If True then include spatial relationships; if
          False then only include the trivial __in_image__ relationship.
        - min_object_size: Ignore objects whose bounding box takes up less than
          this fraction of the image.
        - min_objects_per_image: Ignore images which have fewer than this many
          object annotations.
        - max_objects_per_image: Ignore images which have more than this many
          object annotations.
        - include_other: If True, include COCO-Stuff annotations which have category
          "other". Default is False, because I found that these were really noisy
          and pretty much impossible for the system to model.
        - instance_whitelist: None means use all instance categories. Otherwise a
          list giving a whitelist of instance category names to use.
        - stuff_whitelist: None means use all stuff categories. Otherwise a list
          giving a whitelist of stuff category names to use.
        """
        super(Dataset, self).__init__()
#______________________considers captions____________________________#
        caption_data = None
        with open(caption_json, 'r') as f:
            caption_data = json.load(f)
        captions={}
        for  annotations in caption_data['annotations']:
            captions[annotations['image_id']]=annotations['caption']
        self.captions=captions
#____________________________________________________________________#
        if stuff_only and stuff_json is None:
            print('WARNING: Got stuff_only=True but stuff_json=None.')
            print('Falling back to stuff_only=False.')

        self.image_dir = image_dir
        self.mask_size = mask_size
        self.max_samples = max_samples

        self.include_relationships = include_relationships
        self.set_image_size(image_size)

        with open(instances_json, 'r') as f:
            instances_data = json.load(f)

        stuff_data = None
        if stuff_json is not None and stuff_json != '':
            with open(stuff_json, 'r') as f:
                stuff_data = json.load(f)

        self.image_ids = []
        self.image_id_to_filename = {}
        self.image_id_to_size = {}
        for image_data in instances_data['images']:
            image_id = image_data['id']
            filename = image_data['file_name']
            width = image_data['width']
            height = image_data['height']
            self.image_ids.append(image_id)
            self.image_id_to_filename[image_id] = filename
            self.image_id_to_size[image_id] = (width, height)

        self.vocab = {
            'object_name_to_idx': {},
            'pred_name_to_idx': {},
        }
        object_idx_to_name = {}
        all_instance_categories = []
        for category_data in instances_data['categories']:
            category_id = category_data['id']
            category_name = category_data['name']
            all_instance_categories.append(category_name)
            object_idx_to_name[category_id] = category_name
            self.vocab['object_name_to_idx'][category_name] = category_id
        all_stuff_categories = []
        if stuff_data:
            for category_data in stuff_data['categories']:
                category_name = category_data['name']
                category_id = category_data['id']
                all_stuff_categories.append(category_name)
                object_idx_to_name[category_id] = category_name
                self.vocab['object_name_to_idx'][category_name] = category_id

        if instance_whitelist is None:
            instance_whitelist = all_instance_categories
        if stuff_whitelist is None:
            stuff_whitelist = all_stuff_categories
        category_whitelist = set(instance_whitelist) | set(stuff_whitelist)

        self.image_id_to_objects = defaultdict(list)
        for object_data in instances_data['annotations']:
            image_id = object_data['image_id']
            _, _, w, h = object_data['bbox']
            W, H = self.image_id_to_size[image_id]
            box_area = (w * h) / (W * H)
            box_ok = box_area > min_object_size
            object_name = object_idx_to_name[object_data['category_id']]
            category_ok = object_name in category_whitelist
            other_ok = object_name != 'other' or include_other
            if box_ok and category_ok and other_ok:
                self.image_id_to_objects[image_id].append(object_data)

        if stuff_data:
            image_ids_with_stuff = set()
            for object_data in stuff_data['annotations']:
                image_id = object_data['image_id']
                image_ids_with_stuff.add(image_id)
                _, _, w, h = object_data['bbox']
                W, H = self.image_id_to_size[image_id]
                box_area = (w * h) / (W * H)
                box_ok = box_area > min_object_size
                object_name = object_idx_to_name[object_data['category_id']]
                category_ok = object_name in category_whitelist
                other_ok = object_name != 'other' or include_other
                if box_ok and category_ok and other_ok:
                    self.image_id_to_objects[image_id].append(object_data)
            if stuff_only:
                new_image_ids = []
                for image_id in self.image_ids:
                    if image_id in image_ids_with_stuff:
                        new_image_ids.append(image_id)
                self.image_ids = new_image_ids
                
                all_image_ids = set(self.image_id_to_filename.keys())
                image_ids_to_remove = all_image_ids - image_ids_with_stuff
                for image_id in image_ids_to_remove:
                    self.image_id_to_filename.pop(image_id, None)
                    self.image_id_to_size.pop(image_id, None)
                    self.image_id_to_objects.pop(image_id, None)
                

        self.vocab['object_name_to_idx']['__image__'] = 0

        name_to_idx = self.vocab['object_name_to_idx']
        assert len(name_to_idx) == len(set(name_to_idx.values()))
        max_object_idx = max(name_to_idx.values())
        idx_to_name = ['NONE'] * (1 + max_object_idx)
        for name, idx in self.vocab['object_name_to_idx'].items():
            idx_to_name[idx] = name
        self.vocab['object_idx_to_name'] = idx_to_name

        new_image_ids = []
        total_objs = 0
        for image_id in self.image_ids:
            num_objs = len(self.image_id_to_objects[image_id])
            total_objs += num_objs
            if min_objects_per_image <= num_objs <= max_objects_per_image:
                new_image_ids.append(image_id)
        self.image_ids = new_image_ids
        print(len(self.image_ids))
        self.vocab['pred_idx_to_name'] = [
            '__in_image__',
            'left of',
            'right of',
            'above',
            'below',
            'inside',
            'surrounding',
        ]
        self.vocab['pred_name_to_idx'] = {}
        for idx, name in enumerate(self.vocab['pred_idx_to_name']):
            self.vocab['pred_name_to_idx'][name] = idx

    def set_image_size(self, image_size):
        print('called set_image_size', image_size)
        transform = [Resize(image_size), transforms.ToTensor()]
        self.transform = transforms.Compose(transform)
        self.image_size = (image_size, image_size)

    def total_objects(self):
        total_objs = 0
        for i, image_id in enumerate(self.image_ids):
            if self.max_samples and i >= self.max_samples:
                break
            num_objs = len(self.image_id_to_objects[image_id])
            total_objs += num_objs
        return total_objs

    def __len__(self):
        if self.max_samples is None:
            return len(self.image_ids)
        return min(len(self.image_ids), self.max_samples)

    def __getitem__(self, index):
        image_id = self.image_ids[index]

        filename = self.image_id_to_filename[image_id]
        image_path = os.path.join(self.image_dir, filename)

        with open(image_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                WW, HH = image.size
                image = self.transform(image.convert('RGB'))

        #image = image * 2 - 1

        H, W = self.image_size
        objs, boxes, masks = [], [], []
        for object_data in self.image_id_to_objects[image_id]:
            objs.append(object_data['category_id'])
            x, y, w, h = object_data['bbox']
            x0 = x / WW
            y0 = y / HH
            x1 = (x + w) / WW
            y1 = (y + h) / HH
            boxes.append(torch.FloatTensor([x0, y0, x1, y1]))

            mask = seg_to_mask(object_data['segmentation'], WW, HH)

            mx0, mx1 = int(round(x)), int(round(x + w))
            my0, my1 = int(round(y)), int(round(y + h))
            mx1 = max(mx0 + 1, mx1)
            my1 = max(my0 + 1, my1)
            mask = mask[my0:my1, mx0:mx1]
            mask = imresize(255.0 * mask, (self.mask_size, self.mask_size),
                            mode='constant')
            mask = torch.from_numpy((mask > 128).astype(np.int64))
            masks.append(mask)

        objs.append(self.vocab['object_name_to_idx']['__image__'])
        boxes.append(torch.FloatTensor([0, 0, 1, 1]))
        masks.append(torch.ones(self.mask_size, self.mask_size).long())

        objs = torch.LongTensor(objs)
        boxes = torch.stack(boxes, dim=0)
        masks = torch.stack(masks, dim=0)

        box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        obj_centers = []
        _, MH, MW = masks.size()
        for i, obj_idx in enumerate(objs):
            x0, y0, x1, y1 = boxes[i]
            mask = (masks[i] == 1)
            xs = torch.linspace(x0, x1, MW).view(1, MW).expand(MH, MW)
            ys = torch.linspace(y0, y1, MH).view(MH, 1).expand(MH, MW)
            if mask.sum() == 0:
                mean_x = 0.5 * (x0 + x1)
                mean_y = 0.5 * (y0 + y1)
            else:
                mean_x = xs[mask].mean()
                mean_y = ys[mask].mean()
            obj_centers.append([mean_x, mean_y])
        obj_centers = torch.FloatTensor(obj_centers)

        triples = []
        num_objs = objs.size(0)
        __image__ = self.vocab['object_name_to_idx']['__image__']
        real_objs = []
        if num_objs > 1:
            real_objs = (objs != __image__).nonzero().squeeze(1)
        for cur in real_objs:
            choices = [obj for obj in real_objs if obj != cur]
            if len(choices) == 0 or not self.include_relationships:
                break
            other = random.choice(choices)
            if random.random() > 0.5:
                s, o = cur, other
            else:
                s, o = other, cur

            sx0, sy0, sx1, sy1 = boxes[s]
            ox0, oy0, ox1, oy1 = boxes[o]
            d = obj_centers[s] - obj_centers[o]
            theta = math.atan2(d[1], d[0])

            if sx0 < ox0 and sx1 > ox1 and sy0 < oy0 and sy1 > oy1:
                p = 'surrounding'
            elif sx0 > ox0 and sx1 < ox1 and sy0 > oy0 and sy1 < oy1:
                p = 'inside'
            elif theta >= 3 * math.pi / 4 or theta <= -3 * math.pi / 4:
                p = 'left of'
            elif -3 * math.pi / 4 <= theta < -math.pi / 4:
                p = 'above'
            elif -math.pi / 4 <= theta < math.pi / 4:
                p = 'right of'
            elif math.pi / 4 <= theta < 3 * math.pi / 4:
                p = 'below'
            p = self.vocab['pred_name_to_idx'][p]
            triples.append([s, p, o])

        O = objs.size(0)
        in_image = self.vocab['pred_name_to_idx']['__in_image__']
        for i in range(O - 1):
            triples.append([i, in_image, O - 1])

        triples = torch.LongTensor(triples)
        # do not return the masks
        return image, objs, boxes, triples,torch.tensor([image_id])

def seg_to_mask(seg, width=1.0, height=1.0):
    if type(seg) == list:
        rles = mask_utils.frPyObjects(seg, height, width)
        rle = mask_utils.merge(rles)
    elif type(seg['counts']) == list:
        rle = mask_utils.frPyObjects(seg, height, width)
    else:
        rle = seg
    return mask_utils.decode(rle)


class COCOTrain(CocoDatabase):
    def __init__(self, image_dir, instances_json, stuff_json, stuff_only, **kwargs):
        super().__init__(image_dir=image_dir, instances_json=instances_json, stuff_json=stuff_json, stuff_only=stuff_only, **kwargs)

class COCOValidation(CocoDatabase):
    def __init__(self, image_dir, instances_json, stuff_json, stuff_only, **kwargs):
        super().__init__(image_dir=image_dir, instances_json=instances_json, stuff_json=stuff_json, stuff_only=stuff_only, **kwargs)


def coco_collate_fn(batch):
    all_imgs, all_objs, all_boxes, all_triples = [], [], [], []
    all_obj_to_img, all_triple_to_img = [], []
    all_img_ids=[]
    obj_offset = 0
    for i, (img, objs, boxes, triples,image_id) in enumerate(batch):
        all_imgs.append(img[None])
        if objs.dim() == 0 or triples.dim() == 0:
            continue
        O, T = objs.size(0), triples.size(0)
        all_objs.append(objs)
        all_boxes.append(boxes)
        triples = triples.clone()
        triples[:, 0] += obj_offset
        triples[:, 2] += obj_offset
        all_triples.append(triples)

        all_obj_to_img.append(torch.LongTensor(O).fill_(i))
        all_triple_to_img.append(torch.LongTensor(T).fill_(i))
        obj_offset += O
        all_img_ids.append(image_id)

    all_imgs = torch.cat(all_imgs)
    all_objs = torch.cat(all_objs)
    all_boxes = torch.cat(all_boxes)
    all_triples = torch.cat(all_triples)
    all_obj_to_img = torch.cat(all_obj_to_img)
    all_triple_to_img = torch.cat(all_triple_to_img)
    all_img_ids=torch.cat(all_img_ids)
    out = (all_imgs, all_objs, all_boxes, all_triples, all_obj_to_img, all_triple_to_img,all_img_ids)
    return out

class Resize(object):
    def __init__(self, size, interp=PIL.Image.BILINEAR):
        if isinstance(size, tuple):
            H, W = size
            self.size = (W, H)
        else:
            self.size = (size, size)
        self.interp = interp

    def __call__(self, img):
        return img.resize(self.size, self.interp)
    
if __name__ == "__main__":
    
    main()

