#@title loading utils
import torch
from omegaconf import OmegaConf
import PIL
from util import instantiate_from_config

from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import random
import json
import os
import h5py
import tempfile
from einops import rearrange
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
from diffusers import LMSDiscreteScheduler
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
from skimage.transform import resize as imresize
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
from tqdm.auto import tqdm
from torch import autocast
from omegaconf import OmegaConf
import PIL
from util import instantiate_from_config

from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np



from torch import nn


from typing import Optional
from tools import create_tensor_by_assign_samples_to_img

from PIL import Image
import random
import json
import os
import h5py
from collections import defaultdict
import tempfile
from einops import rearrange
import pycocotools.mask as mask_utils
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
if is_wandb_available():
    import wandb
from transformers import AutoProcessor, CLIPModel
from torch.nn.functional import interpolate
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.21.0.dev0")

logger = get_logger(__name__)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def build_loaders_coco():
    dset_kwargs = {
        'image_dir': './datasets/coco/images/val2017',
        'instances_json': './datasets/coco/annotations/instances_val2017.json',
        'stuff_json': './datasets/coco/annotations/stuff_val2017.json',
        'caption_json':'./datasets/coco/annotations/captions_val2017.json',
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

    loader = DataLoader(dset, batch_size=1, num_workers=4, shuffle=False, collate_fn=collate_fn)
    return loader,dset


def main():
    pretrained_model_path='./Output_coco_mmd'
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet") 
    scheduler = LMSDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

    vae = vae.to('cuda')
    text_encoder = text_encoder.to('cuda')
    unet = unet.to('cuda') 
    ddim_steps = 100
    ddim_eta = 1.0
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.cuda()
    loader,dset = build_loaders_coco()
    vocab=dset.vocab
    root_dir = './test_results_coco_mmd'
    scene_graph_dir = os.path.join(root_dir, 'scene_graph')
    generate_img_dir = os.path.join(root_dir, 'img')

    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    if not os.path.exists(scene_graph_dir):
        os.mkdir(scene_graph_dir)
    if not os.path.exists(generate_img_dir):
        os.mkdir(generate_img_dir)

    n_samples_per_scene_graph = 1
    gcn=CGIPModel(num_objs=184,num_preds=46,layers=5,width=512,embed_dim=512,ckpt_path="./Output_vg_concat/gcn.pt")
    gcn.cuda()
    
    with torch.no_grad():
            img_idx = -1
            for batch_data in loader:
                img_idx += 1
                imgs, objs, boxes, triples, obj_to_img, triple_to_img,img_ids = [x.cuda() for x in batch_data]
                scene_graph_path = os.path.join(scene_graph_dir, str(img_idx)+'_graph.png')
                draw_scene_graph(objs, triples, vocab, output_filename=scene_graph_path,orientation='V')
#___________________________synthetic prompt_____________________________________
                obj_string=[]
                for obj in objs:
                    obj_string.append(vocab['object_idx_to_name'][obj])
                obj_string[len(obj_string)-1]='in_image'
                graph_prompt=",".join(obj_string)
#______________________________________________________________________
                
                no_of_tokens=len(graph_prompt.split(" "))
                #print(graph_prompt,no_of_tokens)
                #print(img_idx,graph_prompt)
                #draw_scene_graph(objs=objs, triples=triples, vocab=vocab, output_filename=scene_graph_path)   
                # #torch.zeros(1,3,256,256)             
                graph_info = [imgs, objs, None, triples, obj_to_img, triple_to_img]
                c_local, c_global=gcn.encode_graph_local_global(graph_info)
                c_global.cuda()
                input_clip=clip_processor(images=imgs, return_tensors="pt")
                clip_feature=clip_model.get_image_features(input_clip['pixel_values'].cuda()).cuda()
                height = 256                        # default height of Stable Diffusion
                width = 256                         # default width of Stable Diffusion

                num_inference_steps = 100            # Number of denoising steps
                scheduler.set_timesteps(num_inference_steps)
                guidance_scale = 7.5                # Scale for classifier-free guidance
                #earlier used 11
                generator = torch.manual_seed(11)   # Seed generator to create the inital latent noise

                batch_size = 1

                text_input = tokenizer(graph_prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
                text_embeddings = text_encoder(text_input.input_ids.to('cuda'))[0]
                # buffer_tensor=torch.zeros(1,256).cuda()
                # graph_encoding_tensor=torch.cat((c_global,buffer_tensor),dim=1).unsqueeze(0)
                graph_encoding_tensor=interpolate(c_global.unsqueeze(1),(768))
                clip_feature_tensor=torch.zeros(graph_encoding_tensor.shape).cuda()
                #clip_feature_tensor=interpolate(clip_feature.unsqueeze(1),(768))
                encoding_tensor=torch.cat((graph_encoding_tensor,clip_feature_tensor),dim=1)
                #graph_encoding_tensor=torch.cat((graph_features,buffer_tensor),dim=1).unsqueeze(1)
                #encoder_hidden_state_concatenated=torch.cat((encoding_tensor,encoder_hidden_states),dim=1).cuda().to(dtype=weight_dtype)
                #text_embeddings[:,1:no_of_tokens,:]=0.7*text_embeddings[:,1:no_of_tokens,:]+0.3*clip_feature_tensor[:,:,:]
                #text_embeddings=torch.cat((encoding_tensor,text_embeddings),dim=1).cuda()

                encoder_hidden_state_concatenated=torch.zeros(text_embeddings.shape).cuda()
                nind=no_of_tokens+1
                aind=0
                bind=0
                isa=True
                for index_val in range(text_embeddings.shape[1]):
                    if(isa and aind<nind):
                        encoder_hidden_state_concatenated[:,index_val,:]=text_embeddings[:,aind,:]
                        aind+=1
                        isa=False
                        continue
                    if ((not isa) and bind<nind):
                        encoder_hidden_state_concatenated[:,index_val,:]=graph_encoding_tensor[:,:,:]
                        bind+=1
                        isa=True
                        continue
                    break
                if(2*nind < text_embeddings.shape[1]):
                    encoder_hidden_state_concatenated[:,2*nind:,:]=text_embeddings[:,2*nind:,:]

                # encoder_hidden_state_concatenated=torch.zeros(text_embeddings.shape).cuda()
                encoder_hidden_state_concatenated.cuda()
                max_length = text_input.input_ids.shape[-1]
                uncond_input = tokenizer(
                    [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
                )
                
                with torch.no_grad():
                    uncond_embeddings = text_encoder(uncond_input.input_ids.to('cuda'))[0]   

                encoder_hidden_state_concatenated_uncond=torch.zeros(uncond_embeddings.shape).cuda()
                nind=no_of_tokens+1
                aind=0
                bind=0
                isa=True
                for index_val in range(uncond_embeddings.shape[1]):
                    if(isa and aind<nind):
                        encoder_hidden_state_concatenated_uncond[:,index_val,:]=uncond_embeddings[:,aind,:]
                        aind+=1
                        isa=False
                        continue
                    if ((not isa) and bind<nind):
                        encoder_hidden_state_concatenated_uncond[:,index_val,:]=graph_encoding_tensor[:,:,:]
                        bind+=1
                        isa=True
                        continue
                    break
                if(2*nind < uncond_embeddings.shape[1]):
                    encoder_hidden_state_concatenated_uncond[:,2*nind:,:]=uncond_embeddings[:,2*nind:,:]

                #encoder_hidden_state_concatenated=encoder_hidden_states # to check dependence over gcn
                encoder_hidden_state_concatenated_uncond.cuda()
                #uncond_embeddings[:,1:no_of_tokens,:]=0.7*uncond_embeddings[:,1:no_of_tokens,:]+0.3*clip_feature_tensor[:,:,:]
                text_embeddings = torch.cat([encoder_hidden_state_concatenated_uncond, encoder_hidden_state_concatenated])
                #print(text_embeddings.shape)

                latents = torch.randn((batch_size, unet.in_channels, height // 8, width // 8),generator=generator,)
                latents = latents.to('cuda')
                latents = latents * scheduler.init_noise_sigma
                

                for t in tqdm(scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                    latent_model_input = torch.cat([latents] * 2)
                    #latent_model_input=latents
                    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    with torch.no_grad():
                        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                    # perform guidance
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = scheduler.step(noise_pred, t, latents).prev_sample
                latents = 1 / 0.18215 * latents

                with torch.no_grad():
                    image = vae.decode(latents).sample
                image = (image / 2 + 0.5).clamp(0, 1)
                image=image.squeeze(0)
                #print(image.shape)
                image = image.detach().cpu().permute(1, 2, 0).numpy()
                
                images = (image * 255).round().astype("uint8")
                #print(images.shape)
                pil_images = Image.fromarray(images)
                # if(img_idx>10):
                #     break
                
                pil_images.save(os.path.join(generate_img_dir, str(img_idx)+'_img.png'))

    return None

def draw_scene_graph(objs, triples, vocab=None, **kwargs):
    output_filename = kwargs.pop('output_filename', 'graph.png')
    orientation = kwargs.pop('orientation', 'V')
    edge_width = kwargs.pop('edge_width', 6)
    arrow_size = kwargs.pop('arrow_size', 1.5)
    binary_edge_weight = kwargs.pop('binary_edge_weight', 1.2)
    ignore_dummies = kwargs.pop('ignore_dummies', True)

    if orientation not in ['V', 'H']:
        raise ValueError('Invalid orientation "%s"' % orientation)
    rankdir = {'H': 'LR', 'V': 'TD'}[orientation]

    if vocab is not None:
        assert torch.is_tensor(objs)
        assert torch.is_tensor(triples)
        objs_list, triples_list = [], []
        for i in range(objs.size(0)):
            objs_list.append(vocab['object_idx_to_name'][objs[i].item()])
        for i in range(triples.size(0)):
            s = triples[i, 0].item()
            # p = vocab['pred_name_to_idx'][triples[i, 1].item()]
            p = triples[i, 1].item()
            o = triples[i, 2].item()
            triples_list.append([s, p, o])
        objs, triples = objs_list, triples_list

    lines = [
        'digraph{',
        'graph [size="5,3",ratio="compress",dpi="300",bgcolor="transparent"]',
        'rankdir=%s' % rankdir,
        'nodesep="0.5"',
        'ranksep="0.5"',
        'node [shape="box",style="rounded,filled",fontsize="48",color="none"]',
        'node [fillcolor="lightpink1"]',
    ]

    for i, obj in enumerate(objs):
        if ignore_dummies and obj == '__image__':
            continue
        lines.append('%d [label="%s"]' % (i, obj))

    next_node_id = len(objs)
    lines.append('node [fillcolor="lightblue1"]')
    for s, p, o in triples:
        p = vocab['pred_idx_to_name'][p]
        if ignore_dummies and p == '__in_image__':
            continue
        lines += [
            '%d [label="%s"]' % (next_node_id, p),
            '%d->%d [penwidth=%f,arrowsize=%f,weight=%f]' % (
                s, next_node_id, edge_width, arrow_size, binary_edge_weight),
            '%d->%d [penwidth=%f,arrowsize=%f,weight=%f]' % (
                next_node_id, o, edge_width, arrow_size, binary_edge_weight)
        ]
        next_node_id += 1
    lines.append('}')

    ff, dot_filename = tempfile.mkstemp()
    with open(dot_filename, 'w') as f:
        for line in lines:
            f.write('%s\n' % line)
    os.close(ff)

    output_format = os.path.splitext(output_filename)[1][1:]
    os.system('dot -T%s %s > %s' % (output_format, dot_filename, output_filename))
    return None

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=True)
    model.cuda()
    model.eval()
    return model

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

class CGIPModel(nn.Module):
    def __init__(self,
                 num_objs: int,
                 num_preds: int,
                 width: int,
                 layers: int,
                 embed_dim: int,
                 ckpt_path: str,
                 ignore_keys: list = [],
                 max_sample_per_img: int = 15
                 ):
        super().__init__()

        # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
        # memory efficient in recent PyTorch releases (>= 1.10).
        # NOTE: timm models always use native GELU regardless of quick_gelu flag.


        self.num_objs = num_objs
        self.num_preds = num_preds
        self.max_relationships_per_image = max_sample_per_img
        self.obj_embeddings = nn.Embedding(num_objs + 1, embed_dim)
        self.pred_embeddings = nn.Embedding(num_preds, embed_dim)

        self.graph_conv = GraphTripleConv(embed_dim, output_dim=embed_dim, hidden_dim=width, pooling='avg', mlp_normalization='none')
        self.graph_net = GraphTripleConvNet(embed_dim, num_layers=layers, hidden_dim=width, pooling='avg', mlp_normalization='none')
        self.graph_projection = nn.Linear(embed_dim * 2, embed_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path)
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")


    def encode_graph_local_global(self, graph):

        image, objs, boxes, triples, obj_to_img, triples_to_img = graph

        batch_size, _, H, W = image.shape

        s, p, o = triples.chunk(3, dim=1)  # All have shape (T, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (T,)
        edges = torch.stack([s, o], dim=1)  # Shape is (T, 2)

        obj_vecs = self.obj_embeddings(objs)
        pred_vecs = self.pred_embeddings(p)

        if isinstance(self.graph_conv, nn.Linear):
            obj_vecs = self.graph_conv(obj_vecs)
        else:
            obj_vecs, pred_vecs = self.graph_conv(obj_vecs, pred_vecs, edges)
        if self.graph_net is not None:
            obj_vecs, pred_vecs = self.graph_net(obj_vecs, pred_vecs, edges)

        obj_fea = self.pool_samples(obj_vecs, obj_to_img)
        pred_fea = self.pool_samples(pred_vecs, triples_to_img)
        graph_global_fea = self.graph_projection(torch.cat([obj_fea, pred_fea], dim=1))

        s_obj_vec, o_obj_vec = obj_vecs[s], obj_vecs[o]
        triple_vec = torch.cat([s_obj_vec, pred_vecs, o_obj_vec], dim=1)
        graph_local_fea = create_tensor_by_assign_samples_to_img(samples=triple_vec, sample_to_img=triples_to_img,
                                                                 max_sample_per_img=self.max_relationships_per_image,
                                                                 batch_size=batch_size)

        return graph_local_fea, graph_global_fea

    def forward(self, graph):
        graph_local_fea, graph_global_fea = self.encode_graph_local_global(graph)

        return graph_local_fea, graph_global_fea

    def pool_samples(self, samples, obj_to_img, pooling='avg'):
        dtype, device = samples.dtype, samples.device
        O, D = samples.size()

        N = obj_to_img.data.max().item() + 1

        out = torch.zeros(N, D, dtype=dtype, device=device)
        idx = obj_to_img.view(O, 1).expand(O, D)
        out = out.scatter_add(0, idx, samples)

        if pooling == 'avg':
            ones = torch.ones(O, dtype=dtype, device=device)
            obj_counts = torch.zeros(N, dtype=dtype, device=device)
            obj_counts = obj_counts.scatter_add(0, obj_to_img, ones)
            obj_counts = obj_counts.clamp(min=1)
            out = out / obj_counts.view(N, 1)
        elif pooling != 'sum':
            raise ValueError('Invalid pooling "%s"' % pooling)

        return out

def convert_weights_to_fp16(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, (nn.MultiheadAttention, Attention)):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def _init_weights(module):
    if hasattr(module, 'weight'):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)

class GraphTripleConv(nn.Module):
    """
    A single layer of scene graph convolution.
    """

    def __init__(self, input_dim, output_dim=None, hidden_dim=512,
                 pooling='avg', mlp_normalization='none'):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        assert pooling in ['sum', 'avg'], 'Invalid pooling "%s"' % pooling
        self.pooling = pooling
        net1_layers = [3 * input_dim, hidden_dim, 2 * hidden_dim + output_dim]
        net1_layers = [l for l in net1_layers if l is not None]
        self.net1 = build_mlp(net1_layers, batch_norm=mlp_normalization)

        net2_layers = [hidden_dim, hidden_dim, output_dim]
        self.net2 = build_mlp(net2_layers, batch_norm=mlp_normalization)

    def forward(self, obj_vecs, pred_vecs, edges):
        dtype, device = obj_vecs.dtype, obj_vecs.device
        O, T = obj_vecs.size(0), pred_vecs.size(0)
        Din, H, Dout = self.input_dim, self.hidden_dim, self.output_dim

        s_idx = edges[:, 0].contiguous()
        o_idx = edges[:, 1].contiguous()

        cur_s_vecs = obj_vecs[s_idx]
        cur_o_vecs = obj_vecs[o_idx]

        cur_t_vecs = torch.cat([cur_s_vecs, pred_vecs, cur_o_vecs], dim=1)
        new_t_vecs = self.net1(cur_t_vecs)

        new_s_vecs = new_t_vecs[:, :H]
        new_p_vecs = new_t_vecs[:, H:(H + Dout)]
        new_o_vecs = new_t_vecs[:, (H + Dout):(2 * H + Dout)]

        pooled_obj_vecs = torch.zeros(O, H, dtype=dtype, device=device)

        s_idx_exp = s_idx.view(-1, 1).expand_as(new_s_vecs)
        o_idx_exp = o_idx.view(-1, 1).expand_as(new_o_vecs)
        pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, s_idx_exp, new_s_vecs)
        pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, o_idx_exp, new_o_vecs)

        if self.pooling == 'avg':
            obj_counts = torch.zeros(O, dtype=dtype, device=device)
            ones = torch.ones(T, dtype=dtype, device=device)
            obj_counts = obj_counts.scatter_add(0, s_idx, ones)
            obj_counts = obj_counts.scatter_add(0, o_idx, ones)

            obj_counts = obj_counts.clamp(min=1)
            pooled_obj_vecs = pooled_obj_vecs / obj_counts.view(-1, 1)

        new_obj_vecs = self.net2(pooled_obj_vecs)

        return new_obj_vecs, new_p_vecs

    def init_parameters(self):
        self.net1.apply(_init_weights)
        self.net2.apply(_init_weights)

class GraphTripleConvNet(nn.Module):
    """ A sequence of scene graph convolution layers  """
    def __init__(self, input_dim, num_layers=5, hidden_dim=512, pooling='avg', mlp_normalization='none'):
        super().__init__()

        self.num_layers = num_layers
        self.gconvs = nn.ModuleList()
        gconv_kwargs = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,'pooling': pooling,
            'mlp_normalization': mlp_normalization,
        }
        for _ in range(self.num_layers):
            self.gconvs.append(GraphTripleConv(**gconv_kwargs))

    def forward(self, obj_vecs, pred_vecs, edges):
        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            obj_vecs, pred_vecs = gconv(obj_vecs, pred_vecs, edges)
        return obj_vecs, pred_vecs

    def init_parameters(self):
        for gc in self.gconvs:
            gc.apply(_init_weights)

def build_mlp(dim_list, activation='relu', batch_norm='none',
              dropout=0, final_nonlinearity=True):
    layers = []
    for i in range(len(dim_list) - 1):
        dim_in, dim_out = dim_list[i], dim_list[i + 1]
        layers.append(nn.Linear(dim_in, dim_out))
        final_layer = (i == len(dim_list) - 2)
        if not final_layer or final_nonlinearity:
            if batch_norm == 'batch':
                layers.append(nn.BatchNorm1d(dim_out))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leakyrelu':
                layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)

class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=True,
            scaled_cosine=False,
            scale_heads=False,
            logit_scale_max=math.log(1. / 0.01),
            attn_drop=0.,
            proj_drop=0.
    ):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.logit_scale_max = logit_scale_max

        self.in_proj_weight = nn.Parameter(torch.randn((dim * 3, dim)) * self.scale)
        if qkv_bias:
            self.in_proj_bias = nn.Parameter(torch.zeros(dim * 3))
        else:
            self.in_proj_bias = None

        if self.scaled_cosine:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        else:
            self.logit_scale = None
        self.attn_drop = nn.Dropout(attn_drop)
        if self.scale_heads:
            self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        L, N, C = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        k = k.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        v = v.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)

        if self.logit_scale is not None:
            attn = torch.bmm(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-1, -2))
            logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
            attn = attn.view(N, self.num_heads, L, L) * logit_scale
            attn = attn.view(-1, L, L)
        else:
            q = q * self.scale
            attn = torch.bmm(q, k.transpose(-1, -2))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask
            attn += attn_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.bmm(attn, v)
        if self.head_scale is not None:
            x = x.view(N, self.num_heads, L, C) * self.head_scale
            x = x.view(-1, L, C)
        x = x.transpose(0, 1).reshape(L, N, C)
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x

if __name__ == '__main__':
    main()