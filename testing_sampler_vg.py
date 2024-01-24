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


def build_loaders():
    dset_kwargs = {
        'vocab_path': './datasets/vg/vocab.json',
        'h5_path': './datasets/vg/test.h5',
        'image_dir': './datasets/vg/images',
        'image_size': (256, 256),
        'max_samples': None,
        'max_objects': 30,
        'use_orphaned_objects': True,
        'include_relationships': True,
    }
    dset = VgSceneGraphDataset(**dset_kwargs)
    collate_fn = vg_collate_fn

    loader = DataLoader(dset, batch_size=1, num_workers=4, shuffle=False, collate_fn=collate_fn)
    return loader


def main():
    pretrained_model_path='./Output_vg_concat'
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
    vocab_file = './datasets/vg/vocab.json'
    with open(vocab_file, 'r') as f:
        vocab = json.load(f)

    loader = build_loaders()

    root_dir = './test_results_vg_concat'
    scene_graph_dir = os.path.join(root_dir, 'scene_graph')
    generate_img_dir = os.path.join(root_dir, 'img')

    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    if not os.path.exists(scene_graph_dir):
        os.mkdir(scene_graph_dir)
    if not os.path.exists(generate_img_dir):
        os.mkdir(generate_img_dir)

    n_samples_per_scene_graph = 1
    config = OmegaConf.load("./config_vg.yaml")
    gcn=CGIPModel(num_objs=179,num_preds=46,layers=5,width=512,embed_dim=512,ckpt_path="./Output_vg_concat/gcn.pt")
    gcn.cuda()
    with torch.no_grad():
            img_idx = -1
            for batch_data in loader:
                img_idx += 1
                imgs, objs, boxes, triples, obj_to_img, triple_to_img = [x.cuda() for x in batch_data]
                scene_graph_path = os.path.join(scene_graph_dir, str(img_idx)+'_graph.png')
                draw_scene_graph(objs, triples, vocab, output_filename=scene_graph_path,orientation='V')
#-----------------simple prompt--------------------------                
                obj_string=[]
               
                for obj in objs:
                    obj_string.append(vocab['object_idx_to_name'][obj])
                obj_string[len(obj_string)-1]='in_image'
                graph_prompt=",".join(obj_string)
                no_of_tokens=len(graph_prompt.split(" "))
                
#------------------relationship prompt---------------------
                 

                graph_info = [imgs, objs, None, triples, obj_to_img, triple_to_img]

                ## image is used for shape only, not using image at sampling time, check graph code to verify
                c_local, c_global=gcn.encode_graph_local_global(graph_info)
                c_global.cuda()

                # objs_list, triples_list = [], []
                # for i in range(objs.size(0)):
                #     objs_list.append(vocab['object_idx_to_name'][objs[i].item()])
                # for i in range(triples.size(0)):
                #     s = triples[i, 0].item()
                #     # p = vocab['pred_name_to_idx'][triples[i, 1].item()]
                #     p = triples[i, 1].item()
                #     o = triples[i, 2].item()
                #     triples_list.append([s, p, o])
                # objs, triples = objs_list, triples_list

                # prompt_triplet=[]
                # prompt_object=["A photo of"]
                # for s, p, o in triples:
                #     if(p==0):
                #         prompt_object.append(objs[s]+",")
                #         continue
                #     prompt_triplet.append(objs[s])
                #     prompt_triplet.append(vocab['pred_idx_to_name'][p])
                #     prompt_triplet.append(objs[o])
                #     prompt_triplet.append("and,")

                # if(len(prompt_triplet)>0):
                #     prompt_triplet.pop()
                # triplet_prompt=" ".join(prompt_triplet)
                # object_prompt=" ".join(prompt_object)
                # graph_prompt="".join([object_prompt,triplet_prompt])

                height = 256                        # default height of Stable Diffusion
                width = 256                         # default width of Stable Diffusion

                num_inference_steps = 100            # Number of denoising steps
                scheduler.set_timesteps(num_inference_steps)
                guidance_scale = 7.5                # Scale for classifier-free guidance
                #earlier used 11
                generator = torch.manual_seed(11)   # Seed generator to create the inital latent noise

                batch_size = 1
                graph_encoding_tensor=interpolate(c_global.unsqueeze(1),(768))
                text_input = tokenizer(graph_prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
                text_embeddings = text_encoder(text_input.input_ids.to('cuda'))[0]
                
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
                #text_embeddings=textGraphModel(text_embeddings,c_global).cuda()
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
                #uncond_embeddings=torch.cat((uncond_embeddings,graph_encoding_tensor),dim=1).cuda()
                #uncond_embeddings=textGraphModel(uncond_embeddings,c_global)
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
                if(img_idx%100==0):
                    print("total generated image",img_idx)
                
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

class VgSceneGraphDataset(Dataset):
    def __init__(self, vocab_path, h5_path, image_dir, image_size=(256, 256), max_objects=10,
                 max_samples=None,
                 include_relationships=True, use_orphaned_objects=True):
        super(VgSceneGraphDataset, self).__init__()
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        self.image_dir = image_dir
        self.image_size = image_size
        self.vocab = vocab
        self.num_objects = len(vocab['object_idx_to_name'])
        self.use_orphaned_objects = use_orphaned_objects
        self.max_objects = max_objects
        self.max_samples = max_samples
        self.include_relationships = include_relationships

        transform = [Resize(image_size), transforms.ToTensor()]  # augmentation
        self.transform = transforms.Compose(transform)

        self.data = {}

        with h5py.File(h5_path, 'r') as f:
            for k, v in f.items():
                if k == 'image_paths':
                    self.image_paths = list(v)
                else:
                    self.data[k] = torch.IntTensor(np.asarray(v))

    def __len__(self):
        num = self.data['object_names'].size(0)
        if self.max_samples is not None:
            return min(self.max_samples, num)
        return num

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, str(self.image_paths[index], encoding="utf-8"))

        with open(img_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                WW, HH = image.size
                image = self.transform(image.convert('RGB'))

        H, W = self.image_size

        obj_idxs_with_rels = set()
        obj_idxs_without_rels = set(range(self.data['objects_per_image'][index].item()))
        for r_idx in range(self.data['relationships_per_image'][index]):
            s = self.data['relationship_subjects'][index, r_idx].item()
            o = self.data['relationship_objects'][index, r_idx].item()
            obj_idxs_with_rels.add(s)
            obj_idxs_with_rels.add(o)
            obj_idxs_without_rels.discard(s)
            obj_idxs_without_rels.discard(o)

        obj_idxs = list(obj_idxs_with_rels)
        obj_idxs_without_rels = list(obj_idxs_without_rels)
        if len(obj_idxs) > self.max_objects - 1:
            obj_idxs = random.sample(obj_idxs, self.max_objects)
        if len(obj_idxs) < self.max_objects - 1 and self.use_orphaned_objects:
            num_to_add = self.max_objects - 1 - len(obj_idxs)
            num_to_add = min(num_to_add, len(obj_idxs_without_rels))
            obj_idxs += random.sample(obj_idxs_without_rels, num_to_add)
        O = len(obj_idxs) + 1

        objs = torch.LongTensor(O).fill_(-1)

        boxes = torch.FloatTensor([[0, 0, 1, 1]]).repeat(O, 1)
        obj_idx_mapping = {}
        for i, obj_idx in enumerate(obj_idxs):
            objs[i] = self.data['object_names'][index, obj_idx].item()
            x, y, w, h = self.data['object_boxes'][index, obj_idx].tolist()
            x0 = float(x) / WW
            y0 = float(y) / HH
            x1 = float(x + w) / WW
            y1 = float(y + h) / HH
            boxes[i] = torch.FloatTensor([x0, y0, x1, y1])
            obj_idx_mapping[obj_idx] = i

        objs[O - 1] = self.vocab['object_name_to_idx']['__image__']

        triples = []
        for r_idx in range(self.data['relationships_per_image'][index].item()):
            if not self.include_relationships:
                break

            s = self.data['relationship_subjects'][index, r_idx].item()
            p = self.data['relationship_predicates'][index, r_idx].item()
            o = self.data['relationship_objects'][index, r_idx].item()
            s = obj_idx_mapping.get(s, None)
            o = obj_idx_mapping.get(o, None)
            if s is not None and o is not None:
                triples.append([s, p, o])

        # Add dummy __in_image__ relationships for all objects
        in_image = self.vocab['pred_name_to_idx']['__in_image__']
        for i in range(O - 1):
            triples.append([i, in_image, O - 1])

        triples = torch.LongTensor(triples)
        return image, objs, boxes, triples

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

def vg_collate_fn(batch):
    all_imgs, all_objs, all_boxes, all_triples = [], [], [], []
    all_obj_to_img, all_triple_to_img = [], []
    obj_offset = 0
    for i, (img, objs, boxes, triples) in enumerate(batch):
        all_imgs.append(img[None])
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

    all_imgs = torch.cat(all_imgs)
    all_objs = torch.cat(all_objs)
    all_boxes = torch.cat(all_boxes)
    all_triples = torch.cat(all_triples)
    all_obj_to_img = torch.cat(all_obj_to_img)
    all_triple_to_img = torch.cat(all_triple_to_img)

    out = (all_imgs, all_objs, all_boxes, all_triples, all_obj_to_img, all_triple_to_img)
    return out

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

class CrossAttentionModel(nn.Module):
    def __init__(self):
        super(CrossAttentionModel, self).__init__()

        self.input_A_layer = nn.Linear(768, 768)
        self.input_B_layer = nn.Linear(512, 768)
        self.cross_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8)
        self.output_layer = nn.Linear(768, 768)

    def forward(self, input_A, input_B):
        # Process Input A
        input_A = self.input_A_layer(input_A)  # [1, 77, 768]
        #print(input_B.shape)
        input_B = self.input_B_layer(input_B.unsqueeze(0))  # [1, 77, 768]

        input_A = input_A.permute(1, 0, 2)  # [77, 1, 768]
        input_B = input_B.permute(1, 0, 2)  # [77, 1, 768]

        cross_attention_output, _ = self.cross_attention(input_A, input_B, input_B)  # Output: [77, 1, 768]

        cross_attention_output = cross_attention_output.permute(1, 0, 2)  # [1, 77, 768]

        output = self.output_layer(cross_attention_output)

        return output  # Output: [1, 77, 768]

if __name__ == '__main__':
    main()