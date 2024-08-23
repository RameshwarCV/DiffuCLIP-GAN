Official Implementation of "Layout Free scene graph to image generation", accepted at the British Machine Vision Conference (BMVC), 2024, Glasgow.

<img width="700" alt="image" src="https://github.com/user-attachments/assets/5649205b-72db-4a14-a368-1edf9afee914">

<img width="700" alt="image" src="https://github.com/user-attachments/assets/6a278d4e-cacb-41f2-bf4c-46539989074b">


## Environement
conda env create -f environment.yaml

conda activate environmentName

mkdir pretrained  (keep trained model for graph encoder here)

## Dataset
Scripts folder contains bash files to download and pre-process both Visual Genome and COCO-stuff dataset. 

For COCO, install COCO API from here https://github.com/cocodataset/cocoapi.
## Training
To train model on any device, run the following commands

**For visual genome**

accelerate launch SG_to_img_vg.py --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" --output_dir=./Output_vg_concat 
--resolution=256 --train_batch_size=2 --gradient_accumulation_steps=1 --learning_rate=5e-6 --lr_scheduler="constant" --lr_warmup_steps=0 --num_train_epochs=1

**For coco stuff**

accelerate launch SG_to_img_coco.py --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" --output_dir=./Output_coco_mmd 
--resolution=256 --train_batch_size=2 --gradient_accumulation_steps=1 --learning_rate=5e-6 --lr_scheduler="constant" --lr_warmup_steps=0 --num_train_epochs=1

## sampling

For sampling, use the following commands. It wiil create test results folder itself. Edit path to the trained diffusion network in sampler files.

**For visual genome**

python testing_sampler_vg.py

**For coco stuff**

python testing_sampler_coco.py
