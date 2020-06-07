#!/usr/bin/env python
# coding: utf-8

# ## install required modules

# ## set paths

# In[242]:


import os
import shutil
import json

import random
import copy
import torch
import cv2
import os
import json
import time
import numpy as np
from PIL import Image

from mmcv import Config
import mmcv
from mmdet.apis import inference_detector

from mmcv.runner import init_dist

from mmdet.datasets import build_dataset
from mmdet.models import build_detector

from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger

from pycocotools import mask as maskUtils

def main(cfg, seed):
    print('start')
    distributed = False
    validate = None
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))
    logger.info('MMDetection Version: {}'.format(__version__))
    logger.info('Config: {}'.format(cfg.text))

    # set random seeds
    logger.info('Set random seed to {}'.format(seed))
    set_random_seed(seed)
    cfg.seed = seed
    meta = dict()
    meta['seed'] = seed
    
    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        datasets.append(build_dataset(cfg.data.val))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.text,
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=validate,
        timestamp=timestamp,
        meta=meta)
    
def generate_rand_color():
    cl = np.random.randint(0, 255, size=(3))
    return (int(cl[0]), int(cl[1]), int(cl[2]))

def visualize_coco_img(coco, img_id, img_dir):
    img_name = [img['file_name'] for img in coco['images'] if img['id'] == img_id][0]
    path_to_img = os.path.join(img_dir, img_name)
    print(path_to_img)
    img = cv2.imread(path_to_img)

    anns = [ann for ann in coco['annotations'] if ann['image_id'] == img_id] #ann[image_id] = ann[id]
    
    cat_colors = {cat['id'] : generate_rand_color() for cat in coco['categories']}
    # plot masks
    for ann in anns:
        segm = ann['segmentation']
        cnt = np.array(segm).reshape(-1,1,2).astype(np.int32)
        cat_id = ann['category_id']
        cat_name = [cat['name'] for cat in coco['categories'] if cat['id'] == cat_id] 
        print(cat_id, cat_name)
        cv2.drawContours(img, [cnt], -1, cat_colors[cat_id], 4)
    return img
   
# In source code in place of image_id = category_id


# In[243]:


log_path = os.path.join('/home/data', "coco_file.json")
coco_file = open(log_path)
coco = json.load(coco_file)
#print(coco)
print(coco["annotations"][1])
#classes_names = [category["image_id"] for category in coco["annotations"]]
#print(classes_names)


# In[208]:


#log_path = os.path.join('/home/data', "test_coco_2.json")
#coco_file = open(log_path)
#coco_1 = json.load(coco_file)
#print(coco_1["annotations"][1])
#classes_names = [category["image_id"] for category in coco["annotations"]]
#print(classes_names)
# catagory id present


# In[251]:


mmdet_dir = '/home'
data_dir = os.path.join(mmdet_dir, 'data')
img_dir = os.path.join(data_dir, 'Img')


# In[188]:


#os.remove('/home/data/Test_coco.json')


# In[332]:


#source = '/home/data/log_file.json'
#to = '/root'
#shutil.move(source, to)


# In[256]:


coco_name = 'coco_file.json'
path_to_coco = os.path.join(data_dir, coco_name)
    
with open(path_to_coco, 'r') as f:
    coco = json.load(f)
    
# check for presence of images mentioned in coco in images folder
imgs = [file for file in os.listdir(img_dir)]
files = [img['file_name'] for img in coco['images']]
for file in files:
    if file not in imgs:
        print('file {} not found in images dir'.format(file))
        
# fix issue with image_id as a string
for ann in coco['annotations']:
    ann['image_id'] = int(ann['image_id'])
    
# fix issue with excessive coco categories
coco['categories'] = coco['categories'][:2]

# renumber ids to start from 1
for ann in coco['annotations']:
    ann['id'] += 1
    ann['image_id'] +=1
    ann['segmentation'] = [ann['segmentation']]
    
for img in coco['images']:
    img['id'] += 1


# In[283]:


num_imgs = len(coco['images'])
ind = np.random.randint(num_imgs) + 1
img = visualize_coco_img(coco, 133, img_dir)
pil = Image.fromarray(img)
pil


# ## save fixed coco

# In[258]:


coco_name = 'coco_file.json'
path_to_fixed_coco = os.path.join(data_dir, 'fixed' + coco_name)
with open(path_to_fixed_coco, 'w') as f:
    json.dump(coco, f)


# ## register dataset

# In[259]:


from mmdet.datasets.coco import CocoDataset
from mmdet.datasets.builder import DATASETS

coco_name = 'coco_file.json'
path_to_fixed_coco = os.path.join(data_dir, 'fixed' + coco_name)

with open(path_to_fixed_coco, 'r') as f:
    coco = json.load(f)
    
# register dataset
clses = tuple(cat['name'] for cat in coco['categories'])
@DATASETS.register_module
class user(CocoDataset):
    CLASSES = clses 

print('label classes are {}'.format(clses))
print('{} images found in {}'.format(len(coco['images']), coco_name))
print('{} annotations found in {}'.format(len(coco['annotations']), coco_name))


# In[260]:


import mmcv
from mmcv import Config

cfg_file = '/home/configs/cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_20e_coco.py' 
cfg = Config.fromfile(cfg_file)
coco_name = 'coco_file.json'
path_to_fixed_coco = os.path.join(data_dir, 'fixed' + coco_name)


# In[261]:


test_coco = 'test_coco_2.json'
path_to_train_coco = os.path.join(data_dir, 'fixed' + test_coco)


# In[263]:


# modify config for training on a single image
cfg.work_dir = data_dir

cfg.data.train.type = 'user'
cfg.data.train.ann_file = path_to_fixed_coco
cfg.data.train.img_prefix = '/home/data/Img'

cfg.data.val.type = 'user'
cfg.data.val.ann_file = 'path_to_train_coco '
cfg.data.val.img_prefix = '/home/data/Train_Image'

cfg.data.val.type = ''
cfg.data.val.ann_file = ''
cfg.data.val.img_prefix = ''

num_gpus = 1
cfg.gpu_ids = range(num_gpus)
cfg.data.imgs_per_gpu = 5 #1
cfg.data.workers_per_gpu = 2
batch_size = cfg.data.imgs_per_gpu * num_gpus
cfg.log_config.interval = 1

cfg.total_epochs = 500
cfg.checkpoint_config.interval = cfg.total_epochs
cfg.lr_config.warmup_iters = 3
cfg.lr_config.step = [350, 450]
base_lr = 0.001 / 8  # base learning rate corresponds to that used for training here 
                                # https://github.com/open-mmlab/mmdetection/blob/master/docs/GETTING_STARTED.md
                                # see also linear scaling rule https://arxiv.org/abs/1706.02677
cfg.optimizer.lr = base_lr * batch_size
print('batch size {}, start lr {}'.format(batch_size, cfg.optimizer.lr))

# rescale train image top actual size
#width = coco['images'][0]['width']
#height = coco['images'][0]['height']
scale = (1280, 720) #(width // 2, height //2)
assert max(scale) <= 2**12, 'train image size should be less {}'.format(2**12)
cfg.train_pipeline[2].img_scale = scale
cfg.data.train.pipeline[2].img_scale = scale
print('train image size {}'.format(scale))


# In[264]:


seed = 41 
main(cfg, seed)


# In[2]:


from mmdet.models import build_detector
from mmdet.apis import inference_detector, init_detector

config_fname = '/home/configs/cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_20e_coco.py'
checkpoint_file = '/home/data/epoch_500.pth'


model = init_detector(config_fname, checkpoint_file)


# In[3]:


def show_result(img,
                result,
                class_names,
                score_thr=0.3,
                wait_time=0,
                out_file=None):
    """Visualize the detection results on the image.
    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        wait_time (int): Value of waitKey param.
        show (bool, optional): Whether to show the image with opencv or not.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.
    Returns:
        np.ndarray or None: If neither `show` nor `out_file` is specified, the
            visualized image is returned, otherwise None is returned.
    """
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    img = img.copy()
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    # draw segmentation masks
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for i in inds:
            color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            mask = segms[i]
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    mmcv.imshow_det_bboxes(
        img,
        bboxes,
        labels,
        class_names=class_names,
        score_thr=score_thr,
        show=False,
        wait_time=wait_time,
        out_file=out_file)
    if not out_file:
        return img

def get_bboxes(bbox_result, clses):
    bboxes = {}
    for res, cls in zip(bbox_result, clses):
        if res.shape[0] > 0:
            cls_bboxes = [list(bbox[:4].astype(np.int32)) + [bbox[4]] for bbox in res]
            bboxes[cls] = cls_bboxes
            
    return bboxes
        
def show_bboxes(img, bboxes, color):
    
    fig,ax = plt.subplots(figsize=(20, 10))
    # Display the image
    ax.imshow(img)
    
    for bbox in bboxes:
        left = bbox[0]
        top = bbox[1]
        right = bbox[2]
        down = bbox[3]
        width = right - left
        height = down - top

        # Create a Rectangle patch
        rect = patches.Rectangle((left,top),width,height,linewidth=1,edgecolor='r',facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()


# In[4]:


img = '/root/Check_6.jpeg'
img = mmcv.imread(img)
img = img.copy()

result = inference_detector(model, img)


img_masked = show_result(img,
                result,
                model.CLASSES,
                score_thr=0.90,
                wait_time=0,
                out_file=None)

pil = Image.fromarray(img_masked)
pil


# In[159]:


#log_path = os.path.join('/home/data', "coco_file.json")
#coco_file = open(log_path)
#coco_2 = json.load(coco_file)
#print(coco_2["annotations"][0])
#classes_names = [category["id"] for category in coco["annotations"]]
#print(log)


# In[288]:


def plot_curve(log_dicts, args):
    if args.backend is not None:
        plt.switch_backend(args.backend)
    sns.set_style(args.style)
    # if legend is None, use {filename}_{key} as legend
    legend = args.legend
    if legend is None:
        legend = []
        for json_log in args.json_logs:
            for metric in args.keys:
                legend.append(f'{json_log}_{metric}')
    assert len(legend) == (len(args.json_logs) * len(args.keys))
    metrics = args.keys

    num_metrics = len(metrics)
    for i, log_dict in enumerate(log_dicts):
        epochs = list(log_dict.keys())
        for j, metric in enumerate(metrics):
            print(f'plot curve of {args.json_logs[i]}, metric is {metric}')
            if metric not in log_dict[epochs[0]]:
                raise KeyError(
                    f'{args.json_logs[i]} does not contain metric {metric}')

            if 'mAP' in metric:
                xs = np.arange(1, max(epochs) + 1)
                ys = []
                for epoch in epochs:
                    ys += log_dict[epoch][metric]
                ax = plt.gca()
                ax.set_xticks(xs)
                plt.xlabel('epoch')
                plt.plot(xs, ys, label=legend[i * num_metrics + j], marker='o')
            else:
                xs = []
                ys = []
                num_iters_per_epoch = log_dict[epochs[0]]['iter'][-1]
                for epoch in epochs:
                    iters = log_dict[epoch]['iter']
                    if log_dict[epoch]['mode'][-1] == 'val':
                        iters = iters[:-1]
                    xs.append(
                        np.array(iters) + (epoch - 1) * num_iters_per_epoch)
                    ys.append(np.array(log_dict[epoch][metric][:len(iters)]))
                xs = np.concatenate(xs)
                ys = np.concatenate(ys)
                plt.xlabel('iter')
                plt.plot(
                    xs, ys, label=legend[i * num_metrics + j], linewidth=0.5)
            plt.legend()
        if args.title is not None:
            plt.title(args.title)
    if args.out is None:
        plt.show()
    else:
        print(f'save curve to: {args.out}')
        plt.savefig(args.out)
        plt.cla()


# In[ ]:




