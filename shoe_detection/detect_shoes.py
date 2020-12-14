from __future__ import division

from models import Darknet
# from models import *
from utils.utils import *
from utils.datasets import *
from utils.nms_footwear import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torchvision.transforms as transforms


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator



def detect_shoes(img, conf_thres=0.1, nms_thres=0.4, box_extension=0):
'''Detect shoes in an image.
Given an image, detect where the shoes are and output the bounding box coordinates,
class confidence scores and confidence score.
Input:
- img: image data from Image.open(img_path).
- conf_thres: confidence score threshold. Float.
- nms_thres: threshold for non maximum suppression.
Output:
- croppend images?
- bounding box coordinates
- confidence scores
'''
    model_def = 'config/yolov3-openimages.cfg'
    weights_path = 'config/yolov3-openimages.weights'
    class_path = 'config/oidv6.names'
    conf_thres = 0.1
    nms_thres = 0.4
    batch_size = 1
    n_cpu = 0
    img_size = 416


    # Extract image as PyTorch tensor
    img_original = transforms.ToTensor()(img)
    img_shape_original = img_original.shape.permute(1,2,0)
    # Pad to square resolution
    img, _ = pad_to_square(img, 0)
    # Resize
    img = resize(img, img_size)
    img = img.unsqueeze_(0)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = Darknet(model_def, img_size=img_size).to(device)
    model.load_darknet_weights(weights_path)

    model.eval()  # Set in evaluation mode
    classes = load_classes(class_path)  # Extracts class labels from file
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    input_imgs = Variable(img.type(Tensor))

    # Get detections
    with torch.no_grad():
        detections = model(input_imgs) # (B, A, )
        detections = non_max_suppression_for_footwear(detections, conf_thres, nms_thres)[0]

    if detections is not None:
        detections = rescale_boxes(detections, img_size, img_shape_original[:2])
        cropped_imgs = []
        bbox_coords = []
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            x1 = round(x1)
            y1 = round(y1)
            x2 = round(x2)
            y2 = round(y2)

            cropped_imgs.append(img_original[:, x1:x2, y1:y2])
            bbox_coords.append([x1,y1,x2,y2])
        return cropped_imgs, bbox_coords
    else:
        return None, None
        

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]


    # # Create plot
    # img = np.array(Image.open(img_path))
    # plt.figure()
    # fig, ax = plt.subplots(1)
    # ax.imshow(img)

    # # Draw bounding boxes and labels of detections
    # if detections is not None:
    #     # Rescale boxes to original image
    #     detections = rescale_boxes(detections, img_size, img.shape[:2])
    #     unique_labels = detections[:, -1].cpu().unique()
    #     n_cls_preds = len(unique_labels)
    #     bbox_colors = random.sample(colors, n_cls_preds)
    #     for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

    #         print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

    #         box_w = x2 - x1
    #         box_h = y2 - y1

    #         color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
    #         # Create a Rectangle patch
    #         bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
    #         # Add the bbox to the plot
    #         ax.add_patch(bbox)
    #         # Add label
    #         plt.text(
    #             x1,
    #             y1,
    #             s=classes[int(cls_pred)]+', %.2f'%conf.item(),
    #             color="white",
    #             verticalalignment="top",
    #             bbox={"color": color, "pad": 0},
    #         )

    # # Save generated image with detections
    # plt.axis("off") 