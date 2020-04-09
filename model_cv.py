import numpy as np
import torch
import torchvision

from io import BytesIO
import base64
import pickle
from torchvision import transforms

import cv2
import PIL



model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval ()

def predict (img):

    normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )
    
    transform_img = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
#     normalize 
    ])

    y= [transform_img(PIL.Image.fromarray(img))]
    
    z = model(y)
    return display_person (np.array(y[0].permute(1, 2, 0)), z[0]['masks'][(z[0]['scores']>0.5)&(z[0]['labels']==1)].detach().numpy().sum (0)==0)

def display_person(img, masks):
    im = img.copy ()
    im[np.stack((masks[0,...],masks[0,...],masks[0,...])).transpose(1, 2, 0)] = 1
    return im