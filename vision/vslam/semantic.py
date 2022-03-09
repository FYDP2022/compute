from enum import IntEnum, unique
from typing import Any
import cv2 as cv
import numpy as np
import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
use_cuda = torch.cuda.is_available()

from vslam.config import CONFIG

@unique
class Material(IntEnum):
  GRASS = 0
  PERSON = 1
  UNDEFINED = 2

class Net(nn.Module):
  def __init__(self, criterion=None):
      super(Net, self).__init__()
      self.criterion = criterion        
      self.mobilenet_v2 = models.mobilenet_v2(pretrained=True)
      self.conv1 = nn.Conv2d(1280, 512, 3, 1)
      self.conv2 = nn.Conv2d(512, 3, 3, 1)
      self.bn1 = nn.BatchNorm2d(512)
      self.bn2 = nn.BatchNorm2d(3)
      self.dropout1 = nn.Dropout2d()
  def forward(self, inp, gts=None): 
      output_size = (inp.size()[2],inp.size()[3])
      for idx, layer in enumerate(self.mobilenet_v2.features):
          if idx == 0:
              x = layer(inp)
          else:
              x = layer(x)
      x = self.conv1(x)
      x = self.bn1(x)
      x = F.relu(x)
      x = self.dropout1(x)
      x = F.interpolate(x, size=(output_size[0] + 2, output_size[1] + 2))
      x = self.conv2(x)
      lfinal = self.bn2(x)

      if self.training:
          return self.criterion(lfinal, gts)
      else:
          return lfinal

class SemanticSegmenation:
  def __init__(self):
    pass

  def get_semantic_image(self, image_processed):
    return self.get_forwarded_image(image_processed.cuda())
    
  def process_inp_image(self, image):
    colored_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    torch_img = torch.from_numpy(colored_img).float()
    return torch.permute(torch_img, (2, 0, 1))

  def get_forwarded_image(self, net, image):
    with torch.no_grad():
      nn_seg_output = net.forward(image[None])
    return torch.argmax(nn_seg_output, dim=1).cpu().numpy()[0]
    