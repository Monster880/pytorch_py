import torch
import glob
import cv2
from PIL import Image
from torchvision import transforms
import numpy as np
from resnet import resnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = resnet()

net.load_state_dict(torch.load())