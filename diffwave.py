from train_utils import *
from models.model import DiffWave
from models.params import params
from datasets.dataset import ConditionalDataset
import os

BASE_NUM_STEPS = 50
BASE_TIME_SCALE = 1


def make_model():
    net = DiffWave(params)
    net.image_size = [1, 3, 256, 256]
    net.sr = 22050
    return net


def make_dataset():
    print(os.path.isdir('/Users/kostiks/study/cs/diffusion/ljspeech/train'))
    return ConditionalDataset(paths="/Users/kostiks/study/cs/diffusion/ljspeech/train", params=params)
