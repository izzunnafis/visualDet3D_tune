import os
import sys
import numpy as np
from easydict import EasyDict
from tqdm import tqdm
from fire import Fire
import logging
import torch

from _path_init import *
from visualDet3D.networks.utils.registry import DETECTOR_DICT, DATASET_DICT, PIPELINE_DICT
from visualDet3D.networks.utils.utils import BackProjection, BBox3dProjector, get_num_parameters
from visualDet3D.utils.timer import Timer
from visualDet3D.utils.utils import LossLogger, cfg_from_file
from visualDet3D.networks.optimizers import optimizers, schedulers
from visualDet3D.data.dataloader import build_dataloader

"""
    Script for launching training process
"""
import os
import sys
import numpy as np
from easydict import EasyDict
from tqdm import tqdm
from fire import Fire
import logging
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from _path_init import *
from visualDet3D.networks.utils.registry import DETECTOR_DICT, DATASET_DICT, PIPELINE_DICT
from visualDet3D.networks.utils.utils import BackProjection, BBox3dProjector, get_num_parameters
from visualDet3D.evaluator.kitti.evaluate import evaluate
import visualDet3D.data.kitti.dataset
from visualDet3D.utils.timer import Timer
from visualDet3D.utils.utils import LossLogger, cfg_from_file
from visualDet3D.networks.optimizers import optimizers, schedulers
from visualDet3D.data.dataloader import build_dataloader

def main(config="config/config.py", checkpooint_path="model.ckpt"):
    """Main function for the training script.

    KeywordArgs:
        config (str): Path to config file.
    """

    ## Get config
    cfg = cfg_from_file(config)

    ## Collect distributed(or not) information
    cfg.dist = EasyDict()

    gpu = min(cfg.trainer.gpu, torch.cuda.device_count() - 1)
    torch.backends.cudnn.benchmark = getattr(cfg.trainer, 'cudnn', False)
    torch.cuda.set_device(gpu)
 
    ## Create the model
    detector = DETECTOR_DICT[cfg.detector.name](cfg.detector)

    ##Load model
    state_dict = torch.load(checkpooint_path, map_location='cuda:{}'.format(0))
    detector.load_state_dict(state_dict)

    ## Convert to cuda
    detector.eval()

    


if __name__ == '__main__':
    Fire(main)
