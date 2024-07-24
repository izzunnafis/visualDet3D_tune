import os
import numpy as np
from easydict import EasyDict
from fire import Fire
from copy import deepcopy
import torch
import cv2

from _path_init import *
from visualDet3D.networks.utils.registry import DETECTOR_DICT
from visualDet3D.networks.utils.utils import BackProjection, BBox3dProjector
from visualDet3D.utils.utils import cfg_from_file, draw_3D_box
from visualDet3D.data.pipeline.augmentation_builder import build_augmentator

        
def draw_bbox_to_img (img, scores, bbox_2d, bbox_3d_corner_homo):
    res_image = denorm(img)
    if len(scores)>0:
        res_image = draw_bbox2d_to_image(res_image, bbox_2d.cpu().numpy())
        for box in bbox_3d_corner_homo:
            box = box.cpu().numpy().T
            res_image = draw_3D_box(res_image, box)

    return res_image


def draw_bbox2d_to_image(image, bboxes2d, color=(255, 0, 255)):
    drawed_image = image.copy()
    for box2d in bboxes2d:
        cv2.rectangle(drawed_image, (int(box2d[0]), int(box2d[1])), (int(box2d[2]), int(box2d[3])), color, 3)
    return drawed_image

def denorm(image):
    new_image = np.array((image * np.array([0.229, 0.224, 0.225]) +  np.array([0.485, 0.456, 0.406])) * 255, dtype=np.uint8)
    return new_image

def get_projection_matrix(file_path, camera_num):
    assert camera_num == 0 or 1, "camera num must be 0 (for left camera) or 1 (for right camera)"
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if camera_num == 0: #left cam
                if line.startswith("P2"): 
                    p = line.split(":")[1].strip()
                    break
            elif camera_num == 1: #right cam
                if line.startswith("P3"):
                    p = line.split(":")[1].strip()
                    break

    p = np.array(p.split(), dtype=np.float32).reshape(3, 4)
    return p

def get_file_list(path):

    filenames = next(os.walk(path), (None, None, []))[2]  # [] if no file
    filenames = sorted(filenames, key = last_4chars)

    return filenames

def last_4chars(x):
    return(x)


def main(config="config/config_file.py", checkpooint_path="path/to/model.ckpt"):
    """Main function for the training script.

    KeywordArgs:
        config (str): Path to config file.
    """

    ## Get config
    cfg = cfg_from_file(config)
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
    detector.eval().cuda()

    ## For visualization
    projector = BBox3dProjector().cuda()
    backprojector = BackProjection().cuda()



    file_dir = os.path.dirname(os.path.abspath(__file__))
    path_left = file_dir+"/../data/Kitti_tracking_data/testing/image_02/0011/" #Adjust this line according to the available data
    path_right = file_dir+"/../data/Kitti_tracking_data/testing/image_03/0011/" #Adjust this line according to the available data
    img_list = get_file_list(path_left)

    # Extract P2 and P3 from the txt file
    calib_path = file_dir+"/../data/Kitti_tracking_data/testing/calib/0011.txt" #Adjust this line according to the available data
    p2 = get_projection_matrix(calib_path, 0)
    p3 = get_projection_matrix(calib_path, 1)

    vidwriter = cv2.VideoWriter("res/res.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 10, (1280, 288)) #You can adjust the save folder path

    for i in range(len(img_list)):
        #img path
        img_l = cv2.imread(os.path.join(path_left, img_list[i]), cv2.IMREAD_COLOR) 
        img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)

        img_r = cv2.imread(os.path.join(path_right, img_list[i]), cv2.IMREAD_COLOR)
        img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)

        p2_tmp = p2.copy()
        p3_tmp = p3.copy()

        preprocess_data = build_augmentator(cfg.data.test_augmentation)
        img_l, img_r, p2_tmp, p3_tmp = preprocess_data(img_l, img_r, deepcopy(p2_tmp),deepcopy(p3_tmp))

        img_l_ts = torch.from_numpy(np.expand_dims(img_l.transpose([2,0,1]),axis=0))
        img_r_ts = torch.from_numpy(np.expand_dims(img_r.transpose([2,0,1]),axis=0))
        p2_ts = torch.from_numpy(np.expand_dims(p2_tmp,axis=0))
        p3_ts = torch.from_numpy(np.expand_dims(p3_tmp,axis=0))

        with torch.no_grad():
            scores, bboxes, cls_indexes = detector.test_forward(img_l_ts.cuda(), img_r_ts.cuda(), p2_ts.cuda().float(), p3_ts.cuda().float())

            bbox_2d = bboxes[:, 0:4]
            bbox_3d_state = bboxes[:, 4:] #[cx,cy,z,w,h,l,alpha]
            bbox_3d_state_3d = backprojector(bbox_3d_state, p2_ts[0]) #[x, y, z, w,h ,l, alpha]
            abs_bbox, bbox_3d_corner_homo, thetas = projector(bbox_3d_state_3d, p2_ts[0].cuda())

        res_img = draw_bbox_to_img(img_l, scores, bbox_2d, bbox_3d_corner_homo)
        res_img = cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)

        for obj_bbox in enumerate(bbox_3d_state_3d):
            print("obj {:d}, x_pos {:.2f}m, y_pos {:.2f}m, z_pos {:.2f}m".format(obj_bbox[0], obj_bbox[1][0].cpu().numpy(), obj_bbox[1][1].cpu().numpy(), obj_bbox[1][2].cpu().numpy()))

        print(i)
        vidwriter.write(res_img)


    vidwriter.release()


if __name__ == '__main__':
    Fire(main)
