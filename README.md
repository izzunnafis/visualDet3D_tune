# Description:

This repo is a fork from [YoloStereo3D](https://github.com/Owen-Liuyuxuan/visualDet3D/) which is  the official implementation of 2021 *ICRA* paper [**YOLOStereo3D: A Step Back to 2D for Efficient Stereo 3D Detection**](https://arxiv.org/abs/2103.09422). This repo is experimental repo to asses, modify, or maybe improve the original architecture.

What have been done so far:
- Experiment with group-wise correlation concept based on [Group-wise Correlation Stereo Network](https://arxiv.org/abs/1903.04025). With this modification, the mAP performance is improved about 3-4% on KITTI Dataset 
- Modify the demo file to do inference on one image

# How to use
- Install CUDA Toolkit >11.2
- Install the library based on the requirement.txt file.
- Build the DCN:
  - ```bash
    cd "visualDet3D/networks/lib/ops/dcn/"
    ```
  - In the make.sh file, ensure that cuda toolkit path is correct
  - ```bash
    ./make.sh
    ```
- Edit the configuration file on the /config
  - Select between original Stereo3D or modified Stereo3DGWC
  - Copy to a python file ex:Stereo3D_example.py
  - Modify the path in the file to locate the dataset
  - Modify the other thing if needed

    **important paths to modify in config** :
    1. cfg.path.data_path: Path to KITTI training data. We expect calib, image_2, image_3, label_2 being the subfolder (directly unzipping the downloaded zips will be fine)
    2. cfg.path.test_path: Path to KITTI testing data.  We expect calib, image_2 being the subfolder.
    3. cfg.path.visualDet3D_path: Path to the "visualDet3D" directorty of the current repo
    4. cfg.path.project_path: Path to the workdirs of the projects (will have temp_outputs, log, checkpoints)


- Precompute the statistical data, about this you can refer to the original repo or the paper
  - ```bash
    ./launchers/det_precompute.sh config/$CONFIG_FILE.py train #$CONFIG_FILE is the python configuration file that is created before
    ./launchers/det_precompute.sh config/$CONFIG_FILE.py test
    ```
  - It will create *.npy and *.pkl file in the folder based on the config file (path.project_path = "/path/to/visualDet3D/result"). Specifically, $PROJECT_PATH/output/training or/and /validation

## For training purpose
- Create the disparity supervision (Please refer to the paper)
    ```bash
    ./launchers/disparity_precompute.sh config/$CONFIG_FILE.py $IsUsingPointCloud(False/True) #False if without pointcloud data
    ```

- Train the model
    ```bash
    ./launcher/train.sh  --config/$CONFIG_FILE.py 0 $experiment_name #Set the experiment name
    ```

- Check for the validation data
    ```bash
    ./launcher/eval.sh --config/$CONFIG_FILE.py 0 $CHECKPOINT_PATH validation/test #$CHECKPOINT_PATH is the result model after the training process
    ```

## For demo purpose
- Please ensure the .npy file is exist in the $PROJECT_PATH/output/training/. It can be downloaded [here](https://drive.google.com/drive/folders/15RZ8-PKkcsbXn7zs4xLOUHe-c-iGeagK?usp=sharing)
- Ensure that the trained model is located in ckpt folder. The pretrained model can be downloaded [here](https://drive.google.com/drive/folders/15RZ8-PKkcsbXn7zs4xLOUHe-c-iGeagK?usp=sharing)

### Inference on single image
  ```bash
  python3 demos/inference_img.py config/$CONFIG_FILE.py ckpt/$MODEL_NAME.pth
  ```
### Inference on a sequence of images
  - Prepare the image sequence, ensure that projection matrices is exist (Formatted as in KITTI dataset)
  - Adjust the file path in the code /demos/"inference_to_vid.py"
  - ```bash
    python3 demos/inference_img.py config/Stereo3D_example.py ckpt/$MODEL_NAME.pth
    ```

## Demo result


![res](https://github.com/user-attachments/assets/592c6286-fba2-403f-825a-ce09818315b3)




## Further Info and Bug Issues

1. Open issues on the repo if you meet troubles or find a bug or have some suggestions.


## Special Thanks

[YoloStereo3D](https://github.com/Owen-Liuyuxuan/visualDet3D/)
