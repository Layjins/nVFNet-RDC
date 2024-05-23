
If you have any questions, please contact us jinxianglai@tencent.com
### 1. DOCKER
docker pull jinxianglai/clvsion-mmcv1.4.1

### 2. Pretrained Model
The pretrained model 'vfnet_r2_101_dcn_ms_2x_51.1.pth' is pretrained on COCO dtaset and is obtained in https://drive.google.com/file/d/1kCiEqAA_VQlhbiNuZ3HWGhBD1JvVpK0c/view provided by https://github.com/hyz-xmaster/VarifocalNet.

| Backbone     | Style     | DCN     | MS <br> train | Lr <br> schd |Inf time <br> (fps) | box AP <br> (val) | box AP <br> (test-dev) | &nbsp; &nbsp; Download  &nbsp; &nbsp;  |
|:------------:|:---------:|:-------:|:-------------:|:------------:|:------------------:|:-----------------:|:----------------------:|:--------------------------------------:|
| R2-101       | pytorch   | Y       | Y             | 2x           | 10.3               | 51.1              | 51.3                   | [model](https://drive.google.com/file/d/1kCiEqAA_VQlhbiNuZ3HWGhBD1JvVpK0c/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1BTwm-knCIT-kzkASjWNMfRWaAwI0ONmC/view?usp=sharing)|

Note that you should modify the path of the pretrained model '/youtu/fuxi-team2-2/persons/niceliu/CLVision2022/models/vfnet_r2_101_dcn_ms_2x_51.1.pth' in the code project.

### 3. Dataset and Annotations Setting
Modify all '/youtu/fuxi-team2-2/CLVision/track2_datasets' in the code project into the path of EgoObjects Dataset and Annotations.

### 4. Run Training
The root path for saving checkpoints is '/youtu/fuxi-team2-2/persons/jinxianglai/CLvision2022/checkpoints' in tools/dist_train_json.sh, and you should modify this root path for saving checkpoints. \
Then,

```bash
bash run.sh
```
### 5. Training Configs
We provide different configs due to the limitations of hardware and execution time. And their performance on val or test set will be reported as follows. All the results are trained on V100 hardware.

| Version     | config    | Backbone     |Fixed Stage| Params | image size | GPU Num. | GPU Memory | Time | AvgAP <br> (val) | AvgAP <br> (test) |
|:------------:|:---------:|:-------:|:-------------:|:-------------:|:------------:|:------------------:|:------------:|:------------------:|:-----------------:|:----------------------:|
| Submited_Best       | configs/clv-challenge/vfnet_r2_101_fpn_mdconv_1x_clv_json.py   | R2-101 | 1      | 62M   | (1433,900)          | 8      | -G | 15h     | 62.84               | 55.94              |
| CLV_1       | configs/clv-challenge/vfnet_r2_101_fpn_mdconv_1gpu_clv_json.py   | R2-101 | 2      | 62M   | (833,500)          | 1      | 14G | 20h     | 60.13               | -              |
| CLV_2       | configs/clv-challenge/vfnet_r2_101_fpn_mdconv_1gpu_clv_json_2.py   | R2-101 | 3      | 62M   | (833,500)          | 1      | 12G | 16h     | 59.54               | -              |
| CLV_3       | configs/clv-challenge/vfnet_r2_101_fpn_mdconv_1gpu_clv_json_3.py   | R2-101 | 3      | 62M   | (1033,600)          | 1      | 14G | 24h     | 59.96               | -              |

To use different trainning configs, you only need to replace the contents after Line 12 in run.sh with the following commands.
#### (1) Submited_Best version
```bash
config=configs/clv-challenge/vfnet_r2_101_fpn_mdconv_1x_clv_json.py
bash tools/dist_train_json.sh $config 8
```
#### (2) CLV_1 version
```bash
config=configs/clv-challenge/vfnet_r2_101_fpn_mdconv_1gpu_clv_json.py
bash tools/dist_train_json.sh $config 1
```
#### (3) CLV_2 version
```bash
config=configs/clv-challenge/vfnet_r2_101_fpn_mdconv_1gpu_clv_json_2.py
bash tools/dist_train_json.sh $config 1
```
#### (4) CLV_3 version
```bash
config=configs/clv-challenge/vfnet_r2_101_fpn_mdconv_1gpu_clv_json_3.py
bash tools/dist_train_json.sh $config 1
```

### 6. Run Testing
```bash
bash tools/dist_submit.sh work_dir GPUS
```
work_dir: you can find the detailed work_dir in the log of training. \
GPUS: number of used GPU. 

For example,
```bash
bash tools/dist_submit.sh /youtu/fuxi-team2-2/persons/jinxianglai/CLvision2022/checkpoints/vfnet_r2_101_fpn_mdconv_1x_clv_json.py_20220528_233423 8
```

### 7. Compress Testing Results
```bash
python3 tools/jsonComp.py work_dir
```
For example,
```bash
python3 tools/jsonComp.py /youtu/fuxi-team2-2/persons/jinxianglai/CLvision2022/checkpoints/vfnet_r2_101_fpn_mdconv_1x_clv_json.py_20220528_233423
```
