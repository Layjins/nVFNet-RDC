pip3 install timm

#### Track2: online benchmark ####
#config=configs/clv-challenge/faster_rcnn_r50_fpn_1x_clv.py
#config=configs/clv-challenge/vfnet_r50_fpn_1x_clv.py
#config=configs/clv-challenge/cascade_rcnn_r50_fpn_1x_clv.py
#config=configs/clv-challenge/gfl_r50_fpn_1x_clv.py
#config=configs/clv-challenge/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_clv.py
#config=configs/clv-challenge/atss_r50_fpn_1x_clv.py
#bash tools/dist_train.sh $config 8


#### Track2: offline benchmark with json ####
#config=configs/clv-challenge/faster_rcnn_r50_fpn_1x_clv_json.py
#config=configs/clv-challenge/vfnet_r50_fpn_1x_clv_json.py
#config=configs/clv-challenge/vfnet_cbr50_cbfpn_1x_clv_json.py
#config=configs/clv-challenge/vfnet_cbr50_cbpafpn_mdconv_1x_clv_json.py
#config=configs/clv-challenge/vfnet_r2_101_fpn_1x_clv_json.py
config=configs/clv-challenge/vfnet_r2_101_fpn_mdconv_1x_clv_json.py
#config=configs/clv-challenge/vfnet_r2_101_fpn_mdconv_1gpu_clv_json.py
#config=configs/clv-challenge/vfnet_swin_t_p4_fpn_1x_clv_json.py
#config=configs/clv-challenge/vfnet_r50_fpn_mdconv_1x_clv_json.py
#config=configs/clv-challenge/vfnet_r50_fpn_mdconv_1gpu_clv_json.py
bash tools/dist_train_json.sh $config 8


#### Track3: online benchmark ####
#config=configs/clv-challenge/vfnet_r2_101_fpn_mdconv_1x_clv_track3.py
#bash tools/dist_train_track3.sh $config 8


#### Track3: offline benchmark with json ####
#config=configs/clv-challenge/vfnet_r50_fpn_1x_clv_json_track3.py
#config=configs/clv-challenge/vfnet_r2_101_fpn_mdconv_1x_clv_json_track3.py
#config=configs/clv-challenge/vfnet_cbr50_cbpafpn_mdconv_1x_clv_json_track3.py
#bash tools/dist_train_json_track3.sh $config 8


#### mmdetection ####
#config=configs/vfnet/vfnet_swin_t_fpn_mstrain_2x_coco.py
#bash tools/dist_train_mmdet.sh $config 8
