rlaunch --max-wait-duration=8h0m0s  --cpu=6 --gpu=4 --memory=204800 -- ./tools/dist_train.sh configs/baselines/faster_rcnn_r50_fpn_1x_lvis.py 4

rlaunch --max-wait-duration=8h0m0s  --cpu=6 --gpu=4 --memory=102400 -- ./tools/dist_train.sh configs/baselines/faster_rcnn_r50_fpn_1x_lvis.py 4