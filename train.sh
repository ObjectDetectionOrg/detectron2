python tools/train_net-custom.py \
--num-gpus 1 \
--config-file configs/COCO-Detection/retinanet_R_50_FPN_1x-vehicle.yaml \
--dist-url='tcp://127.0.0.1:50156'