# python tools/train_net-custom.py \
# --num-gpus 1 \
# --config-file configs/COCO-Detection/retinanet_R_50_FPN_1x-vehicle.yaml \
# --eval-only MODEL.WEIGHTS /workdir/detectron2/output/vehicle-det-retinanet/model_final.pth \
# --dist-url='tcp://127.0.0.1:50156'


python tools/train_net-custom.py \
--num-gpus 1 \
--config-file configs/COCO-Detection/retinanet_R_50_FPN_1x-vehicle-finetune.yaml \
--eval-only MODEL.WEIGHTS /workdir/detectron2/output/vehicle-det-retinanet-finetune/model_0001999.pth