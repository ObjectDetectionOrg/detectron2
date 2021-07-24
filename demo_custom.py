import cv2

import matplotlib.pyplot as plt

import detectron2
# import some common detectron2 utilities
# from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library

config_file = "/workdir/detectron2/configs/COCO-Detection/retinanet_R_50_FPN_1x-vehicle.yaml"
cfg.merge_from_file(config_file)

confidence_threshold = 0.5
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
# cfg.MODEL.WEIGHTS = "demo/retinanet_R_50_FPN_1x.pkl"

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

cfg.MODEL.DEVICE = "cuda:0"
cfg.MODEL.WEIGHTS = "/workdir/detectron2/output/vehicle-det-retinanet/model_final.pth"


cfg.freeze()

im = cv2.imread("/workdir/detectron2/datasets/vehicle_dataset/train/snapshot20210712102123.jpg")
# im = cv2.resize(im, (0,0), fx=0.3, fy=0.3)
# plt.imshow(im)
# plt.show()

predictor = DefaultPredictor(cfg)
outputs = predictor(im)

print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)

# We can use `Visualizer` to draw the predictions on the image.
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

res = out.get_image()[:, :, ::-1]
res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(res)
plt.show()