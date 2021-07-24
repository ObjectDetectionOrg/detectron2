import cv2
import os

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


class VehicleDetector(object):
    def __init__(self, config_file, weight_file, confidence_threshold = 0.5):
        cfg = get_cfg()
        cfg.merge_from_file(config_file)
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set threshold for this model
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

        cfg.MODEL.DEVICE = "cuda:0"
        cfg.MODEL.WEIGHTS = weight_file
        cfg.freeze()

        self.predictor = DefaultPredictor(cfg)

    def detect(self, cvimg):
        outputs = self.predictor(cvimg)
        # numpy array: [label1, label2, ...]
        labels = outputs["instances"].pred_classes.to("cpu").numpy()
        scores = outputs["instances"].scores.to("cpu").numpy()
        # numpy array: [[left, top, right, bottom], [left, top, right, bottom], ...]
        bboxes = outputs["instances"].pred_boxes.to("cpu").tensor.numpy()
        return labels, scores, bboxes


if __name__ == "__main__":
    cur_dir = os.path.dirname(__file__)
    weight_file = "{}/model/model_final.pth".format(cur_dir)
    config_file = "{}/model/retinanet_R_50_FPN_1x-vehicle.yaml".format(cur_dir)
    class_name = ["vehicleL", "vehicleM", 'vehicleN']
    
    detector = VehicleDetector(config_file, weight_file)
    image_dir = os.path.join(cur_dir, "images")
    for filename in os.listdir(image_dir):
        if "out" in filename:
            continue
        filename = os.path.join(image_dir, filename)
        print(filename)
        img = cv2.imread(filename)
        labels, scores, bboxes = detector.detect(img)
        for label, score, bbox in zip(labels, scores, bboxes):
            print(label, score, bbox)
            left = int(bbox[0])
            top = int(bbox[1])
            right = int(bbox[2])
            bottom = int(bbox[3])
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)
            name = class_name[label]
            cv2.putText(img, "{:}:{:.2f}".format(name, score), (left, int((top + bottom)/2)), \
             cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

        out_filename = filename + ".out.jpg"
        print(out_filename)
        cv2.imwrite(out_filename, img)
            