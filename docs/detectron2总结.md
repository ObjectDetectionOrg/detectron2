## 数据标注

labelimg

标注文件格式Pacal VOC



## 构建COCO风格的数据集

```
import xml.etree.ElementTree as ET
import os
import json
import numpy as np

coco = dict()
coco['images'] = []
coco['type'] = 'instances'
coco['annotations'] = []
coco['categories'] = []

category_set = dict()
image_set = set()

category_item_id = -1
image_id = 0
annotation_id = 0


def addCatItem(name):
    global category_item_id
    category_item = dict()
    category_item['supercategory'] = 'none'
    category_item_id += 1
    category_item['id'] = category_item_id
    category_item['name'] = name
    coco['categories'].append(category_item)
    category_set[name] = category_item_id
    return category_item_id


def addImgItem(file_name, size):
    global image_id
    if file_name is None:
        raise Exception('Could not find filename tag in xml file.')
    if size['width'] is None:
        raise Exception('Could not find width tag in xml file.')
    if size['height'] is None:
        raise Exception('Could not find height tag in xml file.')
    image_id += 1
    img_id = "%04d" % image_id
    image_item = dict()
    image_item['id'] = int(img_id)
    # image_item['id'] = image_id
    image_item['file_name'] = file_name
    image_item['width'] = size['width']
    image_item['height'] = size['height']
    coco['images'].append(image_item)
    image_set.add(file_name)
    return image_id


def addAnnoItem(object_name, image_id, category_id, bbox):
    global annotation_id
    annotation_item = dict()
    annotation_item['segmentation'] = []
    seg = []
    # bbox[] is x,y,w,h
    # left_top
    seg.append(bbox[0])
    seg.append(bbox[1])
    # left_bottom
    seg.append(bbox[0])
    seg.append(bbox[1] + bbox[3])
    # right_bottom
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1] + bbox[3])
    # right_top
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1])

    annotation_item['segmentation'].append(seg)

    annotation_item['area'] = bbox[2] * bbox[3]
    annotation_item['iscrowd'] = 0
    annotation_item['ignore'] = 0
    annotation_item['image_id'] = image_id
    annotation_item['bbox'] = bbox
    annotation_item['category_id'] = category_id
    annotation_id += 1
    annotation_item['id'] = annotation_id
    coco['annotations'].append(annotation_item)


def parseXmlFiles(xml_path):
    for f in os.listdir(xml_path):
        if not f.endswith('.xml'):
            continue

        bndbox = dict()
        size = dict()
        current_image_id = None
        current_category_id = None
        file_name = None
        size['width'] = None
        size['height'] = None
        size['depth'] = None

        xml_file = os.path.join(xml_path, f)
        # print(xml_file)

        tree = ET.parse(xml_file)
        root = tree.getroot()
        if root.tag != 'annotation':
            raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))

        # elem is <folder>, <filename>, <size>, <object>
        for elem in root:
            current_parent = elem.tag
            current_sub = None
            object_name = None

            if elem.tag == 'folder':
                continue

            if elem.tag == 'filename':
                file_name = elem.text
                if file_name in category_set:
                    raise Exception('file_name duplicated')

            # add img item only after parse <size> tag
            elif current_image_id is None and file_name is not None and size['width'] is not None:
                if file_name not in image_set:
                    current_image_id = addImgItem(file_name, size)
                    # print('add image with {} and {}'.format(file_name, size))
                else:
                    raise Exception('duplicated image: {}'.format(file_name))
                    # subelem is <width>, <height>, <depth>, <name>, <bndbox>
            for subelem in elem:
                bndbox['xmin'] = None
                bndbox['xmax'] = None
                bndbox['ymin'] = None
                bndbox['ymax'] = None

                current_sub = subelem.tag
                if current_parent == 'object' and subelem.tag == 'name':
                    object_name = subelem.text
                    if object_name not in category_set:
                        current_category_id = addCatItem(object_name)
                    else:
                        current_category_id = category_set[object_name]

                elif current_parent == 'size':
                    if size[subelem.tag] is not None:
                        raise Exception('xml structure broken at size tag.')
                    size[subelem.tag] = int(subelem.text)

                # option is <xmin>, <ymin>, <xmax>, <ymax>, when subelem is <bndbox>
                for option in subelem:
                    if current_sub == 'bndbox':
                        if bndbox[option.tag] is not None:
                            raise Exception('xml structure corrupted at bndbox tag.')
                        bndbox[option.tag] = int(option.text)

                # only after parse the <object> tag
                if bndbox['xmin'] is not None:
                    if object_name is None:
                        raise Exception('xml structure broken at bndbox tag')
                    if current_image_id is None:
                        raise Exception('xml structure broken at bndbox tag')
                    if current_category_id is None:
                        raise Exception('xml structure broken at bndbox tag')
                    bbox = []
                    # x
                    bbox.append(bndbox['xmin'])
                    # y
                    bbox.append(bndbox['ymin'])
                    # w
                    bbox.append(bndbox['xmax'] - bndbox['xmin'])
                    # h
                    bbox.append(bndbox['ymax'] - bndbox['ymin'])
                    # print('add annotation with {},{},{},{}'.format(object_name, current_image_id, current_category_id,
                    #                                                bbox))
                    addAnnoItem(object_name, current_image_id, current_category_id, bbox)


def save_to_classes_file(coco, filename="classes.txt"):
    categories = coco["categories"]
    categories.sort(key=lambda x: x["id"])
    with open(filename, "w") as f:
        for d in categories:
            f.write("{}\n".format(d["name"]))


if __name__ == '__main__':
	########## 修改部分############
    # 修改这里的两个地址，一个是xml文件的父目录；所有xml文件放置在这个目录下
    xml_dir = '/workdir/datasets/custom_dataset/train'
    # 一个是生成的json文件的绝对路径
    json_file = "/workdir/datasets/custom_dataset/train.json"
    classes_file = "/workdir/datasets/custom_dataset/train.txt"
	########## 修改部分############

    if not os.path.exists(os.path.dirname(json_file)):
        os.makedirs(os.path.dirname(json_file))

    parseXmlFiles(xml_dir)

    with open(json_file, 'w') as f:
        json.dump(coco, f)

    save_to_classes_file(coco, classes_file)
    data = np.loadtxt(classes_file, dtype="str")
    print("categories: {}".format(list(data)))
```

## 注册自定义的数据集

复制train_net.py为train_net-custom.py，在train_net-custom.py中加入以下代码

```
from detectron2.data.datasets import register_coco_instances

train_dataset_name = "custom_train"
train_json_file = "datasets/custom_dataset/train.json"
train_image_root = "datasets/custom_dataset/train"
register_coco_instances(train_dataset_name, {}, train_json_file, train_image_root)


test_dataset_name = "custom_test"
test_json_file = "datasets/custom_dataset/train.json"
test_image_root = "datasets/custom_dataset/train"
register_coco_instances(test_dataset_name, {}, test_json_file, test_image_root)
```



## 修改配置文件

retinanet_R_50_FPN_1x-custom.yaml

```
_BASE_: "../Base-RetinaNet.yaml"
MODEL:
  WEIGHTS: "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
  RESNETS:
    DEPTH: 50
  BACKBONE:
    FREEZE_AT: 5
  RETINANET:
    NUM_CLASSES: 3

INPUT:
  MIN_SIZE_TRAIN: (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800)
  MIN_SIZE_TEST: 800
  FORMAT: BGR

DATASETS:
  TRAIN: ('custom_train',)
  TEST: ('custom_test',)

SOLVER:
  STEPS: (10000, 20000)
  MAX_ITER: 30000
  IMS_PER_BATCH: 4
  BASE_LR: 0.001
  CHECKPOINT_PERIOD: 1000
  WARMUP_ITERS: 1000

DATALOADER:
  NUM_WORKERS: 4

OUTPUT_DIR: ./output/custom-det-retinanet
```



## 模型训练

train.sh

```
python tools/train_net-custom.py \
--num-gpus 1 \
--config-file configs/COCO-Detection/retinanet_R_50_FPN_1x-custom.yaml \
--dist-url='tcp://127.0.0.1:50156'
```



## 模型评估

eval.sh

```
python tools/train_net-custom.py \
--num-gpus 1 \
--config-file configs/COCO-Detection/retinanet_R_50_FPN_1x-custom.yaml \
--eval-only MODEL.WEIGHTS /workdir/detectron2/output/custom-det-retinanet/model_final.pth
--dist-url='tcp://127.0.0.1:50156'
```

**结果**

Evaluation results for bbox：

|   AP   |  AP50   |  AP75   | APs  | APm  |  APl   |
| :----: | :-----: | :-----: | :--: | :--: | :----: |
| 99.757 | 100.000 | 100.000 | nan  | nan  | 99.757 |

 Per-category bbox AP:

| category | AP     | category | AP     | category | AP      |
| :------- | :----- | :------- | :----- | :------- | :------ |
| custom  | 99.372 | customM | 99.900 | customN | 100.000 |



## 自带的推理demo

```
python demo/demo.py \
--config-file configs/COCO-Detection/retinanet_R_50_FPN_1x.yaml \
--input demo/input1.jpg demo/input2.jpg \
--opts MODEL.WEIGHTS demo/retinanet_R_50_FPN_1x.pkl
```

主要代码：

```
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

config_file = "/workdir/detectron2/configs/COCO-Detection/retinanet_R_50_FPN_1x-custom.yaml"
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
cfg.MODEL.WEIGHTS = "/workdir/detectron2/output/custom-det-retinanet/model_final.pth"


cfg.freeze()

im = cv2.imread("/workdir/detectron2/datasets/custom_dataset/train/snapshot20210712102123.jpg")
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
```



## 模型简单部署

### 模型文件

Base-RetinaNet.yaml

retinanet_R_50_FPN_1x-custom.yaml，修改为`_BASE_: "./Base-RetinaNet.yaml"`

model_final.pth

### 推理demo

```
import cv2
import os
import random
import time

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


class customDetector(object):
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
    config_file = "{}/model/retinanet_R_50_FPN_1x-custom.yaml".format(cur_dir)
    class_name = ["customL", "customM", 'customN']
    
    detector = customDetector(config_file, weight_file)
    # image_dir = os.path.join(cur_dir, "images")
    image_dir = "/workdir/datasets/origin/images/00008-select"

    for filename in os.listdir(image_dir):
        if "out" in filename:
            continue
        filename = os.path.join(image_dir, filename)
        print(filename)
        img = cv2.imread(filename)

        t = time.time()
        labels, scores, bboxes = detector.detect(img)
        t = time.time() - t
        print("time: {} ms".format(int(t * 1000)))

        for label, score, bbox in zip(labels, scores, bboxes):
            print(label, score, bbox)
            left = int(bbox[0])
            top = int(bbox[1])
            right = int(bbox[2])
            bottom = int(bbox[3])

            colors = [(0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
            positions = [(left, int((top + bottom)/2)), (left, bottom), (int((left + right)/2), top), (int((left + right)/2), int((top + bottom)/2))]

            if label == 0:
                color = (0, 0, 255)
                position = (left, top)
            else:
                color = random.choice(colors)    
                position = random.choice(positions)
                
            cv2.rectangle(img, (left, top), (right, bottom), color, 3)

            name = class_name[label]
            cv2.putText(img, "{:}:{:.2f}".format(name, score), position, \
             cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)

        out_filename = filename + ".out.jpg"
        print(out_filename)
        cv2.imwrite(out_filename, img)
         
```

