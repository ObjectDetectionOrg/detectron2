#!/usr/bin/env python

import argparse
import collections
import datetime
import glob
import json
import os
import os.path as osp
import sys
import uuid

import imgviz
import numpy as np

import labelme

try:
    import pycocotools.mask
except ImportError:
    print("Please install pycocotools:\n\n    pip install pycocotools\n")
    sys.exit(1)


def object_detection_main(args):
    """
    annotation{
"id": int, "image_id": int, "category_id": int, "segmentation": RLE or [polygon], "area": float, "bbox": [x,y,width,height], "iscrowd": 0 or 1,
}

categories[{
"id": int, "name": str, "supercategory": str,
}]
    :param args:
    :return:
    """
    now = datetime.datetime.now()

    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
        licenses=[dict(url=None, id=0, name=None,)],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        anno_type=args.anno_type, # coco之外的字段
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )

    # 将labels.txt中的class_name映射到class_id
    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        class_names.append(class_name)
        class_name_to_id[class_name] = class_id
        data["categories"].append(
            dict(supercategory=None, id=class_id, name=class_name,)
        )
    class_names = tuple(class_names)
    print("class_names:", class_names)
    out_class_names_file = osp.join(args.output_dir, "class_names.txt")
    with open(out_class_names_file, "w") as f:
        f.writelines("\n".join(class_names))
    print("Saved class_names:", out_class_names_file)

    # 获取输入目录下的json文件
    out_ann_file = osp.join(args.output_dir, "annotations.json")
    label_files = glob.glob(osp.join(args.input_dir, "*.json"))
    for image_id, filename in enumerate(label_files):
        print("Generating dataset from:", filename)

        # 读取json文件
        # 从图像文件路径或者从json文件中的imageData字段获取图像，并保存到imageData
        label_file = labelme.LabelFile(filename=filename)

        base = osp.splitext(osp.basename(filename))[0]
        out_img_file = osp.join(args.output_dir, "JPEGImages", base + ".jpg")

        img = labelme.utils.img_data_to_arr(label_file.imageData)
        imgviz.io.imsave(out_img_file, img)
        data["images"].append(
            dict(
                license=0,
                url=None,
                file_name=osp.relpath(out_img_file, osp.dirname(out_ann_file)),
                height=img.shape[0],
                width=img.shape[1],
                date_captured=None,
                id=image_id,
            )
        )

        bboxes = []
        labels = []
        for shape in label_file.shapes:
            if shape["shape_type"] != "rectangle":
                print(
                    "Skipping shape: label={label}, "
                    "shape_type={shape_type}".format(**shape)
                )
                continue

            class_name = shape["label"]
            class_id = class_name_to_id[class_name]

            (xmin, ymin), (xmax, ymax) = shape["points"]
            # swap if min is larger than max.
            xmin, xmax = sorted([xmin, xmax])
            ymin, ymax = sorted([ymin, ymax])

            # 用于显示结果
            bboxes.append([ymin, xmin, ymax, xmax])
            labels.append(class_id)

            # 保存信息到标注文件
            # bbox: x, y, w, h
            x = xmin
            y = ymin
            w = xmax - xmin
            h = ymax - ymin
            bbox = [x, y, w, h]
            area = w * h
            segmentation = [[xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]]

            data["annotations"].append(
                dict(
                    id=len(data["annotations"]),
                    image_id=image_id,
                    category_id=class_id,
                    segmentation=segmentation,
                    area=area,
                    bbox=bbox,
                    iscrowd=0,
                )
            )

        if not args.noviz:
            captions = [class_names[label] for label in labels]
            viz = imgviz.instances2rgb(
                image=img,
                labels=labels,
                bboxes=bboxes,
                captions=captions,
                font_size=15,
            )
            out_viz_file = osp.join(
                args.output_dir, "Visualization", base + ".jpg"
            )
            imgviz.io.imsave(out_viz_file, viz)

    with open(out_ann_file, "w") as f:
        json.dump(data, f)


def instance_segmentation_main(args):
    now = datetime.datetime.now()

    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
        licenses=[dict(url=None, id=0, name=None,)],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type="instances",
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )

    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        class_name_to_id[class_name] = class_id
        data["categories"].append(
            dict(supercategory=None, id=class_id, name=class_name,)
        )

    out_ann_file = osp.join(args.output_dir, "annotations.json")
    label_files = glob.glob(osp.join(args.input_dir, "*.json"))
    for image_id, filename in enumerate(label_files):
        print("Generating dataset from:", filename)

        label_file = labelme.LabelFile(filename=filename)

        base = osp.splitext(osp.basename(filename))[0]
        out_img_file = osp.join(args.output_dir, "JPEGImages", base + ".jpg")

        img = labelme.utils.img_data_to_arr(label_file.imageData)
        imgviz.io.imsave(out_img_file, img)
        data["images"].append(
            dict(
                license=0,
                url=None,
                file_name=osp.relpath(out_img_file, osp.dirname(out_ann_file)),
                height=img.shape[0],
                width=img.shape[1],
                date_captured=None,
                id=image_id,
            )
        )

        masks = {}  # for area
        segmentations = collections.defaultdict(list)  # for segmentation
        for shape in label_file.shapes:
            if shape["shape_type"] != "polygon":
                print(
                    "Skipping shape: label={label}, "
                    "shape_type={shape_type}".format(**shape)
                )
                continue

            points = shape["points"]
            label = shape["label"]
            group_id = shape.get("group_id")
            shape_type = shape.get("shape_type", "polygon")
            mask = labelme.utils.shape_to_mask(
                img.shape[:2], points, shape_type
            )

            if group_id is None:
                group_id = uuid.uuid1()

            instance = (label, group_id)

            if instance in masks:
                masks[instance] = masks[instance] | mask
            else:
                masks[instance] = mask

            # if shape_type == "rectangle":
            #     (x1, y1), (x2, y2) = points
            #     x1, x2 = sorted([x1, x2])
            #     y1, y2 = sorted([y1, y2])
            #     points = [x1, y1, x2, y1, x2, y2, x1, y2]
            # else:
            #     points = np.asarray(points).flatten().tolist()

            points = np.asarray(points).flatten().tolist()

            segmentations[instance].append(points)
        segmentations = dict(segmentations)

        for instance, mask in masks.items():
            cls_name, group_id = instance
            if cls_name not in class_name_to_id:
                continue
            cls_id = class_name_to_id[cls_name]

            mask = np.asfortranarray(mask.astype(np.uint8))
            mask = pycocotools.mask.encode(mask)
            area = float(pycocotools.mask.area(mask))
            bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

            data["annotations"].append(
                dict(
                    id=len(data["annotations"]),
                    image_id=image_id,
                    category_id=cls_id,
                    segmentation=segmentations[instance],
                    area=area,
                    bbox=bbox,
                    iscrowd=0,
                )
            )

        if not args.noviz:
            labels, captions, masks = zip(
                *[
                    (class_name_to_id[cnm], cnm, msk)
                    for (cnm, gid), msk in masks.items()
                    if cnm in class_name_to_id
                ]
            )
            viz = imgviz.instances2rgb(
                image=img,
                labels=labels,
                masks=masks,
                captions=captions,
                font_size=15,
                line_width=2,
            )
            out_viz_file = osp.join(
                args.output_dir, "Visualization", base + ".jpg"
            )
            imgviz.io.imsave(out_viz_file, viz)

    with open(out_ann_file, "w") as f:
        json.dump(data, f)


def keypoint_detection_main(args):
    """
    annotation{
"keypoints": [x1,y1,v1,...], "num_keypoints": int, "[cloned]": ...,
}

# 一级
categories[{
"keypoints": [str], "skeleton": [edge], "[cloned]": ...,
}]

"[cloned]": denotes fields copied from object detection annotations defined above.
        :param args:
        :return:
        """
    now = datetime.datetime.now()

    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
        licenses=[dict(url=None, id=0, name=None, )],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        anno_type=args.anno_type,  # coco之外的字段
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id,
            # "keypoints": [x1,y1,v1,...], "num_keypoints": int,
        ],
        categories=[
            # supercategory, id, name,
            # "keypoints": [str], "skeleton": [edge]
        ],
    )

    # 将labels.txt中的class_name映射到class_id
    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        class_names.append(class_name)
        class_name_to_id[class_name] = class_id
        data["categories"].append(
            dict(supercategory=None, id=class_id, name=class_name,
                 keypoints=["0", "1", "2", "3", "4",
                            "5", "6", "7", "8", "9",
                            "10", "11", "12", "13", "14",
                            "15", "16", "17", "18", "19", "20"],
                 skeleton=[])
        )
    class_names = tuple(class_names)
    print("class_names:", class_names)
    out_class_names_file = osp.join(args.output_dir, "class_names.txt")
    with open(out_class_names_file, "w") as f:
        f.writelines("\n".join(class_names))
    print("Saved class_names:", out_class_names_file)

    # 获取输入目录下的json文件
    out_ann_file = osp.join(args.output_dir, "annotations.json")
    label_files = glob.glob(osp.join(args.input_dir, "*.json"))
    for image_id, filename in enumerate(label_files):
        print("Generating dataset from:", filename)

        # 读取json文件
        # 从图像文件路径或者从json文件中的imageData字段获取图像，并保存到imageData
        label_file = labelme.LabelFile(filename=filename)

        base = osp.splitext(osp.basename(filename))[0]
        out_img_file = osp.join(args.output_dir, "JPEGImages", base + ".jpg")

        img = labelme.utils.img_data_to_arr(label_file.imageData)
        imgviz.io.imsave(out_img_file, img)
        data["images"].append(
            dict(
                license=0,
                url=None,
                file_name=osp.relpath(out_img_file, osp.dirname(out_ann_file)),
                height=img.shape[0],
                width=img.shape[1],
                date_captured=None,
                id=image_id,
            )
        )

        bboxes = []
        labels = []
        annotation = dict(
                        id=None,
                        image_id=None,
                        category_id=None,
                        segmentation=None,
                        area=None,
                        bbox=None,
                        iscrowd=0,
                        keypoints=[],
                        num_keypoints=0,
                    )
        for shape in label_file.shapes:
            if shape["shape_type"] == "rectangle":
                class_name = shape["label"]
                class_id = class_name_to_id[class_name]

                (xmin, ymin), (xmax, ymax) = shape["points"]
                # swap if min is larger than max.
                xmin, xmax = sorted([xmin, xmax])
                ymin, ymax = sorted([ymin, ymax])

                # 用于显示结果
                bboxes.append([ymin, xmin, ymax, xmax])
                labels.append(class_id)

                # 保存信息到标注文件
                # bbox: x, y, w, h
                x = xmin
                y = ymin
                w = xmax - xmin
                h = ymax - ymin
                bbox = [x, y, w, h]
                area = w * h
                segmentation = [[xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]]

                annotation.update(
                    dict(
                        id=len(data["annotations"]),
                        image_id=image_id,
                        category_id=class_id,
                        segmentation=segmentation,
                        area=area,
                        bbox=bbox,
                        iscrowd=0,
                    )
                )
                pass
            elif shape["shape_type"] == "point":
                annotation["num_keypoints"] += 1
                (x, y) = shape["points"][0]
                v = 2
                annotation["keypoints"].append(x)
                annotation["keypoints"].append(y)
                annotation["keypoints"].append(v)
                pass
            else:
                print(
                    "Skipping shape: label={label}, "
                    "shape_type={shape_type}".format(**shape)
                )
                continue

        data["annotations"].append(annotation)

        if not args.noviz:
            captions = [class_names[label] for label in labels]
            viz = imgviz.instances2rgb(
                image=img,
                labels=labels,
                bboxes=bboxes,
                captions=captions,
                font_size=15,
            )
            out_viz_file = osp.join(
                args.output_dir, "Visualization", base + ".jpg"
            )
            imgviz.io.imsave(out_viz_file, viz)

    with open(out_ann_file, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    # input_dir = "/media/home/zql/data/food_dataset/object_detection/input"
    # output_dir = "/media/home/zql/data/food_dataset/object_detection/output"
    # label_file = "/media/home/zql/data/food_dataset/object_detection/labels.txt"
    # anno_type = "object_detection"

    # input_dir = "/media/home/zql/data/food_dataset/instance_segmentation/input"
    # output_dir = "/media/home/zql/data/food_dataset/instance_segmentation/output"
    # label_file = "/media/home/zql/data/food_dataset/instance_segmentation/labels.txt"
    # anno_type = "instance_segmentation"

    # input_dir = "/media/home/zql/data/food_dataset/keypoint_detection/input"
    # output_dir = "/media/home/zql/data/food_dataset/keypoint_detection/output"
    # label_file = "/media/home/zql/data/food_dataset/keypoint_detection/labels.txt"
    # anno_type = "keypoint_detection"

    # input_dir = "/media/home/zql/data/food_dataset/fne_obj_det/input"
    # output_dir = "/media/home/zql/data/food_dataset/fne_obj_det/output"
    # label_file = "/media/home/zql/data/food_dataset/fne_obj_det/labels.txt"
    # anno_type = "object_detection"

    input_dir = "/media/home/zql/data/food_dataset/fne_obj_det/input"
    output_dir = "/media/home/zql/data/food_dataset/fne_obj_det/output"
    label_file = "/media/home/zql/data/food_dataset/fne_obj_det/labels.txt"
    anno_type = "object_detection"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_dir", default=input_dir,
                        help="input annotated directory")
    parser.add_argument("--output_dir", default=output_dir,
                        help="output dataset directory")
    parser.add_argument("--labels", default=label_file,
                        help="labels file")
    parser.add_argument("--anno_type", default=anno_type,
                        help="anno_type: object_detection | instance_segmentation | keypoint_detection")

    parser.add_argument(
        "--noviz", help="no visualization", action="store_true"
    )
    args = parser.parse_args()

    # if osp.exists(args.output_dir):
    #     print("Output directory already exists:", args.output_dir)
    #     sys.exit(1)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(osp.join(args.output_dir, "JPEGImages"), exist_ok=True)
    if not args.noviz:
        os.makedirs(osp.join(args.output_dir, "Visualization"), exist_ok=True)
    print("Creating dataset:", args.output_dir)

    anno_type = args.anno_type

    if anno_type == "object_detection":
        object_detection_main(args)
    elif anno_type == "instance_segmentation":
        instance_segmentation_main(args)
    elif anno_type == "keypoint_detection":
        keypoint_detection_main(args)
