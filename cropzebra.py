import numpy as np
import cv2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import os

Path = "./Zebra/"
filename = os.listdir(Path)

for i in filename:
    im = cv2.imread("./Zebra/"+i)

    cfg = get_cfg()
    cfg.merge_from_file("./detectron2-master/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)

    pred_classes = outputs["instances"].pred_classes
    pred_boxes = outputs["instances"].pred_boxes
    pred_masks = outputs["instances"].pred_masks


    pred_classes_zebra = []
    pred_boxes_zebra = []
    pred_masks_zebra = []


    #only pick up zebra
    L = len(pred_classes)
    for j in range(L):
        if pred_classes[j] == 22:
            pred_classes_zebra.append(outputs['instances'].pred_classes[j])
            pred_boxes_zebra.append(outputs['instances'].pred_boxes[j])
            pred_masks_zebra.append(outputs['instances'].pred_masks[j])


    N = len(pred_classes_zebra)
    if N == 0:
        print('No zebra in image')

    elif N == 1:
        bbox = pred_boxes_zebra[0]
        bbox = bbox.tensor
        bbox = bbox.cpu()
        bbox = np.array(bbox)
        y0 = int(bbox[0,1])
        y1 = int(bbox[0,3])
        x0 = int(bbox[0,0])
        x1 = int(bbox[0,2])
        im_crop = im[y0:y1, x0:x1]
        pred_masks_zebra = np.array(pred_masks_zebra[0].cpu())
        mask_crop = pred_masks_zebra[y0:y1, x0:x1]
        for j in range(3):
            im_crop[:, :, j] = im_crop[:, :, j] * mask_crop

        im_resize = cv2.resize(im_crop,(600,400), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite("./Zebra_crop/"+i,im_resize)
        im_crop = []
        mask_crop = []
        im_resize = []

    else:
        A = []
        for j in range(N):
            A.append(pred_boxes_zebra[j].area())
        index_I = A.index(max(A))

        bbox = pred_boxes_zebra[index_I]
        bbox = bbox.tensor
        bbox = bbox.cpu()
        bbox = np.array(bbox)
        y0 = int(bbox[0, 1])
        y1 = int(bbox[0, 3])
        x0 = int(bbox[0, 0])
        x1 = int(bbox[0, 2])
        im_crop = im[y0:y1, x0:x1]
        pred_masks_zebra = np.array(pred_masks_zebra[index_I].cpu())
        mask_crop = pred_masks_zebra[y0:y1, x0:x1]
        for j in range(3):
            im_crop[:, :, j] = im_crop[:, :, j] * mask_crop

        im_resize = cv2.resize(im_crop, (600, 400), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite("./Zebra_crop/" + i, im_resize)
        im_crop = []
        mask_crop = []
        im_resize = []

