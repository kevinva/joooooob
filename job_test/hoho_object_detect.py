import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import cv2

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()


COCO_INSTANCE_CATEGORY_NAMES = [
    '__BACKGROUND__', 'person', 'bicycle', 'car', 'motorcycle',
    'airplane', 'bus', 'train', 'trunk', 'boat', 'traffic light',
    'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A',
    'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

image = Image.open('./WechatIMG263.jpeg')
transform_d = transforms.Compose([transforms.ToTensor()])
image_t = transform_d(image)
pred = model([image_t])
print(pred)

# pred_class = [COCO_INSTANCE_CATEGORY_NAMES[ii] for ii in list(pred[0]['labels'].numpy())]
# pred_score = list(pred[0]['scores'].detach().numpy())

# pred_boxes = [[ii[0], ii[1], ii[2], ii[3]] for ii in list(pred[0]['boxes'].detach().numpy())]
# pred_index = [pred_score.index(x) for x in pred_score if x > 0.5]

# fontsize = np.int16(image.size[1] / 20)
# font1 = ImageFont.load_default()

# draw = ImageDraw.Draw(image)
# for index in pred_index:
#     box = pred_boxes[index]
# #     cv2.rectangle(img=image, pt1=[int(box[0]), int(box[1])], pt2=[int(box[2]), int(box[3])], color=(0, 0, 225), thickness=3)
#     draw.rectangle(box, outline='blue')
#     texts = pred_class[index] + ';' + str(np.round(pred_score[index], 2))
#     draw.text((box[0], box[1]), texts, fill='blue', font=font1)
# #     font = cv2.FONT_HERSHEY_SIMPLEX
# #     cv2.putText(image, texts, (int(box[0]), int(box[1])), font, 1, (200, 255, 155), cv2.LINE_AA)
    
# # cv2.imshow(image)
# plt.imshow(image)
# plt.show()


cap = cv2.VideoCapture(0)
transform = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)
while True:
    ret, frame = cap.read()
    image = frame
    frame = transform(frame)
    pred = model([frame])

    # 检测出目标的类别和得分
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[ii] for ii in list(pred[0]['labels'].numpy())]
    pred_score = list(pred[0]['scores'].detach().numpy())

    # 检测出目标的边界框
    pred_boxes = [[ii[0], ii[1], ii[2], ii[3]] for ii in list(pred[0]['boxes'].detach().numpy())]

    # 只保留识别的概率大约 0.5 的结果。
    pred_index = [pred_score.index(x) for x in pred_score if x > 0.5]

    for index in pred_index:
        box = pred_boxes[index]
        cv2.rectangle(img=image, pt1=[int(box[0]), int(box[1])], pt2=[int(box[2]), int(box[3])],
                      color=(0, 0, 225), thickness=3)
        texts = pred_class[index] + ":" + str(np.round(pred_score[index], 2))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, texts, (int(box[0]), int(box[1])), font, 1, (200, 255, 155), 2, cv2.LINE_AA)

    cv2.imshow('摄像头', image)
    cv2.waitKey(10)