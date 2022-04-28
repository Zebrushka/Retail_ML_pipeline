# Inference
import torch
from torchvision import transforms
import json
import cv2
from PIL import Image
import numpy as np
import io
import base64


# path to model
path_to_model_clf = 'model/efficientnet-b0.pth'
path_to_model_det = 'model/yolo_detector.pt'

# inference on CPU
device = torch.device('cpu')

# load model
model_clf = torch.load(path_to_model_clf, map_location=device)
model_det = torch.hub.load('ultralytics/yolov5', 'custom', path=path_to_model_det)

# Inference Settings
model_det.conf = 0.3

# eval
model_det.eval()
model_clf.eval()


def crop(input_image):
    """Takes in a `img` an image OpenCV object and `box_points` which is a list
    containg the upper left and lower right coordinates of the bounding box to crop out.
    For example, box_points should be [x1, y1, x2, y2]."""

    def return_bbox(input_image):
        # return bbox and label
        detections = model_det(input_image[..., ::-1])
        print(detections)
        labels, result_bbox = detections.xyxyn[0][:, -1].numpy(), detections.xyxyn[0][:, :-2].numpy()
        index_label = list(labels).index(0)
        print(index_label, result_bbox)
        x1 = result_bbox[index_label][0]
        y1 = result_bbox[index_label][1]
        x2 = result_bbox[index_label][2]
        y2 = result_bbox[index_label][3]

        box_points = [x1, y1, x2, y2]

        return box_points, detections

    image = cv2.imread(input_image)
    dim = (384, 640)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    height, width, channels = image.shape
    print('resize image ', height, width, channels)
    box_points, detections = return_bbox(image)
    print('box point ', box_points)

    x1 = int(float(box_points[0] * 384))
    y1 = int(float(box_points[1] * 640))
    x2 = int(float(box_points[2] * 384))
    y2 = int(float(box_points[3] * 640))

    print('box point adaptation', x1, y1, x2, y2)

    crop_image = image[y1:y2, x1:x2]
    height, width, channels = crop_image.shape
    print('image size after crop', height, width, channels)

    return crop_image, detections


def predict(input_image):
    item, detections = crop(input_image)

    image_for_pred = Image.fromarray(item)

    # Preprocess image for clf
    tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])

    img = tfms(image_for_pred).unsqueeze(0)

    # Load class names
    labels_map = json.load(open('model/labels.txt'))
    labels_map = [labels_map[str(i)] for i in range(15)]

    # Classify
    with torch.no_grad():
        outputs = model_clf(img)

    # Return predictions
    for idx in torch.topk(outputs, k=1).indices.squeeze(0).tolist():
        probability = torch.softmax(outputs, dim=1)[0, idx].item()
        label = labels_map[idx]
        print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=probability * 100))

    # convert to b64
    detections.imgs
    detections.render()
    for img in detections.imgs:
        buffered = io.BytesIO()
        img_base64 = Image.fromarray(img)
        img_base64.save(buffered, format="JPEG")

        image_witt_BBox = base64.b64encode(buffered.getvalue()).decode('utf-8')



    return label, probability, image_witt_BBox