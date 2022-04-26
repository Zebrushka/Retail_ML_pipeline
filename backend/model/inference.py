# Inference
import torch
from torchvision import transforms
import json
import cv2
from PIL import Image

# path to model
path_to_model_clf = 'model/efficientnet-b0.pth'
path_to_model_det = 'model/yolo_detector.pt'

# inference on CPU
device = torch.device('cpu')

# load model
model_clf = torch.load(path_to_model_clf, map_location=device)
model_det = torch.hub.load('ultralytics/yolov5', 'custom', path=path_to_model_det)

# eval
model_det.eval()
model_clf.eval()

def return_label(image):
    # return bbox and label
    frame = cv2.imread(image)
    detections = model_det(frame[..., ::-1])
    result_bbox = detections.pandas().xyxy[0].to_dict(orient="records")
    return result_bbox


def crop(img, box_points):
    """Takes in a `img` an image OpenCV object and `box_points` which is a list containg the upper left and lower right coordinates of the bounding box to crop out. For example,
    box_points should be [x1, y1, x2, y2]."""
    x1 = box_points[0]
    y1 = box_points[1]
    x2 = box_points[2]
    y2 = box_points[3]

    crop_img = img[y1:y2, x1:x2]
    return crop_img


def predict(image, model_clf, model_det):

    # detect objects from yolo

    results = model_det(image)

    # Preprocess image for clf
    tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

    img = tfms(image).unsqueeze(0)

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
        print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=probability*100))

    return label, probability, results