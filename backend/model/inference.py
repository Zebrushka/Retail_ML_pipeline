# Inference
import torch
from torchvision import transforms
import json
from PIL import Image


def predict(image):
    path_to_model = 'efficientnet-b0.pth'
    device = torch.device('cpu')
    model = torch.load(path_to_model, map_location=device)

    # Preprocess image
    tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

    img = tfms(image).unsqueeze(0)

    # Load class names
    labels_map = json.load(open('labels.txt'))
    labels_map = [labels_map[str(i)] for i in range(15)]

    # Classify
    model.eval()
    with torch.no_grad():
        outputs = model(img)

    # Return predictions
    for idx in torch.topk(outputs, k=1).indices.squeeze(0).tolist():
        probability = torch.softmax(outputs, dim=1)[0, idx].item()
        label = labels_map[idx]
        print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=probability*100))
    # probability = torch.softmax(outputs, dim=1)[0, 1].item()
    # label = labels_map[1]

    return label, probability