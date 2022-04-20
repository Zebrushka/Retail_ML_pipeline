# Inference
import torch
from torchvision import transforms
import json
import PIL


def predict(PATH):

    device = torch.device('cpu')
    model = model = torch.load('/model/efficientnet-b0.pth', map_location=device)

    # Preprocess image
    tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    img = tfms(Image.open(PATH)).unsqueeze(0)

    # Load class names
    labels_map = json.load(open('labels.txt'))
    labels_map = [labels_map[str(i)] for i in range(15)]

    # Classify
    model.eval()
    with torch.no_grad():
        outputs = model(img)

    # Return predictions
    # for idx in torch.topk(outputs, k=5).indices.squeeze(0).tolist():
    #     prob = torch.softmax(outputs, dim=1)[0, idx].item()
    #     print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob*100))
    prob = torch.softmax(outputs, dim=1)[0, idx].item()

    return label, probability