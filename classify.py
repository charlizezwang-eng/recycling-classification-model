import sys
import cv2
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from reshape import reshape_model


checkpoint_path = 'models/model_best.pth.tar'
cam_index = 0

ckpt = torch.load(checkpoint_path, map_location='cpu')
model = models.resnet18(weights=None)
model = reshape_model(model, 'resnet18', len(ckpt['classes']))
model.load_state_dict(ckpt['state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

if len(sys.argv) > 1:
    cam_index = int(sys.argv[1])

cap = cv2.VideoCapture(cam_index)
if not cap.isOpened():
    raise RuntimeError(f'Could not open webcam device {cam_index}')

print(f'Webcam opened on device {cam_index}. Press Ctrl+C to quit.')

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print('Failed to read frame from webcam.')
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_frame)
    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(1).item()
        if (pred == 0) or (pred == 1) or ((pred>= 3) and (pred <= 8)) or ((pred>= 11) and (pred<=14)) or (pred ==17) or (pred == 19) or pred == 20 or pred == 22 or pred ==23:
            label = 'Compostable'
        elif (pred == 2) or (pred == 10) or (pred == 18) or ((pred>=25) and (pred<=26)):
            label = 'Recyclable'
        else:
            label = 'Landfill'

    frame_count += 1
    print(f'[{frame_count}] {label} ({pred})')

    if frame_count >= 10:
        break

cap.release()
