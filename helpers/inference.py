import torch
import torchvision.transforms as transforms
from PIL import Image

classes_dict = {0: 'Driving License', 1: 'PAN', 2: 'Passport', 3: 'Adhaar Card', 4: 'Adhaar Card', 5: 'Adhaar Card'}

def classify_image(model, device, image):
    image = image.convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.6544, 0.6596, 0.6643], std=[0.1878, 0.1899, 0.1975]),
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class = torch.argmax(probabilities).item()

    # print(f"Predicted class: {predicted_class}")
    # print(f"Probabilities: {probabilities}")
    # print(f"Class name: {classes_dict[predicted_class]}")
    return classes_dict[predicted_class]
