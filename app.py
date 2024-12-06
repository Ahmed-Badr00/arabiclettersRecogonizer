from flask import Flask, request, jsonify
import torch
from PIL import Image
from torchvision import transforms
import timm

# Initialize Flask app
app = Flask(__name__)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
googlenet = torch.load("models/googlenet.pth", map_location=device)
swin_transformer = torch.load("models/swin_transformer.pth", map_location=device)
googlenet_se = torch.load("models/googlenet_se.pth", map_location=device)

googlenet.eval()
swin_transformer.eval()
googlenet_se.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Voting function
def vote(predictions):
    results = []
    for i in range(len(predictions[0])):
        counts = {}
        for pred in predictions:
            value = pred[i]
            counts[value] = counts.get(value, 0) + 1
        max_count = max(counts.values())
        candidates = [k for k, v in counts.items() if v == max_count]
        results.append(candidates[0] if len(candidates) == 1 else predictions[0][i])
    return results

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the request
    file = request.files['file']
    image = Image.open(file.stream).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Get predictions
    with torch.no_grad():
        preds1 = torch.argmax(googlenet(input_tensor), dim=1).cpu().numpy()
        preds2 = torch.argmax(swin_transformer(input_tensor), dim=1).cpu().numpy()
        preds3 = torch.argmax(googlenet_se(input_tensor), dim=1).cpu().numpy()

    # Ensemble predictions
    final_prediction = vote([preds1, preds2, preds3])

    return jsonify({'prediction': final_prediction[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

