from flask import Flask, request, jsonify, render_template
import torch
from PIL import Image
from torchvision import transforms, models
import timm
import torch.nn as nn
import matplotlib.pyplot as plt
import os 

# Initialize Flask app
app = Flask(__name__)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define classes
classes = ('Heh Middle', 'Noon End', 'Ain Begin', 'Ain End', 'Ain Middle', 'Ain Regular',
        'Alif End', 'Alif Hamza', 'Alif Regular', 'Beh Begin', 'Beh End','Beh Middle','Beh Regular',
           'Dal End', 'Dal Regular', 'Feh Begin', 'Feh End', 'Feh Middle', 'Feh Regular', 'Heh Begin',
           'Heh End', 'Heh Regular', 'Jeem Begin', 'Jeem End', 'Jeem Middle', 'Jeem Regular', 'Kaf Begin'
           , 'Kaf End','Kaf Middle', 'Kaf Regular', 'Lam Alif', 'Lam Begin', 'Lam End', 'Lam Middle',
           'Lam Regular','Mem Begin', 'Mem End', 'Mem Middle', 'Mem Regular', 'Noon Begin', 'Noon End',
           'Noon Middle', 'Noon Regular', 'Qaf Begin', 'Qaf End', 'Qaf Middle', 'Raa end',
           'Raa Regular', 'Saad Begin','Saad End', 'Saad Middle', 'Saad Regular', 'Seen Begin', 'Seen End',
           'Seen Middle', 'Seen Regular','Tah Middle', 'Tah End', 'Tah Regular', 'Waw End', 'Waw Regular',
           'Yaa2 Begin', 'Yaa2 End','Yaa2 Middle', 'Yaa2 Regular')
           
# Define SEBlock and InceptionWithSE
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class InceptionWithSE(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(InceptionWithSE, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(True),
            SEBlock(ch1x1)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.BatchNorm2d(ch3x3red),
            nn.ReLU(True),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(True),
            SEBlock(ch3x3)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.BatchNorm2d(ch5x5red),
            nn.ReLU(True),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(True),
            SEBlock(ch5x5)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(True),
            SEBlock(pool_proj)
        )

    def forward(self, x):
        outputs = [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)]
        return torch.cat(outputs, 1)

class GoogLeNetSE(nn.Module):
    def __init__(self, num_classes=65):
        super(GoogLeNetSE, self).__init__()
        self.base_model = models.googlenet(pretrained=True, transform_input=False)
        self.base_model.inception3a = InceptionWithSE(192, 64, 96, 128, 16, 32, 32)
        self.base_model.inception3b = InceptionWithSE(256, 128, 128, 192, 32, 96, 64)
        self.base_model.inception4a = InceptionWithSE(480, 192, 96, 208, 16, 48, 64)
        self.base_model.inception4b = InceptionWithSE(512, 160, 112, 224, 24, 64, 64)
        self.base_model.inception4c = InceptionWithSE(512, 128, 128, 256, 24, 64, 64)
        self.base_model.inception4d = InceptionWithSE(512, 112, 144, 288, 32, 64, 64)
        self.base_model.inception4e = InceptionWithSE(528, 256, 160, 320, 32, 128, 128)
        self.base_model.inception5a = InceptionWithSE(832, 256, 160, 320, 32, 128, 128)
        self.base_model.inception5b = InceptionWithSE(832, 384, 192, 384, 48, 128, 128)
        self.base_model.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        return self.base_model(x)

# Instantiate models
googlenet = models.googlenet(pretrained=False, aux_logits=False, num_classes=65).to(device)
swin_transformer = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=65).to(device)
googlenet_se = GoogLeNetSE(num_classes=65).to(device)

# Load weights
googlenet.load_state_dict(torch.load("models/googlenett_weights.pth", map_location=device))
swin_transformer.load_state_dict(torch.load("models/swin_transformer_weights.pth", map_location=device))
googlenet_se.load_state_dict(torch.load("models/googlenet_se_weights.pth", map_location=device))

# Set models to evaluation mode
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


# Serve homepage
@app.route('/')
def home():
    return render_template('index.html')

# for image upload
@app.route('/upload', methods=['POST'])
def upload():
    # Ensure the static directory exists
    if not os.path.exists("static"):
        os.makedirs("static")
    
    file = request.files['file']
    image_path = os.path.join("static", "uploaded_image.jpg")
    file.save(image_path)  # Save the uploaded image

    # Display the image using Matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    plt.figure(figsize=(3, 2))  # Desired figure size
    img = mpimg.imread(image_path)
    plt.imshow(img)
    plt.axis('off')  # Hide axes
    output_path = os.path.join("static", "rendered_image.jpg")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Return the rendered image URL
    return jsonify({'image_url': output_path})

# Handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Ensure the static directory exists
    if not os.path.exists("static"):
        os.makedirs("static")
    
    file = request.files['file']
    image_path = os.path.join("static", "uploaded_image.jpg")
    file.save(image_path)  # Save the uploaded image

    # Load and process the image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        preds1 = torch.argmax(googlenet(input_tensor), dim=1).cpu().numpy()
        preds2 = torch.argmax(swin_transformer(input_tensor), dim=1).cpu().numpy()
        preds3 = torch.argmax(googlenet_se(input_tensor), dim=1).cpu().numpy()

    # Voting for the final prediction
    final_prediction = preds1[0]  # Assuming consistency across models
    class_name = classes[final_prediction]

    # Display the image using Matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    plt.figure(figsize=(3, 2))  # Desired figure size
    img = mpimg.imread(image_path)
    plt.imshow(img)
    plt.axis('off')  # Hide axes
    output_path = os.path.join("static", "rendered_image.jpg")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Return the prediction and the rendered image URL
    return jsonify({'prediction': class_name, 'image_url': output_path})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
