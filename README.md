# Illinois-Most-Common-Bees-Identifier
```
This repository contains large files split into smaller parts due to GitHub size limits
Model's file has been split into three smaller .zip files
To reassemble:
Using 7-Zip
Download all model files and move to same folder
Extract from bee_model.zip.001 using 7-Zip with "extract here"
7-Zip will automatically combine and extract the full file
This machine learning model is capable of taking an image of a top 15 most common bee in Illinois as input and outputting its species.

Bees included in model with iNaturalist taxon ID included for convenience:
Western Honey Bee Apis Mellifera 47219
Common Eastern Bumble Bee Bombus Impatiens 118970
Two-spotted Bumble Bee Bombus Bimaculatus 52779
Brown-belted Bumble Bee Bombus Griseocollis 120215
Eastern Carpenter Bee Xylocopa Virginica 51110
Metallic Green Sweat Bee Agapostemon Virescens 82530
Brown Sweat Bee Halictus Ligatus 154298
Blue Orchard Mason Bee Osmia Lignaria 121507
Squash Bee Peponapis Pruinosa 1594041
Southern Carpenter Bee Xylocopa Micans 133788
Unequal Cellophane Bee Colletes Inaequalis 199041
Orange-legged Furrow Bee Halictus Rubicundus 127747
Bicolored Sweat Bee Agapostemon Splendens 199056
American Bumble Bee Bombus Pennsylvanicus 56887
Yellow-banded Bumble Bee Bombus Terricola 121517

Classifies 15 common bee species found in Illinois
Accepts .jpg, .jpeg, or .png images
Input images are automatically resized to 300×300 pixels
Returns the most likely species classification

Accuracy: 87.44%
Architecture: EfficientNetB0
Custom Layers:
Dense (512 units, Swish activation, L2 regularized)
Batch Normalization
Dense (256 units, Swish activation)
Dropout (rate=0.3)
Output: Softmax (15 classes)

Phase 1: Frozen Base
15 epochs
Learning rate: 0.0001
Phase 2: Fine-tuning
25 epochs
Learning rate: 1e-5
First 100 layers of base model frozen
Early stopping and learning rate reduction applied during both phases

Dataset: ~1,500 research-grade bee images per class
Split: 70% train, 15% validation, 15% test
Image Augmentations:
Rotation: ±25°
Shear: ±10°
Zoom: ±20%
Brightness: ±20%
Channel Shift: ±50
Horizontal & Vertical Flip
Fill Mode: Reflect

Evaluation Metrics: Classification report & confusion matrix (included in output)
Optimizer: Adam
Batch Size: 8
Total Epochs: Up to 40 (early stopping applied)

Dependencies needed for full model:
tensorflow>=2.8.0
numpy>=1.21.5
pillow>=9.0.0
pandas>=1.3.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
seaborn>=0.11.0
keras-preprocessing>=1.1.2

Prediction:

==========

# Bee Species Predictor
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained bee model
model = tf.keras.models.load_model('INSERT THE NAME OF YOUR MODEL FILE HERE') 


CLASS_NAMES = [
    "Agapostemon splendens (Bicolored Sweat Bee)",
    "Agapostemon virescens (Metallic Green Sweat Bee)",
    "Apis mellifera (Western Honey Bee)",
    "Bombus bimaculatus (Two-spotted Bumble Bee)",
    "Bombus griseocollis (Brown-belted Bumble Bee)",
    "Bombus impatiens (Common Eastern Bumble Bee)",
    "Bombus pensylvanicus (American Bumble Bee)",
    "Bombus terricola (Yellow-banded Bumble Bee)",
    "Colletes inaequalis (Unequal Cellophane Bee)",
    "Halictus ligatus (Brown Sweat Bee)",
    "Halictus rubicundus (Orange-legged Furrow Bee)",
    "Osmia lignaria (Blue Orchard Mason Bee)",
    "Peponapis pruinosa (Squash Bee)",
    "Xylocopa micans (Southern Carpenter Bee)",
    "Xylocopa virginica (Eastern Carpenter Bee)"
]

def predict_bee_species(img_path):
    """Predict bee species from an image file."""
    img = image.load_img(img_path, target_size=(300, 300))
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    print(f"Predicted: {predicted_class} ({confidence:.1f}% confidence)")

# Example usage
predict_bee_species("INSERT YOUR TESTING IMAGE HERE")

==========



































```
