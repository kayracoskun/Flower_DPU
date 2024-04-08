import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.utils import plot_model

from tensorflow_model_optimization.quantization.keras import vitis_quantize

X = []
y = []

# Adjusted to resize images for uniformity
def resize_and_load_images(input_dir, size=(224, 224)):
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            input_path = os.path.join(input_dir, filename)
            img = Image.open(input_path).resize(size)
            X.append(np.asarray(img) / 255.0)
            y.append(["daisy", "dandelion", "rose", "sunflower", "tulip"].index(input_dir.split('/')[-2]))

    print(f"Done adding images from {input_dir}")

# Enhanced model with batch normalization
def build_enhanced_model(input_shape=(224, 224, 3), num_classes=5):
    model = models.Sequential([
        layers.Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=2,strides=2),
        layers.BatchNormalization(),
        
        layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
        layers.MaxPooling2D(pool_size=2,strides=2),
        layers.BatchNormalization(),

        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(units=128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Preprocess images
for flower in ["daisy", "dandelion", "rose", "sunflower", "tulip"]:
    resize_and_load_images(f"dataset/{flower}/resized")

X = np.array(X)
y = np.array(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

model = build_enhanced_model()

model.compile(
    optimizer='adam', 
    loss=SparseCategoricalCrossentropy(from_logits=False), 
    metrics=['accuracy']
)

model.summary()
plot_model(model, to_file='flower_cnn_arch.png', show_shapes=True, show_layer_names=True)

model.fit(X_train, y_train, epochs=10, validation_split=0.2)

#model.save('flower_cnn.h5')

# Quantize and store AI models
quantizer = vitis_quantize.VitisQuantizer(model)

# Number of data sets to be passed is 100-1000 without labels
quantized_model = quantizer.quantize_model(calib_dataset=X_train[0:500])
quantized_model.save("quantized_flower_cnn.h5")