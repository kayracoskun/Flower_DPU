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

def resize_image(input_dir):
    for filename in os.listdir(input_dir):
        # Check if the file is an image
        if filename.endswith((".jpg", ".jpeg", ".png")):
            input_path = os.path.join(input_dir, filename)
            
            with Image.open(input_path) as img:
                X.append(np.asarray(img) / 255.0)
                
                if input_dir == "dataset/daisy/resized": y.append(0)
                elif input_dir == "dataset/dandelion/resized": y.append(1)
                elif input_dir == "dataset/rose/resized": y.append(2)
                elif input_dir == "dataset/sunflower/resized": y.append(3)
                else: y.append(4)
                
    print("Done adding images from", input_dir)


def build_enhanced_model(input_shape=(224, 224, 3), num_classes=5):
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape, strides=(2, 2), padding='same', name='block1_conv1'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same', name='block1_conv2'))
    model.add(layers.Dropout(0.2))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding='same', name='block2_conv1'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding='same', name='block2_conv2'))
    model.add(layers.Dropout(0.2))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    model.add(layers.Conv2D(256, (3, 3), activation='relu', strides=(1, 1), padding='same', name='block3_conv1'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', strides=(1, 1), padding='same', name='block3_conv2'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', strides=(1, 1), padding='same', name='block3_conv3'))
    model.add(layers.Dropout(0.2))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    model.add(layers.Conv2D(512, (3, 3), activation='relu', strides=(1, 1), padding='same', name='block4_conv1'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', strides=(1, 1), padding='same', name='block4_conv2'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', strides=(1, 1), padding='same', name='block4_conv3'))
    model.add(layers.Dropout(0.2))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    model.add(layers.Conv2D(512, (3, 3), activation='relu', strides=(1, 1), padding='same', name='block5_conv1'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', strides=(1, 1), padding='same', name='block5_conv2'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', strides=(1, 1), padding='same', name='block5_conv3'))
    model.add(layers.Dropout(0.2))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    model.add(layers.Flatten())
    model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model


resize_image("dataset/daisy/resized")
resize_image("dataset/dandelion/resized")
resize_image("dataset/rose/resized")
resize_image("dataset/sunflower/resized")
resize_image("dataset/tulip/resized")

# X and y sets
X = np.array(X)
y = np.array(y)

print(X.shape , y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
print("Train and test datas are created successfully")

model = build_enhanced_model()

model.summary()
plot_model(model, to_file='flower_vgg16_model.png', show_shapes=True, show_layer_names=True)

model.compile(
    optimizer = 'adam', 
    loss = SparseCategoricalCrossentropy(from_logits=False),
    metrics = ['acc']
)

model.fit(X_train, y_train, epochs=10)

# Quantize and store AI models
quantizer = vitis_quantize.VitisQuantizer(model)

# Number of data sets to be passed is 100-1000 without labels
quantized_model = quantizer.quantize_model(calib_dataset=X_train[0:500])
quantized_model.save("quantized_flower_vgg16.h5")