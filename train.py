from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import os

learnrate = 1e-4
EPOCHS = 20
batch = 32

directory = r"C:\Users\akram\Documents\Face AMsk\train"
category = ["with_mask", "without_mask"]
print("Labelling All the Images:")
data = []
labels = []
for category in category:
    path = os.path.join(directory, category)
    for img in os.listdir(path):
        img_path = os.path.join(path,img)
        image = load_img(img_path,target_size=(240, 240))
        image = img_to_array(image)
        image = preprocess_input(image)
        data.append(image)
        labels.append(category)

#LabelBinarizer is used to convert labels to one-hot encoding
lb = LabelBinarizer()  
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=0.20, stratify=labels, random_state=25)

# Data Augumentation Construction
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# MobileNetV2 Model Construction
baseModel = MobileNetV2(weights="imagenet", include_top=False,input_tensor=Input(shape=(240, 240, 3)))

# construct the head of the model that will be placed on top of the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
	layer.trainable = False

print("Model Compilation")
opt = Adam(lr=learnrate, decay=learnrate / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])

print("Head Training")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=batch),
	steps_per_epoch=len(trainX) // batch,
	validation_data=(testX, testY),
	validation_steps=len(testX) // batch,
    epochs=EPOCHS)

print("Network is Being Evaluated")
predNet = model.predict(testX, batch_size=batch)
predNet = np.argmax(predNet, axis=1)
print(classification_report(testY.argmax(axis=1), predNet,target_names=lb.classes_))
print("Saving The Model")
model.save("mask_detector.model", save_format="h5")
