
import numpy as np
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input

IMAGE_SIZE=(224,224)
IMAGE_FULL_SIZE=(224,224,3)
batchSize=32

allImages=np.load("C:\\Users\\Samhi\\Desktop\\dogbreed\\allDogImages.npy")
allLabels=np.load("C:\\Users\\Samhi\\Desktop\\dogbreed\\allDogLables.npy")

print(allImages.shape)
print(allLabels.shape)


print(allLabels)
from sklearn.preprocessing import LabelEncoder
Le=LabelEncoder()
integerLabels=Le.fit_transform(allLabels)
print(integerLabels)

numOfCategories=len(np.unique(integerLabels))
print(numOfCategories)

from tensorflow.keras.utils import to_categorical
allLabelsForModel=to_categorical(integerLabels,num_classes=numOfCategories)
print(allLabelsForModel)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Memory-efficient split using indices only
indices = np.arange(len(allImages))
train_idx, val_idx = train_test_split(indices, test_size=0.3, random_state=42, shuffle=True)

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.2
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# No copying! Only indexing inside flow()
train_generator = train_datagen.flow(
    allImages[train_idx],  # This indexing is lazy / view-based in NumPy
    allLabelsForModel[train_idx],
    batch_size=32,
    shuffle=True
)

val_generator = val_datagen.flow(
    allImages[val_idx],
    allLabelsForModel[val_idx],
    batch_size=32,
    shuffle=False
)




from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0

# Base Model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=IMAGE_FULL_SIZE)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.4)(x)
output = Dense(numOfCategories, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.summary()


from tensorflow.keras.optimizers import Adam
model.compile(
    optimizer=Adam(learning_rate=3e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

steps_per_epoch = len(train_generator)
validation_steps = len(val_generator)


history = model.fit(
    train_generator,
    validation_data=val_generator,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=5,
    verbose=1
)


# Stage-2 Fine Tuning (continue training)
for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_generator,
    validation_data=val_generator,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=10,
    verbose=1
)

model.save("dog_breed_model.h5")
print("Model saved successfully!")

 
