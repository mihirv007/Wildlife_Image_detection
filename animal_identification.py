import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers,models


#load dataset

train_dir='/Users/mihirverma/Animal_indentification/animals/train'

test_dir='/Users/mihirverma/Animal_indentification/animals/val'

image_size=(224,224)
#training dataset

train_datagen=ImageDataGenerator(rescale=1./255,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True
)

training_set=train_datagen.flow_from_directory(train_dir,
                        target_size=image_size,
                        batch_size=64,
                        class_mode='sparse'
)

#testing dataset 

test_datagen=ImageDataGenerator(rescale=1./255)

testing_set=test_datagen.flow_from_directory(test_dir,
            target_size=image_size,
            batch_size=64,
            class_mode='sparse'
)

#define model architecture

model=models.Sequential([
    layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(224,224,3)),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Dropout(0.25),
    layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Dropout(0.25),
    layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu'),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(5,activation='softmax')
])

#compile the model

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#train the model

training_of_model=model.fit(x=training_set,validation_data=testing_set,epochs=20)

#Evalution the model

test_loss,test_acc=model.evaluate(testing_set)
print('accuracy:',test_acc)
print('loss',test_loss)

#save the model

model.save('/Users/mihirverma/Animal_indentification/animal_model_1.0.keras')

