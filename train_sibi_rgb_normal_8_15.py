import h5py
import numpy as np
from i3d_inception import Inception_Inflated3d
import keras
from keras import optimizers
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from keras.preprocessing.image import ImageDataGenerator

FRAME_HEIGHT = 224
FRAME_WIDTH = 224
NUM_FRAMES = 118
NUM_CLASSES = 15
BATCH = 4

#Training I3D 8 module inception freeze transfer learning imagenet dan kinetic dataset 15 class

hf = h5py.File("sibi_rgb_normal_15.h5","r")
train_label_file = open("train_sibi_2.txt","r")
test_label_file = open("test_validation_sibi_2.txt","r")
validation_label_file = open("test_validation_sibi_2.txt","r")
train_raw_labels = train_label_file.read().split("\n")
test_raw_labels = test_label_file.read().split("\n")
validation_raw_labels = validation_label_file.read().split("\n")
train_labels = keras.utils.to_categorical(np.array(train_raw_labels),num_classes = NUM_CLASSES)
test_labels = keras.utils.to_categorical(np.array(test_raw_labels),num_classes = NUM_CLASSES)
validation_labels = keras.utils.to_categorical(np.array(validation_raw_labels),num_classes = NUM_CLASSES)


def generator(type):
  i=0
  counter = 0
  if type=="train":
    while True:
      batch_features = np.zeros((BATCH,NUM_FRAMES, FRAME_WIDTH, FRAME_HEIGHT,3))
      batch_labels = np.zeros((BATCH,NUM_CLASSES))
      for i in range(BATCH):
        batch_features[i] = hf["train"][counter%180]
        batch_labels[i] = train_labels[counter%180]
        # print("Index: "+str(i)+", Counter: "+str(counter))
        # print(batch_labels)
        counter+=1
      yield batch_features,batch_labels
  elif type=="test":
    while True:
      batch_features = np.zeros((1,NUM_FRAMES, FRAME_WIDTH, FRAME_HEIGHT,3))
      batch_labels = np.zeros((1,NUM_CLASSES))
      for i in range(1):
        batch_features[i] = hf["test"][counter%60]
        batch_labels[i] = test_labels[counter%60]
        # print("Index: "+str(i))
        # print(batch_labels)
        counter+=1
      yield batch_features,batch_labels
  elif type=="validation":
    while True:
      batch_features = np.zeros((BATCH,NUM_FRAMES, FRAME_WIDTH, FRAME_HEIGHT,3))
      batch_labels = np.zeros((BATCH,NUM_CLASSES))
      for i in range(BATCH):
        batch_features[i] = hf["validation"][counter%60]
        batch_labels[i] = validation_labels[counter%60]
        # print("Index: "+str(i))
        # print(batch_labels)
        counter+=1
      yield batch_features,batch_labels

rgb_model = Inception_Inflated3d(
            include_top=False,
            weights='rgb_imagenet_and_kinetics',
            input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH,3),
            classes=NUM_CLASSES,endpoint_logit=False)

opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)

index_freeze_layer = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,32,33,34,35,36,37,39,40,41,42,43,44,45,46,47,48,49,50,53,54,55,56,57,58,60,61,62,63,64,65,66,67,68,69,70,71,73,74,75,76,77,78,80,81,82,83,84,85,86,87,88,89,90,91,93,94,95,96,97,98,100,101,102,103,104,105,106,107,108,109,110,111,113,114,115,116,117,118,120,121,122,123,124,125,126,127,128,129,130,131,133,134,135,136,137,138,140,141,142,143,144,145,146,147,148,149,150,151,154,155,156,157,158,159,161,162,163,164,165,166,167,168,169,170,171,172]
cc = 0
for layer in rgb_model.layers:
  print("Layer - "+str(cc)+" "+layer.name)
  if cc in index_freeze_layer:
    print("jadi false")
    layer.trainable = False
  print("Trainable = "+str(layer.trainable)+"\n")
  cc+=1

rgb_model.summary()

rgb_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=opt,
              metrics=['accuracy'])

best_checkpoint = ModelCheckpoint('sibi_rgb_normal_6_weights_best.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
checkpoint = ModelCheckpoint('sibi_rgb_normal_6_weights_epoch.hdf5', monitor='val_acc', verbose=1, save_best_only=False, mode='max')
csv_logger = CSVLogger('sibi_rgb_normal_6.log', append=False)
tensorboard = TensorBoard(log_dir='./sibi_rgb_normal_6_tf-logs')
callbacks_list = [checkpoint,best_checkpoint, csv_logger, tensorboard]


# len(hf["train"])
# len(hf["validation"])
rgb_model.fit_generator(generator("train"), steps_per_epoch=180//BATCH, epochs=200, callbacks=callbacks_list,shuffle=True,validation_data = generator("validation"),validation_steps=60//BATCH)

score = rgb_model.predict_generator(generator("test"),steps=60)
np.save("sibi_rgb_normal_result_6",score)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

hf.close()
