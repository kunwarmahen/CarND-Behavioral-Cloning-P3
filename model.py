import numpy as np
from keras.layers.core import Dense, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.models import Sequential
import csv
import cv2
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

BATCH_SIZE = 128
EPOCHS = 3

def load_image(path):
    image = mpimg.imread(path)
    return image

def flip_image(image):
    flipped_image = cv2.flip(image, 1)
    return flipped_image

def generate_training_data(BATCH_SIZE, X_train, y_train):
    '''
    Generate training data
    '''
    while True:
        num_examples = len(y_train)
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            images = list()
            labels = list()
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            for k in range(len(batch_x)):
                image = load_image(batch_x[k])
                steering = float(batch_y[k])
                flip_option = np.random.randint(2)
                images.append(image)
                labels.append(steering)              
                if flip_option==1: # Randomly flip image
                    flipped_steering = steering * -1.0
                    images.append(flip_image(image))
                    labels.append(flipped_steering)                    
            yield shuffle(np.array(images, dtype=np.float64),np.array(labels, dtype=np.float64))


def normalize(X):
    '''
    Nomalize image
    '''
    return X/255.0 - 0.5

def load_driving_log(path):
    '''
    Load CSV and build X_train and y_train
    '''
    f = open(path)
    reader = csv.reader(f)
    X_train = []
    y_train = []
    for line in reader: # Split and import data
        center = line[0]
        left = line[1]
        right = line[2]
        steering_angle = line[3]
        X_train.append(center.strip())
        y_train.append(float(steering_angle.strip()))
        X_train.append(left.strip())
        y_train.append(float(steering_angle.strip()) + 0.2) # augment steering angle for left and right images
        X_train.append(right.strip())
        y_train.append(float(steering_angle.strip()) - 0.2)
        
    f.close()
    return X_train, y_train


drive_log = 'driving_log.csv'
X_train,y_train = load_driving_log(drive_log) # Load data
X_train, y_train = shuffle(X_train, y_train) # Shuffle data
print('Total Lines Loaded ', len(y_train))

X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.2, random_state=42) # Split data into train and cv data
num_examples = len(y_train)

print('Total Training Loaded ', num_examples)
print('Total CV Loaded ', len(y_cv))

# Build model
model = Sequential()
model.add(Cropping2D(cropping=((60, 20), (0, 0)),input_shape=(160, 320, 3))) # Strip off top and bottom of frames
model.add(Lambda(normalize)) # Normalize image
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.summary()

# Compile and train the model
model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])
history=model.fit_generator(generator=generate_training_data(BATCH_SIZE=BATCH_SIZE, X_train=X_train, y_train=y_train),samples_per_epoch=num_examples*2, nb_epoch=EPOCHS, validation_data=generate_training_data(BATCH_SIZE=BATCH_SIZE, X_train=X_cv, y_train=y_cv), nb_val_samples=len(y_cv), verbose=2)
model.save('model.h5')

print(history.history)
print('Accuracy is ' + str(history.history['acc'][-1]))
print('Validation accuracy is ' + str(history.history['val_acc'][-1]))