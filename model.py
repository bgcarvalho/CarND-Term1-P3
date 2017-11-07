import numpy as np
import pandas as pd
import matplotlib.image as mpimg
#from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Dropout, Lambda
from keras.layers import MaxPooling2D
from keras.layers import Cropping2D
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras import __version__ as keras_version

# traning was recorded in 8 diferent times
training_files = 8

# CSV columns
c = ['center', 'left', 'right', 'steer', 'throttle', 'break', 'speed']
samples = np.empty((0, 4))
nlabels = 0
#for s in range(1, training_files):
for s in [3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15]:
    # read CSV file with steering angle
    file = "./data/driving_{:0>2d}.csv".format(s)
    csvdata = pd.read_csv(file, header=None, names=c)

    # remove too low speed == car almost stopped
    csvdata = csvdata[csvdata['speed'] >= 0.05]

    nl = csvdata.shape[0]
    nlabels += nl
    n_ = np.vectorize(lambda x: 'driving_{:0>2d}'.format(s))

    # temporary var
    tmp = csvdata.iloc[:, [0, 3]].values
    tmp = np.concatenate((tmp, np.zeros((nl, 1))), axis=1)
    tmp = np.concatenate((tmp, n_(np.ones((nl, 1)))), axis=1)

    # duplicate (flip images)
    tmp = np.concatenate((tmp, tmp), axis=0)
    tmp[:nl, 2] = 0  # first half
    tmp[nl:, 2] = 1  # second half

    # concatenate
    samples = np.concatenate((samples, tmp), axis=0)

    # left
    left = csvdata.iloc[:, [1, 3]].values
    left = np.concatenate((left, np.zeros((nl, 1))), axis=1)
    left = np.concatenate((left, n_(np.ones((nl, 1)))), axis=1)
    left[:, 1] += 0.225  # adjust steering

    # concatenate
    samples = np.concatenate((samples, left), axis=0)

    # right
    right = csvdata.iloc[:, [2, 3]].values
    right = np.concatenate((right, np.zeros((nl, 1))), axis=1)
    right = np.concatenate((right, n_(np.ones((nl, 1)))), axis=1)
    right[:, 1] -= 0.225  # adjust steering

    # concatenate
    samples = np.concatenate((samples, right), axis=0)

# total samples
print(samples.shape)

# original shape
owidth = 320
oheight = 160

# crop
stripetop = 60
stripebot = 20


def generator(samples, batch_size=32):
    num_samples = len(samples)
    #
    # Loop forever so the generator never terminates
    #  but it only executes when code calls "next()"
    #
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                filep = batch_sample[0].split('/')
                name = './data/' + batch_sample[3] + '/' + filep[-1]

                # use CV2 to read images
                # center_image = cv2.imread(name)

                # use Matplotlib to read images
                center_image = mpimg.imread(name, format='jpeg')
                center_angle = float(batch_sample[1])

                # flip image
                if batch_sample[2]:
                    center_image = np.fliplr(center_image)
                    center_angle = -center_angle

                # train data
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


# split 80% training, 20% validation
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Python automatically interprets the function as generator
#  and WON'T execute it right away
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# print(train_generator)
# print(len(train_samples))
# print(samples[0])



# =============================================================
# MODEL ARCHITECTURE
# =============================================================
# Create a sequential model => linear stack of layers
#
model = Sequential()

# CROP images
i_s = (oheight, owidth, 3)
crop = (stripetop, stripebot)
model.add(Cropping2D(cropping=(crop, (0, 0)), input_shape=i_s))

# normalization and mean centering
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

# Convolutional section
model.add(Conv2D(24, (7, 7), strides=(2, 2), activation='relu'))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(60, (5, 5), activation='relu'))
model.add(Conv2D(72, (3, 3), activation='relu'))

# use dropout to avoid overfitting
model.add(Dropout(0.5))

# flatten the cube to a dense layer
model.add(Flatten())

# dense layers
model.add(Dense(72, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(48, activation='relu'))

# final layer named "steer_angle", so it can be used elsewhere (transfer)
model.add(Dense(1, name='steer_angle'))

# inspect model layers
print(model.summary())





# =============================================================
# TRAINING
# =============================================================

cb_earlystop = EarlyStopping(min_delta=1e-3, patience=2, verbose=1)
cb = [cb_earlystop]
#cb = []

# learning rate
opt = optimizers.Adam(lr=0.001, decay=1e-4)

# compile model
model.compile(loss='mse', optimizer=opt)

# train
# samples_per_epoch=len(train_samples),
history = model.fit_generator(train_generator,
                              steps_per_epoch=len(train_samples) / 320,
                              validation_data=validation_generator,
                              validation_steps=len(validation_samples) / 320,
                              epochs=10,
                              verbose=1,
                              callbacks=cb)
# save to disk
model.save('model.h5')

print('')
print('model.h5 saved')
print('')

show_hist = True

if show_hist:
    print(history.history)


show_plots = False

if show_plots:
    import matplotlib.pyplot as plt
    # print(history.history)
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('epochs')
    plt.ylabel('')
    plt.title('Model training')
    plt.legend(['Training loss', 'Validation loss'])
