# Behavioral Cloning** 

## Project Report

This project consists of using a car simulator to record driving data 
(steering angle, throttle, break and speed) together with track images, 
feed this data into and deep learning network and clone the behavior of the
driver, as the predictions of the network enables the simulator to run 
autonomously.

The steps of this project are the following:

- Use the simulator to collect driving data
- Create a convolutional neural network (CNN) with Keras and TensorFlow
- Train and validate the model with simulator data
- Test the model to successfully drive around the track without leaving the road
- Descrive and discuss results

### Files

This project has four main files:

- [`model.py`](./model.py) - Keras/TensorFlow architecture implemented in Python
- [`drive.py`](./drive.py) - provided WSGI server to send predictions back to the simulator,
and also a PI controller to filter throttle and speed
- `model.h5` - Generated model in HDF format enabling the predictions to run elsewhere
- [`video.mp4`](./) - video created with the simulator in autonomous mode
- [`report.md`](./) - this report.

To run the simulator autonomously, execute:

```python drive.py model.h5```

The `model.py` is commented and follows PEP8. The model uses Python generators,
not only by a rubric requirement, but also as be functional on available 
resources. Loading the whole dataset to memory is not feasible.

### Model Architecture

This project uses an architecture consisting of:

- 5 convolutional layers (concern with image patterns)
- 4 fully connected layers (concern with steering)
- activations are RELU
- a Dropout layer is used to add non-linearity and avoid overfitting 

This arrangement of layers is suggested following the success of 
[NVIDIA Team](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)
with self-driving cars. Although different in image size, the NVIDIA model
poses a great reference for this project.

Keras utilities are used to crop, normalize and center pixels. By doing this
in model layers, there no need to make any change in `drive.py`.

|Layer (type)               | Output Shape            | Param #  |
|---------------------------|-------------------------|----------|
|cropping2d_1 (Cropping2D)  |(None, 80, 320, 3)       | 0        |
|lambda_1 (Lambda)          |(None, 80, 320, 3)       | 0        |
|conv2d_1 (Conv2D)          |(None, 37, 157, 24)      | 3552     | 
|conv2d_2 (Conv2D)          |(None, 17, 77, 36)       | 21636    |
|conv2d_3 (Conv2D)          |(None, 7, 37, 48)        | 43248    |
|conv2d_4 (Conv2D)          |(None, 3, 33, 60)        | 72060    |
|conv2d_5 (Conv2D)          |(None, 1, 31, 72)        | 38952    |
|dropout_1 (Dropout)        |(None, 1, 31, 72)        | 0        |
|flatten_1 (Flatten)        |(None, 2232)             | 0        |
|dense_1 (Dense)            |(None, 48)               | 107184   |
|dense_2 (Dense)            |(None, 32)               | 1568     |
|dense_3 (Dense)            |(None, 16)               | 528      |
|steer_angle (Dense)        |(None, 1)                | 17       |

Total params: 288,745
Trainable params: 288,745
Non-trainable params: 0

For training was chosen to use 80% of samples, the 20% left for validation.
The optimizer is Adam with learning rate of 0.0007, a little lower than its
default value of 0.001.

The training data was recorded in distinct session, evaluating the model each
time, so it has generated 8 zip files. These recording sessions contain:

- about 2 full laps driving with the car in the center of the road
- 1 lap driving in zig-zag, exposing the model to "new" images
- recovering training in corners, driving the car from the edges in direction
the center of the road
- 1 full lap driving clockwise

So with this dataset we build an implicit constrain that the car should not
go off the road.

| | |
|-|-|
|[Crossing bridge - Center image](images/center_2017_10_26_22_00_37_383.png)|[Driving clockwise - Right image](images/right_2017_11_02_23_32_18_506.png)|


[Driving autonomously](images/autonomous1.png)


