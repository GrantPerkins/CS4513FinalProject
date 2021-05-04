# Grant Perkins CS 4513 Final Project

Grant Perkins (gcperkins@wpi.edu)

In this project, I developed a distributed machine learning solution with TensorFlow and AWS SageMaker.

Features:
 - Distributed computing:
   + One virtual computer runs this Jupyter Notebook
   + One virtual computer stores dataset (in AWS S3)
   + Multiple virtual computers run the training job
 - Machine Learning:
   + Trains a convolutional neural network to recognize digits in the MNIST dataset
   + Customizable hyperparameters (epochs, batch size, learning rate)

I started this project on 4/30/2021, for CS 4513.

## Project details

### Files
`mnist.ipynb` contains a Jupyter notebook to run in SageMaker.
`output.ipynb` shows the expected output if your run all the blocks from mnist.ipynb.
`mnist.py` contains all of the actual machine learning code

### Section 1: Data

In this section, I download the dataset, and upload it to a computer controlled by AWS S3, a long-term data service.
The dataset is split between all of the workers, so N slices of the dataset are uploaded to S3.

### Section 2: Training

This step runs `mnist.py` across many workers. Data is downloaded from AWS S3. Each worker has a different "rank", and 
this rank is used to determine which slice of the dataset the worker should use. The rank 0 worker is considered the
master. This master worker prints metrics every 50 epochs. `mnist.py` trains a custom convolutional neural network. It
calculates loss using cross entropy. I am using an Adam Optimizer.

The majority of modifications between a normal MNIST model and mine is in the `training_step` function. This function 
leverages a SageMaker API to calculate loss from all of the workers in the master worker.

