# Grant Perkins CS 4513 Final Project

In this project, I developed a distributed machine learning solution with AWS SageMaker.

Features:
 - Distributed computing:
   + One virtual computer runs this Jupyter Notebook
   + One virtual computer stores dataset
   + N virtual computers run the training job
   + One virtual computer runs the inference job (small load)
 - Machine Learning:
   + Trains a custom neural network to recognize digits in the MNIST dataset
   + Customizable hyperparameters (epochs, batch size, learning rate)
   + Inference on whatever image (classification)

I started this project on 4/30/2021, for CS 4513.

## Section 1: Data

In this section, I download the dataset, and upload it to a computer controlled by AWS S3, a long-term data service.

## Section 2: Training

Now that the data is in the correct place, I can train a neural network for the dataset. I do this on N separate computers. N can be any natural number.

