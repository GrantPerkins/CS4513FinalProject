import os
import json
from shutil import copyfile
from pathlib import Path
import tensorflow as tf
import smdistributed.dataparallel.tensorflow as dist


def config_gpus():
    """
    Multiple GPUs are being used across the workers. This function initializes GPUs for use.
    :return: None
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_visible_devices(gpus[dist.local_rank()], 'GPU')


def get_hyperparameters():
    """
    Read the hyperparameters specified in the Jupyter Notebook
    :return: epochs, batch size, and learning rate
    """
    with open("/opt/ml/input/config/hyperparameters.json", 'r') as f:
        # SageMaker writes the hyperparameters as a json file to the same place every time
        hyper_file = json.load(f)
        mnist_epochs = int(hyper_file["epochs"])
        mnist_batch_size = int(hyper_file["batch_size"])
        mnist_learning_rate = float(hyper_file["learning_rate"])
    return mnist_epochs, mnist_batch_size, mnist_learning_rate


def get_dataset(mnist_batch_size):
    """
    get the dataset slice that corresponds to the worker
    :param mnist_batch_size: size of the batch. a hyperparameter
    :return: a tf.data.Dataset object
    """
    data_slice_name = 'mnist-%d.npz' % dist.rank()
    keras_path = os.path.join(str(Path.home()), ".keras/datasets/")  # keras expects data to be here
    os.makedirs(keras_path, exist_ok=True)
    copyfile("/opt/ml/input/data/training/" + data_slice_name,
             keras_path + data_slice_name)  # copy file from magic S3 location to keras path
    (mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data(path=data_slice_name)
    # need to convert from 0-255 int pixel format to float, labels to ints
    data_slice = tf.data.Dataset.from_tensor_slices(
        (tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
         tf.cast(mnist_labels, tf.int64))
    ).repeat().shuffle(100000).batch(mnist_batch_size)  # repeat infinitely, shuffle, and set batch size
    return data_slice


def create_model(mnist_learning_rate):
    """
    Creates a new keras model for learning
    :param mnist_learning_rate: learning rate for the Adam Optimizer
    :return: mode, loss function, and optimizer
    """
    # neural net
    mnist_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
        tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    mnist_loss = tf.losses.SparseCategoricalCrossentropy()
    # learning rate is proportional to number of workers
    mnist_optimizer = tf.optimizers.Adam(mnist_learning_rate * dist.size())
    return mnist_model, mnist_loss, mnist_optimizer


@tf.function
def training_step(images, labels, batch):
    """
    Runs one training step, with considerations for the distributed nature of the job.
    :param images: images, sorted
    :param labels: labels for images, same order
    :param batch: batch number. variables must be broadcasted on first batch
    :return: loss for this step across all workers
    """
    with tf.GradientTape() as tape:
        # record losses for automatic differentiation (learning)
        probs = model(images, training=True)
        loss_value = loss(labels, probs)

    # need to wrap tape with more tape, because it is distributed
    tape = dist.DistributedGradientTape(tape)

    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))  # learn on this worker

    if batch == 0:
        # workers share learning broadcast by broadcasting variables. one set of variables across workers.
        dist.broadcast_variables(model.variables, root_rank=0)
        dist.broadcast_variables(optimizer.variables(), root_rank=0)

    # all loss values from all workers reduced to one loss value
    loss_value = dist.oob_allreduce(loss_value)  # Average the loss across workers
    return loss_value


def train(mnist_epochs):
    """
    Train CNN
    :param mnist_epochs: number of training steps to run for
    :return: None
    """
    for batch, (images, labels) in enumerate(dataset.take(mnist_epochs // dist.size())):
        loss_value = training_step(images, labels, batch)
        # print loss every 50 epochs in master worker
        if batch % 50 == 0 and dist.rank() == 0:
            print('Step #%d\tLoss: %.6f' % (batch, loss_value))


if __name__ == "__main__":
    # runs training, but distributed
    dist.init()
    config_gpus()
    print("Worker number:", dist.rank())
    epochs, batch_size, learning_rate = get_hyperparameters()
    model, loss, optimizer = create_model(learning_rate)
    dataset = get_dataset(batch_size)
    train(epochs)

    # save model as master
    if dist.rank() == 0:
        checkpoint_dir = "/opt/ml/model"
        model.save(os.path.join(checkpoint_dir, '1'))
