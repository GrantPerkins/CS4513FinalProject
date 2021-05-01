import os
import json
from shutil import copyfile
from pathlib import Path
import tensorflow as tf
import smdistributed.dataparallel.tensorflow as dist


def config_gpus():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    tf.config.experimental.set_visible_devices(gpus[dist.local_rank()], 'GPU')


def get_hyperparameters():
    with open("/opt/ml/input/config/hyperparameters.json", 'r') as f:
        file = json.load(f)
        mnist_epochs = int(file["epochs"])
        mnist_batch_size = int(file["batch_size"])
        mnist_learning_rate = float(file["learning_rate"])
    return mnist_epochs, mnist_batch_size, mnist_learning_rate


def get_dataset(mnist_batch_size):
    data_slice = 'mnist-%d.npz' % dist.rank()
    keras_path = os.path.join(str(Path.home()), ".keras/datasets/")
    os.makedirs(keras_path, exist_ok=True)
    copyfile("/opt/ml/input/data/training/" + data_slice, keras_path + data_slice)
    (mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data(path=data_slice)
    mnist_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
         tf.cast(mnist_labels, tf.int64))
    )
    mnist_dataset = mnist_dataset.repeat().shuffle(100000).batch(mnist_batch_size)
    return mnist_dataset


def create_model(mnist_learning_rate):
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

    # SMDataParallel: dist.size()
    # LR for 8 node run : 0.000125
    # LR for single node run : 0.001
    mnist_optimizer = tf.optimizers.Adam(mnist_learning_rate * dist.size())
    tf.train.Checkpoint(model=mnist_model, optimizer=optimizer)
    return mnist_model, mnist_loss, mnist_optimizer


@tf.function
def training_step(images, labels, first_batch):
    with tf.GradientTape() as tape:
        probs = mnist_model(images, training=True)
        loss_value = mnist_loss(labels, probs)

    # SMDataParallel: Wrap tf.GradientTape with SMDataParallel's DistributedGradientTape
    tape = dist.DistributedGradientTape(tape)

    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if first_batch:
        # SMDataParallel: Broadcast model and optimizer variables
        dist.broadcast_variables(model.variables, root_rank=0)
        dist.broadcast_variables(optimizer.variables(), root_rank=0)

    # SMDataParallel: all_reduce call
    loss_value = dist.oob_allreduce(loss_value)  # Average the loss across workers
    return loss_value


def train(mnist_epochs):
    for batch, (images, labels) in enumerate(dataset.take(mnist_epochs // dist.size())):
        loss_value = training_step(images, labels, batch == 0, model, loss, optimizer)
        if batch % 50 == 0 and dist.rank() == 0:
            print('Step #%d\tLoss: %.6f' % (batch, loss_value))


if __name__ == "__main__":
    dist.init()
    config_gpus()
    print("Worker number:", dist.rank())
    epochs, batch_size, learning_rate = get_hyperparameters()
    model, loss, optimizer = create_model(learning_rate)
    dataset = get_dataset(batch_size)
    train(epochs)
    # SMDataParallel: Save checkpoints only from master node.
    if dist.rank() == 0:
        checkpoint_dir = "/opt/ml/model"
        model.save(os.path.join(checkpoint_dir, '1'))
