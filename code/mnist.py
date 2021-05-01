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


@tf.function
def training_step(images, labels, first_batch):
    with tf.GradientTape() as tape:
        probs = model(images, training=True)
        loss_value = loss(labels, probs)

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


dist.init()
config_gpus()
print("Worker number:", dist.rank())
epochs, batch_size, learning_rate = get_hyperparameters()
    
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
    tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])
loss = tf.losses.SparseCategoricalCrossentropy()
optimizer = tf.optimizers.Adam(learning_rate * dist.size())
dataset = get_dataset(batch_size)

# train
for batch, (images, labels) in enumerate(dataset.take(epochs // dist.size())):
    batch_loss_value = training_step(images, labels, batch == 0)
    if batch % 50 == 0 and dist.rank() == 0:
        print('Step #%d\tLoss: %.6f' % (batch, batch_loss_value))

# SMDataParallel: Save checkpoints only from master node.
if dist.rank() == 0:
    checkpoint_dir = "/opt/ml/model"
    print("Saving to",checkpoint_dir)
    model.save(os.path.join(checkpoint_dir, '1'))
