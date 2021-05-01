import os
import json
from shutil import copyfile
from pathlib import Path
import tensorflow as tf
import smdistributed.dataparallel.tensorflow as dist

dist.init()
print("Dist size", dist.size(), "rank:", dist.rank())

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.experimental.set_visible_devices(gpus[dist.local_rank()], 'GPU')

with open("/opt/ml/input/config/hyperparameters.json", 'r') as f:
    file = json.load(f)
    epochs = int(file["epochs"])
    batch_size = int(file["batch_size"])
    learning_rate = float(file["learning_rate"])

data_slice = 'mnist-%d.npz' % dist.rank()
keras_path = os.path.join(str(Path.home()), ".keras/datasets/")
input_path = "/opt/ml/input/data/training/"
os.makedirs(keras_path, exist_ok=True)
copyfile(input_path + data_slice, keras_path + data_slice)
(mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data(path=data_slice)

dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
     tf.cast(mnist_labels, tf.int64))
)
dataset = dataset.repeat().shuffle(100000).batch(batch_size)

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
loss = tf.losses.SparseCategoricalCrossentropy()
optimizer = tf.optimizers.Adam(learning_rate * dist.size())
checkpoint_dir = os.environ['SM_MODEL_DIR']
checkpoint = tf.train.Checkpoint(model=mnist_model, optimizer=optimizer)


@tf.function
def training_step(images, labels, first_batch):
    with tf.GradientTape() as tape:
        probs = mnist_model(images, training=True)
        loss_value = loss(labels, probs)

    # SMDataParallel: Wrap tf.GradientTape with SMDataParallel's DistributedGradientTape
    tape = dist.DistributedGradientTape(tape)

    grads = tape.gradient(loss_value, mnist_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, mnist_model.trainable_variables))

    if first_batch:
        # SMDataParallel: Broadcast model and optimizer variables
        dist.broadcast_variables(mnist_model.variables, root_rank=0)
        dist.broadcast_variables(optimizer.variables(), root_rank=0)

    # SMDataParallel: all_reduce call
    return dist.oob_allreduce(loss_value)  # Average the loss across workers


for batch, (images, labels) in enumerate(dataset.take(epochs // dist.size())):
    batch_loss_value = training_step(images, labels, batch == 0)

    if batch % 50 == 0 and dist.rank() == 0:
        print('Step #%d\tLoss: %.6f' % (batch, batch_loss_value))

# SMDataParallel: Save checkpoints only from master node.
if dist.rank() == 0:
    mnist_model.save(os.path.join(checkpoint_dir, '1'))
