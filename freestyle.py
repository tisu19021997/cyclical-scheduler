"""A "pythonic" file version of the notebook I used"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from CyclicalLR import CyclicalLR
from OneCycleLR import OneCycle
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import Model

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

X_means = X_train.mean(axis=0)
X_stds = X_train.std(axis=0)

X_train_scaled = (X_train - X_means) / X_stds
X_test_scaled = (X_test - X_means) / X_stds


def test_scaler(start, end, policy='triangular', gamma=0.99994):
    plt.figure(figsize=(8, 3))
    arrs = []
    step_size = 10
    batch_size = step_size * 2
    epoch = 5
    iterations = 0
    lr = start

    if policy == 'triangular':
        scale_fn = lambda x: 1
    elif policy == 'triangular2':
        scale_fn = lambda x: 1 / (2 ** (x - 1))
    else:
        scale_fn = lambda x: gamma ** (x)

    for i in range(epoch):
        for _ in range(batch_size):

            iterations += 1
            cycle = np.floor(1 + iterations / (2 * step_size))
            x = np.abs(iterations / step_size - 2 * cycle + 1)

            lr = start + (end - start) * np.maximum(0, (1 - x))

            if 'triangular' in policy:
                scale_factor = scale_fn(cycle)
            else:
                scale_factor = scale_fn(iterations)

            arrs.append(lr * scale_factor)

    plt.plot(range(iterations), arrs)
    plt.title(policy)


test_scaler(start=0.001, end=0.006, policy='triangular')
test_scaler(start=0.001, end=0.006, policy='triangular2')
test_scaler(start=0.001, end=0.006, policy='exp_range', gamma=0.9998)

"""# Build a new model using the pretrained VGG16"""


def build_vgg16_model():
    keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    # Freeze all layers
    base_model.trainable = False

    # Add a model on top of the pretrained VGG16.
    inputs = keras.Input(shape=(32, 32, 3))
    outputs = base_model(inputs, training=False)
    x = keras.layers.Flatten()(outputs)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    # x = keras.layers.Dense(4096, activation='elu', kernel_initializer='he_normal')(x)
    # x = keras.layers.Dense(4096, activation='elu', kernel_initializer='he_normal')(x)
    x = keras.layers.Dense(10, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    optimizer = keras.optimizers.Nadam()

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model


def build_momentum_vgg16_model():
    keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    # Freeze all layers
    base_model.trainable = False

    # Add a model on top of the pretrained VGG16.
    inputs = keras.Input(shape=(32, 32, 3))
    outputs = base_model(inputs, training=False)
    x = keras.layers.Flatten()(outputs)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    # x = keras.layers.Dense(4096, activation='elu', kernel_initializer='he_normal')(x)
    # x = keras.layers.Dense(4096, activation='elu', kernel_initializer='he_normal')(x)
    x = keras.layers.Dense(10, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    optimizer = keras.optimizers.SGD(momentum=0.009)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model


def build_simple_model():
    keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)

    # Simple 3-layer model
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(32, 32, 3)),
        keras.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
        keras.layers.Dense(10, activation='softmax'),
    ])

    optimizer = keras.optimizers.Nadam(lr=base_lr)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


"""# Cyclical LR"""

# Define some constants
batch_size = 512
epochs = 5
step_size = 98
base_lr = 0.001
max_lr = 0.006

"""## Triangular Policy"""

model = build_momentum_vgg16_model()

clr = CyclicalLR(max_lr=max_lr, base_lr=base_lr, step_size=step_size)

history = model.fit(X_train_scaled, y_train,
                    batch_size=batch_size, epochs=epochs,
                    validation_data=(X_test_scaled, y_test),
                    callbacks=[clr])

clr.plot_lr((14, 3.5))

"""## Triangular2 Policy"""

model = build_momentum_vgg16_model()

clr = CyclicalLR(policy='triangular2', max_lr=max_lr, base_lr=base_lr, step_size=step_size, cyclical_momentum=True)

history = model.fit(X_train_scaled, y_train,
                    batch_size=batch_size, epochs=epochs,
                    validation_data=(X_test_scaled, y_test),
                    callbacks=[clr])

"""The LR is cut in half for each cycle (equals to 2 steps)"""

clr.plot_lr((14, 3.5))

"""## Exponential Range Policy"""

model = build_momentum_vgg16_model()

clr = CyclicalLR(policy='exp_range', gamma=0.9994, max_lr=max_lr, base_lr=base_lr, step_size=step_size,
                 cyclical_momentum=True)

history = model.fit(X_train_scaled, y_train,
                    batch_size=batch_size // 2, epochs=epochs,
                    validation_data=(X_test_scaled, y_test),
                    callbacks=[clr])

clr.plot_lr((14, 3.5))

"""It is also possible to reset the scheduler and start training again."""

clr._reset()

model.fit(X_train_scaled, y_train,
          batch_size=batch_size // 2, epochs=epochs,
          validation_data=(X_test_scaled, y_test),
          callbacks=[clr])

clr.plot_lr((14, 3.5))

"""# 1Cycle Policy using pre-trained VGG16"""

batch_size = 512
epochs = 1
max_lr = 0.006
min_div_factor = 1e7
steps_per_epoch = math.ceil(len(X_train_scaled) / batch_size)


def test_anneal(anneal, start, end, min):
    one_cycle = OneCycle(max_lr=max_lr, cyclical_momentum=True,
                         epochs=epochs, steps_per_epoch=steps_per_epoch,
                         anneal=anneal, min_div_factor=min_div_factor, inc_ratio=0.5)

    arr = []
    for i in range(100):
        arr.append(one_cycle.annealer(start, end, i / 100))

    for i in range(100):
        arr.append(one_cycle.annealer(end, min, i / 100))

    plt.plot(range(200), arr, label=anneal)
    plt.legend()


test_anneal('linear', 0.001, 0.006, 0.006 / 1e4)
test_anneal('cosine', 0.001, 0.006, 0.006 / 1e4)
test_anneal('exp', 0.001, 0.006, 0.006 / 1e4)

"""## Linear Annealing with 1Cycle Policy"""

model = build_momentum_vgg16_model()

print(f'Training with {steps_per_epoch} steps per epoch')

one_cycle = OneCycle(max_lr=max_lr, base_lr=base_lr, cyclical_momentum=True,
                     epochs=epochs, steps_per_epoch=steps_per_epoch,
                     anneal='linear', min_div_factor=min_div_factor, inc_ratio=0.5)

history = model.fit(X_train_scaled, y_train,
                    batch_size=batch_size, epochs=epochs,
                    validation_data=(X_test_scaled, y_test),
                    callbacks=[one_cycle])

"""Let's plot the learning rate and see if it follows the policy."""

one_cycle.plot_lr((10, 3.5))

"""## Cosine Annealing with 1Cycle Policy"""

model = build_momentum_vgg16_model()

print(f'Training with {steps_per_epoch} steps per epoch')

one_cycle = OneCycle(max_lr=max_lr, base_lr=base_lr, cyclical_momentum=True,
                     epochs=epochs, steps_per_epoch=steps_per_epoch,
                     anneal='cosine', min_div_factor=min_div_factor)

history = model.fit(X_train_scaled, y_train,
                    batch_size=batch_size, epochs=epochs,
                    validation_data=(X_test_scaled, y_test),
                    callbacks=[one_cycle])

one_cycle.plot_lr()

"""## Exponentially Annealing with 1Cycle Policy"""

model = build_momentum_vgg16_model()

print(f'Training with {steps_per_epoch} steps per epoch')

one_cycle = OneCycle(max_lr=max_lr, cyclical_momentum=True,
                     epochs=epochs, steps_per_epoch=steps_per_epoch,
                     anneal='exp', min_div_factor=min_div_factor)

history = model.fit(X_train_scaled, y_train,
                    batch_size=batch_size, epochs=epochs,
                    validation_data=(X_test_scaled, y_test),
                    callbacks=[one_cycle])

"""The **momentum** seems to be like linear because its value is too close to 1."""

one_cycle.plot_lr()
