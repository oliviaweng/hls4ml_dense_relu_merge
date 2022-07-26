import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_datasets as tfds
import argparse
import yaml

import models


def normalize_img(image, label):
    """
    Normalizes images: `uint8` -> `float32`.
    """
    return tf.cast(image, tf.float32) / 255., label


def expand_img_dim(image, label):
    """
    Make sure images have shape (28, 28, 1)
    """
    return tf.expand_dims(image, -1), label


def main(args):
    with open(args.config) as file:
    	our_config = yaml.safe_load(file)

    save_dir = our_config['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    model_name = our_config['model']['name']
    model_file_path = os.path.join(save_dir, 'model_best.h5')

    num_epochs = our_config['model']['epochs']

    # Prepare dataset
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
        

    ds_train = ds_train.map(
        normalize_img, 
        num_parallel_calls=tf.data.AUTOTUNE
    )
    if 'conv2d' in model_name:
        # Make sure images have shape (28, 28, 1)
        ds_train = ds_train.map(expand_img_dim)
        print(ds_train)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(
        normalize_img, 
        num_parallel_calls=tf.data.AUTOTUNE
    )
    if 'conv2d' in model_name:
        # Make sure images have shape (28, 28, 1)
        ds_test = ds_test.map(expand_img_dim)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)


    # quantization parameters
    if 'quantized' in model_name:
        logit_total_bits = our_config["quantization"]["logit_total_bits"]
        logit_int_bits = our_config["quantization"]["logit_int_bits"]
        activation_total_bits = our_config["quantization"]["activation_total_bits"]
        activation_int_bits = our_config["quantization"]["activation_int_bits"]
        model = getattr(models, model_name)(
            logit_total_bits, 
            logit_int_bits, 
            activation_total_bits, 
            activation_int_bits
        )
    else:
        model = getattr(models, model_name)()
        
    print(model)
    # print model summary
    print('#################')
    print('# MODEL SUMMARY #')
    print('#################')
    print(model.summary())
    print('#################')

    tf.keras.utils.plot_model(model,
                              to_file=f'{save_dir}/{model_name}.png',
                              show_shapes=True,
                              show_dtype=True,
                              show_layer_names=True,
                              rankdir="TB",
                              expand_nested=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    model.fit(
        ds_train,
        epochs=num_epochs,
        validation_data=ds_test,
        callbacks=[ModelCheckpoint(model_file_path, monitor='val_loss', verbose=True, save_best_only=True)]

    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="baseline.yml", help="specify yaml config")

    args = parser.parse_args()

    main(args)