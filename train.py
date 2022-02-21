import os
if os.system('nvidia-smi') == 0:
    import setGPU
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_datasets as tfds
import glob
import sys
import argparse
import yaml
import csv
import kerop

import models


def normalize_img(image, label):
  """
  Normalizes images: `uint8` -> `float32`.
  """
  return tf.cast(image, tf.float32) / 255., label


def main(args):
    with open(args.config) as file:
    	our_config = yaml.safe_load(file)

    save_dir = our_config['save_dir']
    model_name = our_config['model']['name']
    model_file_path = os.path.join(save_dir, 'model_best.h5')

    # Prepare dataset
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)


    # quantization parameters
    # if 'quantized' in model_name:
    #     logit_total_bits = config["quantization"]["logit_total_bits"]
    #     logit_int_bits = config["quantization"]["logit_int_bits"]
    #     activation_total_bits = config["quantization"]["activation_total_bits"]
    #     activation_int_bits = config["quantization"]["activation_int_bits"]
    #     alpha = config["quantization"]["alpha"]
    #     use_stochastic_rounding = config["quantization"]["use_stochastic_rounding"]
    #     logit_quantizer = config["quantization"]["logit_quantizer"]
    #     activation_quantizer = config["quantization"]["activation_quantizer"]
    #     final_activation = bool(config['model']['final_activation'])

    model = getattr(models, model_name)()
    print(model)
    # print model summary
    print('#################')
    print('# MODEL SUMMARY #')
    print('#################')
    print(model.summary())
    print('#################')

    tf.keras.utils.plot_model(model,
                              to_file=f'{model_name}.png',
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
        epochs=6,
        validation_data=ds_test,
        callbacks=[ModelCheckpoint(model_file_path, monitor='val_loss', verbose=True, save_best_only=True)]

    )




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="baseline.yml", help="specify yaml config")

    args = parser.parse_args()

    main(args)