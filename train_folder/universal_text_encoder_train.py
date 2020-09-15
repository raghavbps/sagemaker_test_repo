import argparse
import json
import os
import tensorflow as tf
import boto3
import numpy as np
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.python.keras import regularizers
from awscli.errorhandler import ClientError
from test1 import printtest

# def load_embedder():
#     embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
#     return embed


def model(x_train, y_train, x_test, y_test):
    """Generate a simple model"""
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(512,)),
        keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        keras.layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=50, batch_size=256)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('\nTest accuracy:', test_acc)
    print(model.summary())

    return model


def _load_training_data(base_dir):
    """Load training data"""
    x_train = np.load(os.path.join(base_dir, 'x_train.npy'))
    y_train = np.load(os.path.join(base_dir, 'y_train.npy'))
    return x_train, y_train


def _load_testing_data(base_dir):
    """Load testing data"""
    x_test = np.load(os.path.join(base_dir, 'x_test.npy'))
    y_test = np.load(os.path.join(base_dir, 'y_test.npy'))
    return x_test, y_test


def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    return parser.parse_known_args()


if __name__ == "__main__":
    print("Started training again")
    printtest()
    args, unknown = _parse_args()

    print(unknown)
    print(args)
    print("Base Dir")
    print(args.train)

    train_data, train_labels = _load_training_data(args.train)
    print(train_data.shape)
#     embedder = load_embedder()
#     train_data = embedder(train_data)
    print(train_data.shape)
    eval_data, eval_labels = _load_testing_data(args.train)
#     eval_data = embedder(eval_data)

    spam_classifier = model(train_data, train_labels, eval_data, eval_labels)

    print("Training Over")
    print(args.current_host)
    print(args.hosts[0])
    print(args.sm_model_dir)
    if args.current_host == args.hosts[0]:
        print("Saving model")
        spam_classifier.save(os.path.join(args.sm_model_dir, 'my_model.h5'))
        s3 = boto3.resource('s3')
        my_bucket = s3.Bucket('sagemaker-test-raghavbps')
        s3_client = boto3.client('s3')
        try:
            response = s3_client.upload_file(os.path.join(args.sm_model_dir, 'my_model.h5'), 'sagemaker-test-raghavbps',
                                     'spam_classifier/own_model_1.h5')
        except ClientError as e:
            logging.error(e)
#         for my_bucket_object in my_bucket.objects.all():
#             print(my_bucket_object.key)
#         for my_bucket_object in my_bucket.objects.all():
#             print(my_bucket_object)