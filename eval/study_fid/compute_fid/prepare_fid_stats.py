"""Helper script to pre-compute reference statistics (mu, sigma) from real datasets
using Inception-v3 for Fréchet Inception Distance (FID) computation.
"""
import os
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
tf1.disable_v2_behavior()
import get_fid


def prepare_mnist_stats(save_dir, sample_count=10000):
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "fid_stats_mnist.npz")
    if os.path.exists(out_path):
        print(f"MNIST FID stats already exist at: {out_path}")
        return out_path

    print("Computing Inception reference statistics for MNIST...")
    (x_train, _), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train[:sample_count]
    # Pad from 28x28 to 32x32
    padded = np.pad(x_train, ((0, 0), (2, 2), (2, 2)), mode='constant')
    # Repeat grayscale to 3 channels (RGB) and ensure uint8/float in [0, 255]
    rgb = np.repeat(np.expand_dims(padded, axis=-1), 3, axis=-1).astype(np.float32)

    tf1.reset_default_graph()
    inception_path = get_fid.check_or_download_inception(None)
    get_fid.create_inception_graph(str(inception_path))

    gpu_options = tf1.GPUOptions(per_process_gpu_memory_fraction=0.3)
    with tf1.Session(config=tf1.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf1.global_variables_initializer())
        mu, sigma = get_fid.calculate_activation_statistics(rgb, sess, batch_size=100, verbose=True)

    np.savez(out_path, mu=mu, sigma=sigma)
    print(f"Saved MNIST FID statistics to {out_path}")
    return out_path


def ensure_cifar10_cached():
    import urllib.request
    import tarfile
    keras_dir = os.path.expanduser('~/.keras/datasets')
    os.makedirs(keras_dir, exist_ok=True)
    tar_path = os.path.join(keras_dir, 'cifar-10-batches-py.tar.gz')
    extracted_dir = os.path.join(keras_dir, 'cifar-10-batches-py')

    if not os.path.exists(extracted_dir) and not os.path.exists(tar_path):
        url = "https://huggingface.co/datasets/liangnanying/cifar-10-python/resolve/main/cifar-10-python.tar.gz"
        print("Downloading CIFAR-10 from fast CDN mirror (completes in seconds)...")
        try:
            urllib.request.urlretrieve(url, tar_path)
            print("Downloaded! Extracting archive...")
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=keras_dir)
            print("CIFAR-10 successfully cached in ~/.keras/datasets/!")
        except Exception as e:
            print("Note: Fast mirror download skipped, using default Keras source:", e)


def prepare_cifar10_stats(save_dir, sample_count=10000):
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "fid_stats_cifar10_train.npz")
    if os.path.exists(out_path):
        print(f"CIFAR-10 FID stats already exist at: {out_path}")
        return out_path

    print("Computing Inception reference statistics for CIFAR-10...")
    ensure_cifar10_cached()
    (x_train, _), _ = tf.keras.datasets.cifar10.load_data()
    x_train = x_train[:sample_count].astype(np.float32)

    tf1.reset_default_graph()
    inception_path = get_fid.check_or_download_inception(None)
    get_fid.create_inception_graph(str(inception_path))

    gpu_options = tf1.GPUOptions(per_process_gpu_memory_fraction=0.3)
    with tf1.Session(config=tf1.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf1.global_variables_initializer())
        mu, sigma = get_fid.calculate_activation_statistics(x_train, sess, batch_size=100, verbose=True)

    np.savez(out_path, mu=mu, sigma=sigma)
    print(f"Saved CIFAR-10 FID statistics to {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-compute FID stats for datasets")
    parser.add_argument("--dataset", type=str, choices=["MNIST", "CIFAR10", "ALL"], default="ALL")
    parser.add_argument("--out_dir", type=str, default="fid_stats")
    parser.add_argument("--samples", type=int, default=10000)
    args = parser.parse_args()

    if args.dataset in ["MNIST", "ALL"]:
        prepare_mnist_stats(args.out_dir, args.samples)
    if args.dataset in ["CIFAR10", "ALL"]:
        prepare_cifar10_stats(args.out_dir, args.samples)
