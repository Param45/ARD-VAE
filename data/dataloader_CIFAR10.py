import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float32')
import tensorflow.keras as keras
import copy
import os

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


class dataloader_cifar10():
    def __init__(self, dataset_name, t_stat_samples, batch_size=100):

        if dataset_name=='CIFAR10':
            ensure_cifar10_cached()
            (self.imgs_train, _), (_, _) = keras.datasets.cifar10.load_data()
            self.imgs_train = (self.imgs_train/255.0).astype(np.float32)
            self.val_data_count = (10000//batch_size)*batch_size
            self.train_data_count = ((self.imgs_train.shape[0] - self.val_data_count)//batch_size)*batch_size

        self.t_stat_samples = t_stat_samples
        self.batch_size = batch_size
        self.train_dataset = None
        self.t_stat_dataset = None
        self.val_dataset = None


    def create_dataset(self, data, batch_size):
        return tf.data.Dataset.from_tensor_slices(data).shuffle(data.shape[0]).batch(batch_size)


    def split_train_n_val_data(self):

        self.x_train = self.imgs_train[:self.train_data_count]
        self.x_val = self.imgs_train[self.train_data_count:self.train_data_count+self.val_data_count]

        print("Train Image Stats:")
        print("Number : ", self.x_train.shape[0])
        print("Min    : ", np.min(self.x_train))
        print("Max    : ", np.max(self.x_train))

        print("Validation Image Stats:")
        print("Number : ", self.x_val.shape[0])
        print("Min    : ", np.min(self.x_val))
        print("Max    : ", np.max(self.x_val))

        return self.x_train, self.x_val


    def sample_t_stat_n_train_data(self):

        # Sample train and T-Stat indices from the training data
        T_Stat_samples_indices = np.random.choice(self.x_train.shape[0], self.t_stat_samples, replace=False).tolist()
        Train_samples_indices = list(set(np.arange(self.x_train.shape[0]).tolist()).difference(set(T_Stat_samples_indices)))

        # Test the overlap between the training data and the T-Stat samples
        train_lag_indices_overlap = set(Train_samples_indices).intersection(set(T_Stat_samples_indices))
        assert len(train_lag_indices_overlap) == 0, "There is overlap between the training and T-Stat samples"

        # Data to be used in training GENs
        x_train_sgd = self.x_train[Train_samples_indices]
        x_train_t_stat = self.x_train[T_Stat_samples_indices]

        print("Input Image Stats:")
        print("Number : ", x_train_sgd.shape[0])
        print("Min    : ", np.min(x_train_sgd))
        print("Max    : ", np.max(x_train_sgd))

        print("T-Stat Image Stats:")
        print("Number : ", x_train_t_stat.shape[0])
        print("Min    : ", np.min(x_train_t_stat))
        print("Max    : ", np.max(x_train_t_stat))

        return x_train_sgd, x_train_t_stat


    def create_t_stat_n_train_dataset(self):

        x_train_sgd, x_train_t_stat = self.sample_t_stat_n_train_data()
        self.train_dataset = self.create_dataset(x_train_sgd, self.batch_size)
        self.t_stat_dataset = self.create_dataset(x_train_t_stat, self.batch_size)
        return self.train_dataset, self.t_stat_dataset

    def create_val_dataset(self):

        val_dataset = self.create_dataset(self.x_val, self.batch_size)
        return val_dataset