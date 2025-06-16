import numpy as np
import struct
import tensorflow as tf

class FashionMNISTLoader:
    def __init__(self, img_path, lbl_path):
        self.images = self._load_images(img_path)
        self.labels = self._load_labels(lbl_path)

    def _load_images(self, path):
        with open(path, 'rb') as f:
            _, num, rows, cols = struct.unpack(">IIII", f.read(16))
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data.reshape(num, rows, cols, 1).astype(np.float32) / 255.0

    def _load_labels(self, path):
        with open(path, 'rb') as f:
            _, num = struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)

    def get_dataset(self, batch_size=64, shuffle=True):
        ds = tf.data.Dataset.from_tensor_slices((self.images, self.labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=10000)
        return ds.batch(batch_size)
