import numpy as np
class data_loader(object):
    def __init__(self,features, train, val, test):
        self.features = features
        self.train = list(train)
        self.test = list(test)
        self.val = list(val)

        self.orignal_train = list(train)
        self.train_index = 0
        self.val_index = 0
        self.test_index = 0

    def get_train_batch(self, batch_size):
        end_index = batch_size+self.train_index
        if end_index > len(self.train):
            end_index = len(self.train)

        batch = self.train[self.train_index:end_index]
        self.train_index = end_index
        return batch

    def get_val_batch(self, batch_size):
        end_index = batch_size + self.val_index
        if end_index > len(self.val):
            end_index = len(self.val)

        batch = self.val[self.val_index:end_index]
        self.val_index = end_index
        return batch

    def get_test_batch(self, batch_size):
        end_index = batch_size + self.test_index
        if end_index > len(self.test):
            end_index = len(self.test)

        batch = self.test[self.test_index:end_index]
        self.test_index = end_index
        return batch

    def train_end(self, shuffle=False):
        if self.train_index == len(self.train):
            self.train_index = 0
            if shuffle:
                np.random.shuffle(self.train)
            else:
                self.train = self.orignal_train
            return True
        return False

    def val_end(self):
        if self.val_index == len(self.val):
            self.val_index = 0
            return True
        return False

    def test_end(self):
        if self.test_index == len(self.test):
            self.test_index = 0
            return True
        return False