from sklearn.preprocessing import OneHotEncoder
import numpy as np
import scipy.io
from torch.utils.data import Dataset

class CUBSequence(Dataset):

    def __init__(self, x_set, y_set, transform=None):
        self.x, self.y = x_set, y_set
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = {'feature': self.x[idx], 'label' : self.y[idx]}
        if self.transform:
            sample = self.transform(sample)
        return sample
    
class CUB_data():
    def __init__(self, att_fpath, res_fpath):
        self.load_atts(att_fpath)
        self.load_features(res_fpath)
        self.make_sets()
        
    def load_atts(self, fpath):
        'Load attributes'
        att_splits = scipy.io.loadmat(fpath)
        self.class_attributes = att_splits['att'].T
        self.train_idx = att_splits['train_loc'].reshape(-1) - 1
        self.val_idx = att_splits['val_loc'].reshape(-1) - 1
        self.test_seen_idx = att_splits['test_seen_loc'].reshape(-1) - 1
        self.test_unseen_idx = att_splits['test_unseen_loc'].reshape(-1) -1
        self.trainval_idx = att_splits['trainval_loc'].reshape(-1) - 1

    def load_features(self, fpath):
        res101 = scipy.io.loadmat(fpath)
        self.features = res101['features'].T
        self.labels = res101['labels'] - 1
        self.onehot_labels = OneHotEncoder(sparse=False).fit_transform(self.labels)
        
    def make_sets(self):
        self.sets = {}
        self.sets['train_X'] = np.array([self.features[i] for i in self.train_idx], dtype='float32')
        self.sets['val_X'] = np.array([self.features[i] for i in self.val_idx], dtype='float32')
        self.sets['test_seen_X'] = np.array([self.features[i] for i in self.test_seen_idx], dtype='float32')
        self.sets['test_unseen_X'] = np.array([self.features[i] for i in self.test_unseen_idx], dtype='float32')
        self.sets['train_a'] = np.array([self.labels[i] for i in self.train_idx], dtype='float32')
        self.sets['val_a'] = np.array([self.labels[i] for i in self.val_idx], dtype='float32')
        self.sets['test_seen_a'] = np.array([self.labels[i] for i in self.test_seen_idx], dtype='float32')
        self.sets['test_unseen_a'] = np.array([self.labels[i] for i in self.test_unseen_idx], dtype='float32')
        self.sets['trainval_X'] = np.array([self.features[i] for i in self.trainval_idx], dtype='float32')
        self.sets['trainval_a'] = np.array([self.labels[i] for i in self.trainval_idx], dtype='float32')
        
    def get_gen(self, mode, **args):
        mode = mode.lower()
        if mode in ['train', 'val', 'trainval', 'test_seen', 'test_unseen']:
            return CUBSequence(self.sets[mode+'_X'], self.sets[mode+'_a'], **args)
        else:
            print(mode, 'is an incorrect mode.')