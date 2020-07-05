from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tabulate import tabulate
import random
import numpy as np

from pdb import set_trace as bp

class Database():
    "Simulated data structure"
    def __init__(self, data_num, compare_num):
        self.embs = np.ndarray((data_num,128), dtype=float)
        self.labels = []
        self.indices = 0
        self.thresholds = []
        self.compare_num = compare_num
        self.class_dict = dict()

    def __len__(self):
        return self.indices

    def insert(self, label, emb):
        " Insert testing data "
        self.embs[self.indices] = emb
        self.labels.append(label)
        self.thresholds.append(0)
        self.add_to_dict(label[0])
        self.update_thresholds(emb, label)
        self.indices += 1

    def add_to_dict(self, label):
        if bool(self.class_dict.keys()) is False or label not in self.class_dict.keys():
            self.class_dict.setdefault(label, [])
        self.class_dict[label].append(self.indices)

    def update_thresholds(self, embTest, labelTest):
        max_thd = -1
        # Get class info
        all_classes = self.class_dict.keys()
        class_num = len(all_classes)

        compare_indices = None

        # Not enough images, compare all
        if class_num <= self.compare_num and self.indices <= self.compare_num:
            compare_indices = range(self.indices)
        # Not enough classes, but too many images, equally get images from each classes, last random
        elif class_num <= self.compare_num and self.indices > self.compare_num:
            # Number of images get from each classes
            mul = int(np.floor(float(self.compare_num/class_num)))
            compare_indices = []
            last = []
            cnt = 0
            # Equally select from each classes
            for c in all_classes:
                cur_class_indices = self.class_dict[c]
                if len(cur_class_indices) >= mul:
                    tmp = random.sample(cur_class_indices, mul)
                    compare_indices.extend(tmp)
                    last.extend([v for v in cur_class_indices if v not in tmp])
                    cnt += mul
                else:
                    compare_indices.extend(cur_class_indices)
                    cnt += len(cur_class_indices)
            # Random select for last 
            compare_indices.extend(random.sample(last, self.compare_num-cnt))
        # Too many classes, get one image from random classes
        elif class_num > self.compare_num:
            compare_classes = random.sample(list(all_classes), self.compare_num)
            compare_indices = [random.choice(self.class_dict[c]) for c in compare_classes]

        # Comparing
        for indx in compare_indices:
            # If different class
            if self.labels[indx] != labelTest:
                # Calculate similarity
                new_thd = get_similarity(embTest, self.embs[indx])
                # Update others
                if new_thd > self.thresholds[indx]:
                    self.thresholds[indx] = new_thd
                # Update self
                if new_thd > max_thd:
                    max_thd = new_thd
        if max_thd > -1:
            self.thresholds[self.indices] = max_thd

    def get_most_similar(self, embTest):
        testTiles = np.tile(embTest, (self.indices, 1))
        similarities = np.sum(testTiles*self.embs[0:self.indices], axis=1)
        max_similarity = np.max(similarities)
        max_id = np.argmax(similarities)
        return max_id, max_similarity

    def get_threshold_by_id(self, id):
        return self.thresholds[id]

    def get_label_by_id(self, id):
        return self.labels[id]

    def print_database(self):
        " Debug usage "
        data = zip(range(self.indices), self.labels, self.thresholds)
        table = tabulate(data, headers=['Index', 'Label', 'Threshold'])
        print(table)

    def contains(self, labelTest):
        if self.indices > 0:
            if labelTest in self.labels:
                return True
        return False

    def thresholds_mean(self):
        return np.mean(self.thresholds)

def get_similarity(embA, embB):
    ans = np.sum(embA*embB)
    return ans