import re
import os
import random
import tarfile
import numpy as np
from six.moves import urllib
from torchtext import data
import codecs

class MR(data.Dataset):
    # url = 'https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
    # filename = 'rt-polaritydata.tar'
    # dirname = 'rt-polaritydata'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, label_field, examples=None, shuffle=True, oversamplling=False, **kwargs):
        """Create an MR dataset instance given a path and fields.

        Arguments:
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            path: Path to the data file.
            examples: The examples contain all the data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """

        def clean_str(string):
            """
            Tokenization/string cleaning for all datasets except for SST.
            Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
            """
            string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
            string = re.sub(r"\'s", " \'s", string)
            string = re.sub(r"\'ve", " \'ve", string)
            string = re.sub(r"n\'t", " n\'t", string)
            string = re.sub(r"\'re", " \'re", string)
            string = re.sub(r"\'d", " \'d", string)
            string = re.sub(r"\'ll", " \'ll", string)
            string = re.sub(r",", " , ", string)
            string = re.sub(r"!", " ! ", string)
            string = re.sub(r"\(", " \( ", string)
            string = re.sub(r"\)", " \) ", string)
            string = re.sub(r"\?", " \? ", string)
            string = re.sub(r"\s{2,}", " ", string)
            return string.strip()

        def clean_str_sst(string):
            """
            Tokenization/string cleaning for the SST dataset
            """
            string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
            string = re.sub(r"\s{2,}", " ", string)
            return string.strip().lower()

        # text_field.preprocessing = data.Pipeline(clean_str_sst)
        fields = [('text', text_field), ('label', label_field)]

        if examples is None:
            path = self.dirname if path is None else path
            examples = []
            with codecs.open(os.path.join(path, "./data/data.txt"), errors='ignore') as sents, \
                    codecs.open(os.path.join(path, "./label/label.txt"), errors='ignore') as labels:
                for line, label in zip(sents, labels):
                    line = line.strip()
                    label = label.strip()
                    # if label == '0':
                    #     examples += [
                    #         data.Example.fromlist([line, 'negative'], fields)]
                    # else:
                    #     examples += [
                    #         data.Example.fromlist([line, 'positive'], fields)]

                    examples += [
                        data.Example.fromlist([line, label], fields)]

        if shuffle:
            random.seed(23456)#23456, English,except ST, all use 23456.
            # 11 86.47
            random.shuffle(examples)

        super(MR, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, train=None, validation=None, test=None,
               dev_ratio=.1,small = False, test_ratio=0.1,root='.', **kwargs):
        """Create dataset objects for splits of the MR dataset.

        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            dev_ratio: The ratio that will be used to get split validation dataset.
            shuffle: Whether to shuffle the data before split.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose trees
                subdirectory the data files will be stored2.
            train: The filename of the train data. Default: 'train.txt'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        # path = cls.download_or_unzip(root)
        path = root
        np.random.seed(23456)
        train_data = None if train is None else cls(
            path + train, text_field, label_field,
            **kwargs)



        # validation is not None and test is not None
        val_data = None if validation is None else cls(
            path + validation, text_field, label_field,shuffle=False,
            **kwargs)

        test_data = None if test is None else cls(
            path + test, text_field, label_field,shuffle=False,
            **kwargs)

        if validation is None and test is None:
            examples = cls(path, text_field, label_field, **kwargs)
            split_ratio = dev_ratio + test_ratio
            split_index = -1 * int(split_ratio * len(examples))
            test_index = -1 * int(test_ratio * len(examples))
            np.random.shuffle(examples[:split_index])
            train_data, val_data, test_data = cls(path, text_field, label_field, examples=examples[:split_index]), \
                                         cls(path, text_field, label_field, examples=examples[split_index:test_index]),\
                                   cls(path, text_field, label_field, examples=examples[test_index:])
        elif validation is None and test is not None:
            examples = train_data.examples
            dev_index = -1 * int(dev_ratio * len(examples))
            np.random.shuffle(examples[:dev_index])
            train_data, val_data = cls(path, text_field, label_field, examples=examples[:dev_index]), \
                                    cls(path, text_field, label_field, examples=examples[dev_index:])

        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)
