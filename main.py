#! /usr/bin/env python
import operator
import os
import argparse
import datetime
import random

import torch
import torchtext.data as data
import torchtext.datasets as datasets
from torch.autograd import Variable

import model
import train
import data_loader
import numpy as np
#import utils

# class cnn_classifer():
parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=7, help='number of epochs for train [default: 10]')
parser.add_argument('-batch-size', type=int, default=50, help='batch size for training [default: 48]')
parser.add_argument('-log-interval', type=int, default=1,
                    help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=10
                    , help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
# data 
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
parser.add_argument('-lang', type=str, default='', help='choose which language of datasets')
parser.add_argument('-data-dir', type=str, default=os.getcwd() + "/",
                    help='choose which language of datasets')
# model
parser.add_argument('-length', type=int, default=None, help='number of embedding dimension [default: 300]')
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=300, help='number of embedding dimension [default: 300]')
parser.add_argument('-senti-embed-dim', type=int, default=1,
                    help='number of sentiment embedding dimension [default: 300]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5',
                    help='comma-separated kernel size to use for convolution')
parser.add_argument('-senti-kernel-size', type=int, default=5,
                    help=' kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
# word vector
parser.add_argument('-vec-dir', type=str, default=os.getcwd() + "/")
parser.add_argument('-vector-cache', type=str, default='input_vectors.pt')  # 'sent_input_vectors.pt'
parser.add_argument('-sent-vector-cache', type=str, default='sent_input_vectors.pt')
parser.add_argument('-data-cache', type=str, default='/Users/Xin/Documents/17_Spring/glove/')
parser.add_argument('-word-vector', type=str, default='glove.840B', help='load glove word vector')
parser.add_argument('-trainable', action='store_true', default=True, help="whether word vector is trainable")
parser.add_argument('-senti-trainable', action='store_false', default=True,
                    help="whether sentiment word vector is trainable")
parser.add_argument('-concat', action='store_true', default=False, help="concatenate two vectors")
parser.add_argument('-multi-channels', action='store_true', default=False, help="construct another channel")
parser.add_argument('-random', type=bool, default=False, help="whether normal embedding is initialized")
parser.add_argument('-senti-random', action='store_true', default=False, help="whether senti embedding is initialized")
parser.add_argument('--seed', type=int, default=11, metavar='S', help='random seed (default: 11,3435,53254, 34353435,34203)')
parser.add_argument('-data-source', type=str, default='', help='load glove word vector')
args = parser.parse_args()
torch.manual_seed(args.seed)


# load ST dataset
def st(text_field, label_field, train=None, val=None, test=None, **kargs):
    train_data, dev_data, test_data = data_loader.MR.splits(text_field, label_field,
                                                            train=train, validation=val, test=test, root=args.data_dir)

    text_field.build_vocab(train_data, dev_data, test_data)
    build_wv(text_field)
    label_field.build_vocab(train_data, dev_data, test_data)
    '''
    if True:
        sd_ratio = 0.5
        examples = train_data.examples
        sd_index = -1 * int(sd_ratio * len(examples))
        train_data = train_data[:sd_index]
        fields = [('text', text_field), ('label', label_field)]
        train_data = data.Dataset(train_data, fields)
    '''
    train_iter, dev_iter, test_iter = data.Iterator.splits(
        (train_data, dev_data, test_data),
        batch_sizes=(args.batch_size, len(dev_data), len(test_data)), sort=False, shuffle=True,
        **kargs)
    return train_iter, dev_iter, test_iter


def build_wv(text_field):
    if args.word_vector:
        print (args.vec_dir + args.vector_cache)
        if os.path.isfile(args.vec_dir + args.vector_cache):
            if args.concat:
                v1 = torch.load(args.vec_dir + args.vector_cache)
                v2 = torch.load(args.vec_dir + args.sent_vector_cache)
                if "vader" in args.word_vector:
                    v2 = utils.vader_proprecessing(v2)
                v = torch.cat((v1, v2), 1)
                v1_num = np.sum(v1.numpy(), 1)
                print(len(np.where(v1_num != 0)[0]))
                v2_num = np.sum(v2.numpy(), 1)
                print(len(np.where(v2_num != 0)[0]))
                text_field.vocab.vectors = v

            else:
                text_field.vocab.vectors = torch.load(args.vec_dir + args.vector_cache)

        else:
            v_pre = text_field.vocab.load_vectors(wv_dir=args.data_cache, wv_type=args.word_vector,
                                                  wv_dim=args.embed_dim, unk_init=None, lang=None)  #
            print ("V_pre: " + str(v_pre))
            os.makedirs(os.path.dirname(args.vec_dir + args.vector_cache), exist_ok=True)
            torch.save(text_field.vocab.vectors, args.vec_dir + args.vector_cache)


def voc_pres(text_field):
    vpre = text_field.vocab.load_vectors(wv_dir=args.data_cache, wv_type=args.word_vector,
                                         wv_dim=args.embed_dim, unk_init=None)
    print(vpre)


def load_sent_wv(text_field):
    vpre = text_field.vocab.load_vectors(wv_dir=args.data_cache, wv_type=args.word_vector,
                                         wv_dim=args.senti_embed_dim, unk_init=None, lang=None)
    print(vpre)


def check_senti_weights_update(text_field):
    model_file = os.path.join(os.getcwd(), 'snapshot/2017-10-31_16-20-08/snapshot_steps556.pt')
    cnn = torch.load(model_file)
    senti_weights = cnn.senti_embed.weight.data.numpy()
    train_data, dev_data, test_data = data_loader.MR.splits(text_field, label_field, train="train", validation="dev",
                                                            test="test",
                                                            root=args.data_dir)
    # train = "train", validation = "dev", test = "test",
    text_field.build_vocab(train_data, dev_data, test_data)
    words = text_field.vocab.itos
    v_org = torch.load(args.vec_dir + args.sent_vector_cache)
    v_org = utils.vader_proprecessing(v_org)
    word_diff_score = {}
    for i, word in enumerate(words):
        word_diff_score[word] = [v_org[i][0], senti_weights[i][0], abs(v_org[i][0] - senti_weights[i][0])]
    sorted_w = sorted(word_diff_score.items(), key=lambda x: x[1][2], reverse=True)
    new_words = []
    org_scores = []
    new_scores = []
    for wands in sorted_w:
        w = wands[0]
        # new_words.append(w)
        if abs(wands[1][0]) > 0.01:
            new_words.append(w)
            org_scores.append(wands[1][0])
            new_scores.append(wands[1][1])
    utils.visual(new_words, org_scores, new_scores, 0, 50)
    print()


    # English


#args.data_dir = '/Users/Xin/Documents/17_Spring/NN_Baseline/English/data/st/'
args.vector_cache = 'g300_nt_st_iv.pt'  # 'g300_st_input_vectors.pt' 85.89 86.11
args.sent_vector_cache = 's26_nt_st_iv.pt'  # 'new_s26_en_input_vectors.pt'


from nltk.tokenize import TweetTokenizer

tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)


args.lr = 0.0004  # 86.22
args.embed_dim = 300
args.senti_embed_dim = 26
args.multi_channels = True
args.concat = True
# args.w2v = True
# args.senti_trainable = False
# args.senti_random = True
# args.trainable = False
# args.data_cache = "/Users/Xin/Documents/17_Spring/glove/polyglot/"
args.data_cache = "/Users/Xin/Documents/WSDM/cnn-text-classification/nnew_svm_embedding/"
args.word_vector = "t:en.svm"
args.vec_dir = args.vec_dir + args.lang + "/" + args.data_source + "/"



#args.senti_trainable = False
# load data
print("\nLoading data...")

# tokenizer.tokenize utils.cn_tokenize 'moses'  segmenter.tokenize
text_field = data.Field(lower=True, tokenize=tokenizer.tokenize)
label_field = data.Field(sequential=False)

# check_senti_weights_update(text_field)
train_iter, dev_iter, test_iter = st(text_field, label_field, train = "train", val = "dev", test = "test", device=-1, repeat=False)
# train = "train", val = "dev", test = "test",
# train_iter, dev_iter = s1_nl(text_field, label_field, device=-1, repeat=False)
# load_sent_wv(text_field)
# voc_pres(text_field)
# update args and print
args.embed_num = len(text_field.vocab)
args.class_num = len(label_field.vocab) - 1
args.cuda = (not args.no_cuda) and torch.cuda.is_available()
del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

# model
cnn = None
if args.snapshot is None:
    if args.multi_channels is not True and args.concat is True:
        args.embed_dim = args.embed_dim + args.senti_embed_dim
    cnn = model.CNN_Text(args)
    # cnn = model.CNN_Text(args)
    if args.word_vector:
        if args.multi_channels:
            cnn.embed.weight.data = text_field.vocab.vectors[:, :args.embed_dim]
            if args.senti_random is False:
                cnn.senti_embed.weight.data = text_field.vocab.vectors[:, args.embed_dim:]
            if args.senti_trainable is False:
                cnn.senti_embed.weight.requires_grad = False
        else:
            if args.random is not True:
                cnn.embed.weight.data = text_field.vocab.vectors
            print()
        if args.trainable is not True:
            cnn.embed.weight.requires_grad = False
else:
    print('\nLoading model from [%s]...' % args.snapshot)
    try:
        cnn = torch.load(args.snapshot)
    except:
        print("Sorry, This snapshot doesn't exist.")
        exit()

if args.cuda:
    cnn = cnn.cuda()

# train or predict

if args.predict is not None:
    label = train.predict(args.predict, cnn, text_field, label_field)
    print('\n[Text]  {}[Label] {}\n'.format(args.predict, label))
elif args.test:
    try:
        train.eval(test_iter, cnn, args)
    except Exception as e:
        print("\nSorry. The test dataset doesn't  exist.\n")
else:
    train.train(train_iter, dev_iter, test_iter, cnn, args)
