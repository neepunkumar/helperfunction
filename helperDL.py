#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 21:57:28 2023

@author: Module COMP5625M assessment - helper.py
"""
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import os
import numpy as np

import random
import math


"""
====> Encoder Network used for feature extraction

"""

class EncoderCNN(nn.Module):
    def __init__(self):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)

        # TO COMPLETE
        # keep all layers of the pretrained net except the last one
        layers = list(resnet.children())[:-1]   
        # unpack the layers and create a new Sequential
        self.resnet = nn.Sequential(*layers)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
            
        features = features.reshape(features.size(0), -1)
        return features
    
    

"""
====> Data cleaning and splitting 

"""

def gen_clean_captions_df(df):

    # Remove spaces in the beginning and at the end
    # Convert to lower case
    # Replace all non-alphabet characters with space
    # Replace all continuous spaces with a single space
    clean_caption = df["caption"] \
        .str.strip() \
        .str.lower() \
        .replace(r"[^a-z ]+", "", regex=True) \
        .replace(r"[ ]+", " ", regex=True)

    # add to dataframe
    df["clean_caption"] = clean_caption

    return df


def split_ids(image_id_list, train=.7, valid=0.1, test=0.2):
    """
    Args:
        image_id_list (int list): list of unique image ids
        train (float): train split size (between 0 - 1)
        valid (float): valid split size (between 0 - 1)
        test (float): test split size (between 0 - 1)
    """
    list_copy = image_id_list.copy()
    random.shuffle(list_copy)
    
    train_size = math.floor(len(list_copy) * train)
    valid_size = math.floor(len(list_copy) * valid)
    
    return list_copy[:train_size], list_copy[train_size:(train_size + valid_size)], list_copy[(train_size + valid_size):]



"""
====> Building vocabularies 

"""

    
MIN_FREQUENCY = 3
def build_vocab(df_ids, new_file, vocab):
    """ 
    Parses training set token file captions and builds a Vocabulary object and dataframe for 
    the image and caption data

    Returns:
        vocab (Vocabulary): Vocabulary object containing all words appearing more than min_frequency
    """
    word_mapping = Counter()

    # for index in df.index:
    for index, id in enumerate(df_ids):
        caption = str(new_file.loc[new_file['image_id']==id]['clean_caption'])
        for word in caption.split():
            # also get rid of numbers, symbols etc.
            if word in word_mapping:
                word_mapping[word] += 1
            else:
                word_mapping[word] = 1



    # add the words to the vocabulary
    for word in word_mapping:
        # Ignore infrequent words to reduce the embedding size
        if word_mapping[word] > MIN_FREQUENCY:
            vocab.add_word(word)

    return vocab


"""
====> Linking image and target labels (texts) 

"""

MAX_SEQ_LEN = 47
class COCO_Features(Dataset):
    """ COCO custom dataset with features and vocab, compatible with torch.utils.data.DataLoader. """
    
    def __init__(self, df, vocab, features, padded=False, max_len=MAX_SEQ_LEN):
        """ Set the path for images, captions and vocabulary wrapper.
        
        Args:
            df (str list): dataframe of image meta
            captions (str list): list of str captions
            vocab: vocabulary wrapper
            features: torch Tensor of extracted features
        """
        self.df = df
        self.vocab = vocab
        self.features = features
        
        self.padded = padded
        self.max_len = max_len

    def __getitem__(self, index):
        """ Returns one data pair (feature and target caption). """

        # path = IMAGE_DIR + str(self.df.iloc[index]['file_name'])

        entry = self.df.iloc[index]
        image_id = entry["file_name"]   

        image_features = self.features[image_id]

        # convert caption (string) to word ids.
        tokens = self.df.iloc[index]['clean_caption'].split()
        caption = []

        # build the Tensor version of the caption, with token words
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('<end>'))
        target = torch.Tensor(caption)

        return image_features, target.int()

    def __len__(self):
        return len(self.df)
    

def caption_collate_fn(data):
    """ Creates mini-batch tensors from the list of tuples (image, caption).
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 224, 224).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 224, 224).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length from longest to shortest.
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # merge images (from tuple of 3D tensor to 4D tensor).
    # if using features, 2D tensor to 3D tensor. (batch_size, 256)
    images = torch.stack(images, 0) 

    # merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    # pad with zeros
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths

def caption_collate_padded_fn(data):
    """ Creates mini-batch tensors from the list of tuples (image, caption).
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 224, 224).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 224, 224).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length from longest to shortest.
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # merge images (from tuple of 3D tensor to 4D tensor).
    # if using features, 2D tensor to 3D tensor. (batch_size, 256)
    images = torch.stack(images, 0) 

    # merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    # pad with zeros
    targets = torch.zeros(len(captions), MAX_SEQ_LEN).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths

"""
====> Decode caption from -- word_ids to words

"""
def decode_caption(sampled_ids, vocab):
    """ 
    Args:
        ref_captions (str list): ground truth captions
        sampled_ids (int list): list of word IDs from decoder
    """
    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        if word != '<start>':
            if word == '<end>':
                break
        
        sampled_caption.append(word)

    sentence = ' '.join(sampled_caption)
    return sentence


def timshow(x):
    x = (x-x.min())/(x.max()-x.min())
    x = x.detach().clamp_(min=0, max=1).mul(255).type(torch.uint8)
    xa = np.transpose(x.numpy(),(1,2,0))
    plt.imshow(xa)
    plt.axis('off')
    plt.show()
    return xa