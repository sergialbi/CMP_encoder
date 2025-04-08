"""
Useful functions for loading the data, pre-processing and training

@author: Sergi Albiach
@institution: BSC
@date: 05/10/2022
@version: 0.1
"""
import pandas as pd
import numpy as np
import os
import re
import math
from random import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel, AutoTokenizer


#########################
##### DATA HANDLING #####
#########################
def cmp_transform(cmp:str, level_codes:int) -> str:
    """
    Transform a CMP_code in one of the three categories: the original code, the parent code or the general category
    @param cmp: the CMP_code to transform
    @param level_codes: the level of transformation to be applied to the code
    return: the transformed code or None if an invalid strict level is specified
    """
    if level_codes == 0: #ORIGINAL CODE (103.2 --> 103.2)
        return str(cmp)
    elif level_codes == 1: #PRENT CODE (103.2 --> 103)
        return str(cmp).split(".")[0]
    elif level_codes == 2: #GENERAL CATEGORY (103.2 --> 1)
        return str(cmp)[0]
    else:
        print(f"[ERROR] An invalid level of strictness ({level_codes}) was specified! --> level_codes = [0, 1, 2]")
    return ""


def histogram(df:pd.DataFrame, category:str="cmp_code", save:bool=False, save_name:str="histogram"):
    """
    Show and save (if desired) the histogram of a DataFrame's column
    @param df: DataFrame from where to extract the column
    return: nothing
    """
    df.groupby([category]).size().plot.bar(figsize=(15, 5))
    if save:
        plt.savefig(f"{save_name}.png", dpi=300)
    plt.show()


def read_data(data_path:str, level_codes:int=1, show_histogram:bool=False, save_histogram:bool=False) -> pd.DataFrame:
    """
    Function to read the data in data_path, transform its CMP_codes to a level of strictness and show its histogram if desired
    @param data_path: path of the dataset to read
    @param level_codes: level of strictness to be applied [0, 1, 2]
    @param show_histogram: flag to show or not the histogram of the cmp_codes
    @param save_histogram: flag to save or not the histogram of the cmp_codes
    return: the df with the 'cmp_code' category transformed by level_codes
    """
    print("Loading the data...")
    extension = data_path.split(".")[1]
    data_path = os.path.abspath(data_path)
    if extension=="xlsx":
        df = pd.read_excel (data_path)
    elif extension == "csv":
        df = pd.read_csv(data_path)	
    else:
        print("Given file extension cannot be processed.")
        return None

    cmp_codes_transformed = np.asarray([cmp_transform(code, level_codes) for code in df["cmp_code"]], dtype=str)
    df['cmp_code'] = cmp_codes_transformed

    if show_histogram:
        histogram(df, category='cmp_code', save=save_histogram, save_name="histogram_original")
    return df


class Dataset(torch.utils.data.Dataset):
    """
    Implementation of our own dataset.
    The features are the tokenized text and the label is the codified 'cmp_code'
    """

    def __init__(self, df:pd.DataFrame, tokenizer:AutoTokenizer, labels_dict:dict):
        self.labels = [labels_dict[label] for label in df['cmp_code']]
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['text']]

    def classes(self):
        return self.labels

    def __len__(self) -> int:
        # Compute the length of the Dataset (number of samples)
        return len(self.labels)

    def get_batch_labels(self, idx:list) -> np.array:
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx:list):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx:list):
        # Get a batch of inputs and labels
        return self.get_batch_texts(idx), self.get_batch_labels(idx)


class BertClassifier(nn.Module):
    """
    The Encoder to transform text into the 'cmp_code' prediction, the BertClassifier.
    """

    def __init__(self, tokenizer_name:str, dropout:float=0.5, num_classes:int=56, pretrained_model_path=""):
        self.num_classes = num_classes
        super(BertClassifier, self).__init__()
        if pretrained_model_path!="":
            print(f"Using pretrained model from '{pretrained_model_path}'")
            self.bert = BertModel.from_pretrained(pretrained_model_path)
        else:
            self.bert = BertModel.from_pretrained(tokenizer_name)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, self.num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        #final_layer = self.relu(linear_output)
        final_layer = self.softmax(linear_output)

        return final_layer


def print_results(train_acc:np.array, train_loss:np.array, val_acc:np.array, val_loss:np.array, save_path:str, title:str="", extra_save=""):
    """
    Function to plot the trainig and validation results
    @param train_acc: train accuracies per epoch
    @param train_loss: train losses per epoch
    @param val_acc: validation accuracy per epoch
    @param val_loss: validation loss per epoch
    @param save_path: path to save the figures
    return: nothing
    """
    # x_axis ticks
    #x = np.asarray([f"{i+1}" for i in range(len(train_acc))])
    x = np.asarray([i+1 for i in range(len(train_acc))])
    
    fig, axis = plt.subplots(1,3, figsize=(25,5))
    fig.suptitle(title)

    y1 = val_acc
    y2 = val_loss
    axis[0].set_title("Validation Metrics")
    axis00_2 = axis[0].twinx()
    axis00_2.plot(x, y2, 'blue')
    axis[0].plot(x, y1, 'red')
    axis[0].set_xlabel('epoch')
    axis[0].set_ylabel('accuracy', color='red')
    axis00_2.set_ylabel('loss', color='blue')

    y1 = val_acc
    y2 = train_acc
    axis[1].set_title("Accuracies")
    axis[1].plot(x, y1, 'red', label="val_acc")
    axis[1].plot(x, y2, 'blue', label="train_acc")
    axis[1].set_xlabel('epoch')
    axis[1].set_ylabel('accuracy')
    axis[1].legend()

    y1 = val_loss
    y2 = train_loss
    axis[2].set_title("Losses")
    axis[2].plot(x, y1, 'red', label="val_loss")
    axis[2].plot(x, y2, 'blue', label="train_loss")
    axis[2].set_xlabel('epoch')
    axis[2].set_ylabel('loss')
    axis[2].legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path+f"accuracy_losses{extra_save}.png", bbox_inches='tight', dpi=300)
    plt.show()

def print_results_f1(train_f1:np.array, train_loss:np.array, val_f1:np.array, val_loss:np.array, save_path:str, title:str="", extra_save=""):
    """
    Function to plot the trainig and validation results
    @param train_acc: train accuracies per epoch
    @param train_loss: train losses per epoch
    @param val_acc: validation accuracy per epoch
    @param val_loss: validation loss per epoch
    @param save_path: path to save the figures
    return: nothing
    """
    # x_axis ticks
    #x = np.asarray([f"{i+1}" for i in range(len(train_acc))])
    x = np.asarray([i+1 for i in range(len(train_f1))])
    
    fig, axis = plt.subplots(1,3, figsize=(25,5))
    fig.suptitle(title)

    
    axis[0].set_title("Validation Metrics")
    axis00_2 = axis[0].twinx()
    y1 = val_f1
    axis00_2.plot(x, y1, 'red')
    axis00_2.set_ylabel('F1', color='red')
    y2 = val_loss
    axis[0].plot(x, y2, 'blue')
    axis[0].set_ylabel('loss', color='blue')
    axis[0].set_xlabel('epoch')
    

    y1 = val_f1
    y2 = train_f1
    axis[1].set_title("F1 Score (macro)")
    axis[1].plot(x, y1, 'red', label="val_f1")
    axis[1].plot(x, y2, 'blue', label="train_f1")
    axis[1].set_xlabel('epoch')
    axis[1].set_ylabel('f1 score')
    axis[1].legend()

    y1 = val_loss
    y2 = train_loss
    axis[2].set_title("Losses")
    axis[2].plot(x, y1, 'red', label="val_loss")
    axis[2].plot(x, y2, 'blue', label="train_loss")
    axis[2].set_xlabel('epoch')
    axis[2].set_ylabel('loss')
    axis[2].legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path+f"f1_losses{extra_save}.png", bbox_inches='tight', dpi=300)
    plt.show()

"""
def is_data_correct(data, labels):
    data_check = data is not None and data.size != 0
    label_check = labels is not None and labels.size != 0
    if data_check and label_check:
        print(f"Correctly loaded data with {data.size} samples")
        return True
    else:
        print("Data is empty, cannot continue.")
        return False

def make_partition(data, labels, train=0.7, val=0.2, test=0.1, seed=12345):
    print("Making train/val/test partitions...")
    rng = np.random.default_rng(seed=seed)
    
    num_samples = len(data)
    indxs = np.arange(num_samples)
    rng.shuffle(indxs)

    val_num_samples = int(val*num_samples)
    test_num_samples = int(test*num_samples)
    train_num_samples = int(train*num_samples) + ((int(train*num_samples) + val_num_samples + test_num_samples) - num_samples)

    start, end = 0, train_num_samples
    train_data, train_labels = data[indxs[start:end]], labels[indxs[start:end]]
    start, end = train_num_samples, train_num_samples + val_num_samples
    val_data, val_labels = data[indxs[start:end]], labels[indxs[start:end]]
    start, end = train_num_samples + val_num_samples, train_num_samples + val_num_samples + test_num_samples
    test_data, test_labels = data[indxs[start:end]], labels[indxs[start:end]]

    return train_data, train_labels, val_data, val_labels, test_data, test_labels

def process_sentence(sentence):
        return re.sub("[.,!?\\-]", '', sentence.lower())

def preprocess_sentences(orig_sentences):
    process_vec = np.vectorize(process_sentence)
    return process_vec(orig_sentences)


def create_tokens(sentences, vocabulary):
    word_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
    for i, token in enumerate(vocabulary):
        word_dict[token] = i + 4

    token_list = list()
    for sentence in sentences:
        arr = [word_dict[s] for s in sentence.split()]
        token_list.append(arr)

    return word_dict, token_list


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def make_batch(word_dict, token_list, number_dict,sentences,vocab_size, maxlen=62, batch_size=6, max_pred=5, n_layers = 6, n_heads = 12, d_model = 768, d_ff = 768 * 4,d_k_v=64, n_segments=2 ):
    d_k = d_v = d_k_v
    batch = []
    positive = negative = 0
    while positive != batch_size/2 or negative != batch_size/2:
        tokens_a_index, tokens_b_index= randrange(len(sentences)), randrange(len(sentences))
        tokens_a, tokens_b= token_list[tokens_a_index], token_list[tokens_b_index]

        input_ids = [word_dict['[CLS]']] + tokens_a + [word_dict['[SEP]']] + tokens_b + [word_dict['[SEP]']]

        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

        #MASK LM
        n_pred =  min(max_pred, max(1, int(round(len(input_ids) * 0.15)))) # 15 % of tokens in one sentence

        cand_maked_pos = [i for i, token in enumerate(input_ids)
                          if token != word_dict['[CLS]'] and token != word_dict['[SEP]']]
        shuffle(cand_maked_pos)
        masked_tokens, masked_pos = [], []
        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            if random() < 0.8:  # 80%
                input_ids[pos] = word_dict['[MASK]'] # make mask
            elif random() < 0.5:  # 10%
                index = randint(0, vocab_size - 1) # random index in vocabulary
                input_ids[pos] = word_dict[number_dict[index]] # replace

        # Zero Paddings
        n_pad = maxlen - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

        # Zero Padding (100% - 15%) tokens
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)

        if tokens_a_index + 1 == tokens_b_index and positive < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True]) # IsNext
            positive += 1
        elif tokens_a_index + 1 != tokens_b_index and negative < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False]) # NotNext
            negative += 1
    return batch
"""
