"""
Training of the encoder.

@author: Sergi Albiach
@institution: BSC
@date: 22/11/2022
@version: 1.0
"""

from random import *
from lib.utils import *

import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from torch import nn
from tqdm import tqdm



class BertClassifier(nn.Module):
    """
    The Encoder to transform text into the 'cmp_code' prediction, the BertClassifier.
    """

    def __init__(self, tokenizer_name:str, dropout:float=0.5, num_classes:int=56):
        self.num_classes = num_classes
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained(tokenizer_name)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, self.num_classes)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer


def get_cmp_code_df(model, tokenizer, df, bs=8):
    """
    Function to get the 'cmp_code' from all the sentences in the DF
    @param model: the model to train, the classifier.
    @param tokenizer: the Tokenizer used to tokenize the sentences
    @param df: the dataframe containing the data to test
    return: the first and scond labels for the sentences in the DF
    """
    all_orig_labels = sorted(np.unique(df["cmp_code"]))
    out_codes1, out_codes2 = np.empty(len(df), dtype=object), np.empty(len(df), dtype=object)
    df = df.loc[range(0,1000)]

    # Generate the DataLoader for the DF
    print("Preparing the data...")
    test_dataloader = torch.utils.data.DataLoader(Dataset(df, tokenizer, labels_dict), batch_size=bs)

    # Move to corresponding device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()

    total_acc_test = 0

    with torch.no_grad():
        # For every batch in the Dataloader
        for i,(test_input, test_label) in enumerate(test_dataloader):
            print(f"Batch {i+1}/{len(test_dataloader)}")
            # Obtain inputs and move them to corresponding device
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            # Forward pass
            output = model(input_id, mask)

            out = output.detach().cpu().numpy()
            #ind = np.argmax(out)
            inds = [np.argsort(-o)[:2] for o in out]
            ind1 = [ind[0] for ind in inds]
            ind2 = [ind[1] for ind in inds]

            start = i*bs
            end = start+bs if start+bs <= len(df) else len(df)

            out_codes1[start:end] = np.asarray(all_orig_labels)[ind1]
            out_codes2[start:end] = np.asarray(all_orig_labels)[ind2]

            # Accuracy computation
            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc
    
    print(f'Accuracy: {total_acc_test / len(df): .3f}')
    return out_codes1, out_codes2


def get_cmp_code_sentences(model, tokenizer, sentences, code_title_dict, max_length=512, print_label=True):
    """
    Function to get the 'cmp_code' from a set of sentences
    @param model: the model to train, the classifier.
    @param tokenizer: the Tokenizer used to tokenize the sentences
    @param sentences: the sentences to classify
    @param code_title_dict: dictionary contaiing the relation between the codebook cmp_code and its description
    @param max_length: maximum length for the tokenizer
    @param print_label: flag to print in the terminal the predicted code for the sentences
    return: the predicted first and second codes of the sentences
    """
    all_orig_labels = sorted(np.unique(df["cmp_code"]))
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()

    out_codes1, out_codes2 = np.empty(len(sentences), dtype=object), np.empty(len(sentences), dtype=object)

    for i,sentence in enumerate(sentences):
        if len(sentences)>1:
            print(f"Sentence {i+1}/{len(sentences)}")
        tokenized_sentence = tokenizer(sentence, 
                                        padding='max_length', max_length = max_length, truncation=True,
                                        return_tensors="pt")
        
        with torch.no_grad():
            mask = tokenized_sentence['attention_mask'].to(device)
            input_id = tokenized_sentence['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            
            out = output.detach().cpu().numpy()[0]
            #ind = np.argmax(out)
            inds = np.argsort(-out)[:2]
            ind1, ind2 = inds[0], inds[1]
            code1, code2 = str(all_orig_labels[ind1]), str(all_orig_labels[ind2])
            out_codes1[i], out_codes2[i] = code1, code2

        if print_label:
            print(f'Output class is: {str(code1)} - {code_title_dict.get(str(code1))}\n')
    
    return out_codes1, out_codes2


def predict_and_add_to_file(model, tokenizer, df):
    """
    Function to add to the DF the first and second predictions of the model
    @param model: the model to train, the classifier.
    @param tokenizer: the Tokenizer used to tokenize the sentences
    @param df: the DF to classify its sentences
    return: the original df plus the predicted first and second codes of the sentences
    """
    out_codes1, out_codes2 = get_cmp_code_df(model, tokenizer, df)
    df["is_correct"] = df["cmp_code"] == out_codes1
    df["prediction1"] = out_codes1
    df["prediction2"] = out_codes2
    
    return df



if __name__ == "__main__":
    data_path = "/home/salbiach/Documents/projectes/politics/cmp_encoder/dataset/CMPD_model_test.xlsx"
    save_path = "/home/salbiach/Documents/projectes/politics/cmp_encoder/results/torchmodel_weights.pth"
    tokenizer_name = "dccuchile/bert-base-spanish-wwm-cased"
    codebook_path = "/home/salbiach/Documents/projectes/politics/cmp_encoder/dataset/official_docs/codebook.xlsx"

    df = read_data(data_path, level_codes=1)

    all_orig_labels = sorted(np.unique(df["cmp_code"]))
    labels_dict = {f'{code}':i for i,code in enumerate(all_orig_labels)}

    codebook = pd.read_excel(codebook_path)
    codes, titles = np.asarray(codebook["code"], dtype=int), np.asarray(codebook["title"], dtype=str)
    code_title_dict = {str(c):t for c,t in zip(codes, titles)}

    saved_model = BertClassifier(tokenizer_name = tokenizer_name, num_classes = len(all_orig_labels))
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    saved_model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))

    input_read = ''
    options = [0, 1, 2]
    while input_read not in options:
        input_read = int(input( "Select one of the folowing options:\n"+
                            "(0) EXIT\n"
                            "(1) Predict the text written in terminal\n"+
                            "(2) Predict a file\n"+
                            "--> "))
        if input_read not in options:
            print("Not a valid option!\n\n")

    if input_read == 0:
        print("Bye!")
    
    elif input_read == 1:
        print("-----------------------------")
        print("     PREDICT IN TERMINAL     ")
        print("-----------------------------")
        print("Type 'q' in order to exit. Write your text in: [Spanish]")
        input_read = input("Write your sentence: ")
        while (input_read != 'q'):
            if str(input_read) in all_orig_labels:
                print(f"Code {input_read} means: {code_title_dict.get(str(input_read))}\n")
            else:
                _,_ = get_cmp_code_sentences(saved_model, tokenizer, [input_read], code_title_dict)
            input_read = input("('q' to exit) Next sentence: ")

    elif input_read == 2:
        print("---------------------------")
        print("     PREDICT FROM FILE     ")
        print("---------------------------")
        input_read = input("Write the path of your file ('d' for default): ")
        if input_read == 'd':
            df_to_read = df.copy()

        else:
            df_to_read = read_data(input_read, level_codes=1)
        df_out = predict_and_add_to_file(saved_model, tokenizer, df_to_read)
        name = data_path.split('.')[0]+"_predictions.xlsx"
        df_out.to_excel(name, index=False)
