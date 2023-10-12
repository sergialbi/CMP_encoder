"""
Training of the encoder.

@author: Sergi Albiach
@institution: BSC
@date: 22/11/2022
@version: 1.0
"""


from lib.utils import *

from random import *
import os
import argparse
import sys
import pandas as pd
import torch
import numpy as np

from transformers import BertTokenizer
from torch import nn
from torch.optim import Adam
from sklearn.metrics import f1_score
from tqdm import tqdm


def train(model: BertClassifier, tokenizer:BertTokenizer, labels_dict:dict, train_data:pd.DataFrame, val_data:pd.DataFrame, epochs:int, criterion, optimizer, bs:int=2, patience=5):
    """
    Function to train the 'model' for a number of epochs and evaluate it with 'val_data'
    @param model: the model to train, the classifier.
    @param tokenizer: the Tokenizer used to tokenize the sentences
    @param labels_dict: a dictionary relting the original labels with the codification
    @param train_data: the dataframe containing the training data
    @param val_data: the dataframe contining the validation data
    @param epochs: the number of epochs to train the model for
    @param criterion: the loss used to train the model
    @param optimizer: the optimizer used to train the model
    @param bs: the batch size used to train the model
    return: the train and validaton accuracy and losses per epoch
    """
    # Generate the DataLoaders for Training and Validation
    train_dataloader = torch.utils.data.DataLoader(Dataset(train_data, tokenizer, labels_dict), batch_size=bs, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(Dataset(val_data, tokenizer, labels_dict), batch_size=bs)
    total_labels = list(labels_dict.values())

    # Move to corresponding device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    
    # Placeholders for storing metric evolution
    train_acc, train_loss, train_f1, train_f1_macro = [],[],[],[]
    val_acc, val_loss, val_f1, val_f1_macro = [],[],[],[]

    best_loss=np.inf
    patience_counter = 0

    # For every epoch
    for epoch_num in range(epochs):
        print(f"Epoch {epoch_num+1}/{epochs}")

        total_acc_train, total_loss_train = 0, 0
        total_acc_val, total_loss_val = 0, 0

        targets = []
        outputs = []

        ##### TRAIN #####
        # For every batch in the Dataloader
        for train_input, train_label in tqdm(train_dataloader):
            # Obtain inputs and move them to corresponding device
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            # Forward pass
            output = model(input_id, mask)

            # Accuracy and Loss computation      
            pred_label = output.argmax(dim=1) 
            outputs.extend(pred_label.detach().cpu().numpy())
            targets.extend(train_label.detach().cpu().numpy())

            acc = (pred_label == train_label).sum().item()
            total_acc_train += acc

            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        train_f1_macro.append(f1_score(outputs, targets, average='macro', labels=total_labels, zero_division=0))
        train_acc.append(total_acc_train / len(train_data))
        train_loss.append(total_loss_train / len(train_data))

        targets = []
        outputs = []
        with torch.no_grad():
            ##### VALIDATE #####
            # For every batch in the Dataloader
            for val_input, val_label in val_dataloader:
                # Obtain inputs and move them to corresponding device
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                # Forward pass
                output = model(input_id, mask)
                
                # Accuracy and Loss computation
                pred_label = output.argmax(dim=1)
                outputs.extend(pred_label.detach().cpu().numpy())
                targets.extend(val_label.detach().cpu().numpy())
                acc = (pred_label == val_label).sum().item()
                total_acc_val += acc
                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

        val_f1_macro.append(f1_score(outputs, targets, average='macro', labels=total_labels, zero_division=0))
        val_acc.append(total_acc_val / len(val_data))
        val_loss.append(total_loss_val / len(val_data))

        print(f'Epoch: {epoch_num + 1} '+
            f'| Train Loss: {train_loss[epoch_num]: .3f} | Train Accuracy: {train_acc[epoch_num]: .3f} | Train F1 (macro): {train_f1_macro[epoch_num]: .3f}'+
            f'| Val Loss: {val_loss[epoch_num]: .3f} | Val Accuracy: {val_acc[epoch_num]: .3f} | Val F1 (macro): {val_f1_macro[epoch_num]: .3f}')
        
        if val_loss[epoch_num] < best_loss:
            best_loss = val_loss[epoch_num]
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"[WARNING] Exitting because patience ({patience}) on loss has been reached")
            break


    return np.asarray(train_acc), np.asarray(train_loss), np.asarray(val_acc), np.asarray(val_loss)


def evaluate(model: BertClassifier, tokenizer:BertTokenizer, labels_dict:dict, test_data:pd.DataFrame):
    """
    Function to evaluate the 'model' with 'test_data'
    @param model: the model to train, the classifier.
    @param tokenizer: the Tokenizer used to tokenize the sentences
    @param labels_dict: a dictionary relting the original labels with the codification
    @param test_data: the dataframe containing the data to test
    return: nothing
    """
    # Generate the DataLoader for the Testing DF
    test_dataloader = torch.utils.data.DataLoader(Dataset(test_data, tokenizer, labels_dict), batch_size=2)
    total_labels = list(labels_dict.values())

    # Move to corresponding device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()

    total_acc_test = 0

    targets = []
    outputs = []
    with torch.no_grad():
        # For every batch in the Dataloader
        for test_input, test_label in test_dataloader:
            # Obtain inputs and move them to corresponding device
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            # Forward pass
            output = model(input_id, mask)

            # Accuracy computation
            pred_label = output.argmax(dim=1)
            outputs.extend(pred_label.detach().cpu().numpy())
            targets.extend(test_label.detach().cpu().numpy())
            acc = (pred_label == test_label).sum().item()
            total_acc_test += acc

    test_f1_macro = f1_score(outputs, targets, average='macro', labels=total_labels, zero_division=0)
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f} | Test F1 Score (macro): {test_f1_macro: .3f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=str, default="/home/bsc37/bsc37726/politics/cmp_encoder/dataset/", help="The path to the dataset file for training")
    parser.add_argument("--train-file-name", type=str, default="CMPD_c1_train_rile.xlsx", help="The name of the training file")
    parser.add_argument("--tokenizer", type=str, default='/home/bsc37/bsc37726/politics/cmp_encoder/lib/tokenizers/bert-base-spanish-wwm-cased/', help="The BERT tokenizer and pre-trained weights name (download) or path")
    parser.add_argument("--save-model-path", type=str, default="/home/bsc37/bsc37726/politics/cmp_encoder/results/", help="The path to store the model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train the model")
    parser.add_argument("--lr", type=int, default=1e-6, help="Learning rate")
    parser.add_argument("--bs", type=int, default=2, help="Batch size")
    parser.add_argument('--reduced', action='store_true', help="Use reeduced dataset, define size with '--reduced-size'")
    parser.add_argument("--reduced-size", type=int, default=10000, help="Size of the reduced dataset")
    parser.add_argument("--results-path", type=str, default="/home/bsc37/bsc37726/politics/cmp_encoder/results/", help="The path to store the figures")
    parser.add_argument('--print-graphs', action='store_true', help="Print results graph")
    args = parser.parse_args(sys.argv[1:])

    # Obtain dataframe containing sentences and labels; and compute the number of classes
    train_path = os.path.join(args.data_path, args.train_file_name)
    val_path = os.path.join(args.data_path, "CMPD_val_rile.xlsx")
    test_path = os.path.join(args.data_path, "CMPD_test_rile.xlsx")
    train_df, val_df, test_df = read_data(train_path, level_codes=1), read_data(val_path, level_codes=1), read_data(test_path, level_codes=1)
    all_orig_labels = sorted(np.unique(list(train_df["cmp_code"])))
    num_classes = len(all_orig_labels)
    labels_dict = {f'{code}':i for i,code in enumerate(all_orig_labels)}

    # Create tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    model = BertClassifier(tokenizer_name = args.tokenizer, num_classes = num_classes)

    # Define hyperparameters of the training
    EPOCHS = args.epochs
    LR = args.lr
    BS = args.bs
    BS_VAL = 8
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= LR)

    # Split the dataset into training, validation and testing
    np.random.seed(112)
    if args.reduced:
        n_samples_train = len(train_df) if args.reduced_size > len(train_df) else args.reduced_size
        per = n_samples_train / len(train_df)
        n_samples_val, n_samples_test = int(len(val_df)*per), int(len(test_df)*per)
        train_data = train_df.iloc[0:n_samples_train]
        val_data = val_df.iloc[0:n_samples_val]
        test_data = test_df.iloc[0:n_samples_test]
    else:
        train_data = train_df
        val_data = val_df
        test_data = test_df
    print("Sizes of the target train, validation and test datasets:", len(train_data),len(val_data), len(test_data))
    
    ##### TRAIN #####
    train_acc, train_loss, val_acc, val_loss = train(model, tokenizer, labels_dict, train_data, val_data, EPOCHS, loss_fn, optimizer, BS)

    ##### EVALUATE #####
    evaluate(model,tokenizer, labels_dict, test_data)

    # Print training results and save model
    if args.print_graphs:
        file_name=args.data_path.split("/")[-1].split(".")[0]
        title=f"{file_name} metrics"
        print_results(train_acc, train_loss, val_acc, val_loss, args.results_path, title)
    
    print("Saving model...")
    save_path = os.path.join(args.save_model_path, "torchmodel_weights.pth")
    torch.save(model.state_dict(), save_path)
    
