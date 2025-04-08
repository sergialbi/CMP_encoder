import pandas as pd
import numpy as np
import random
from tqdm import tqdm

from transformers import BertTokenizer, BertForPreTraining
import torch

import argparse
import sys
    
def set_random_seed(own_seed):
    torch.manual_seed(own_seed)
    random.seed(own_seed)
    np.random.seed(own_seed)


def load_dataset(args, data_path, tokenizer):
    data = pd.read_excel(data_path)
    texts = data["text"]
    bag = [sentence for paragraph in texts for sentence in paragraph.split('.') if sentence != ""]
    bag_size = len(bag)
    
    sentence_a = []
    sentence_b = []
    label = []

    for paragraph in texts:
        sentences = [sentence for sentence in paragraph.split('.') if sentence != ""]
        num_sentences = len(sentences)
        if num_sentences > 1:
            start = random.randint(0, num_sentences-2)
            sentence_a.append(sentences[start])
            if random.random() > args.nsp_prob: # we concatenate the actual next sentence
                sentence_b.append(sentences[start+1])
                label.append(0)
            else: # we concatenate a random sentence
                sentence_b.append(bag[random.randint(0, bag_size-1)])
                label.append(1)
    
    inputs = tokenizer(sentence_a, sentence_b, return_tensors='pt',
                       max_length=args.max_length, truncation=True, padding='max_length')
    inputs['next_sentence_label'] = torch.LongTensor([label]).T
    inputs['labels'] = inputs.input_ids.detach().clone()

    rand = torch.rand(inputs.input_ids.shape)
    mask_arr = (rand < args.mlm_prob) * (inputs.input_ids != 101) *  (inputs.input_ids != 102) *  (inputs.input_ids != 0)

    for i in range(inputs.input_ids.shape[0]):
        selection = torch.flatten(mask_arr[i].nonzero()).tolist()
        inputs.input_ids[i, selection] = 103

    dataset = PoliticsDataset(inputs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.bs, shuffle=True)
    return loader

class PoliticsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)


def pretrain(args, data_loader, model):
    # Move to device and set the model to train mode
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()

    # Load optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Pretrain model loop
    for epoch in range(args.epochs):
        loop = tqdm(data_loader, leave=True)
        for batch in loop:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            next_sentence_label = batch['next_sentence_label'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, token_type_ids = token_type_ids,
                            attention_mask = attention_mask,
                            next_sentence_label = next_sentence_label,
                            labels = labels)
            loss = outputs.loss
            loss.backward()
            optim.step()

            loop.set_description(f"Epoch {epoch+1}/{args.epochs}")
            loop.set_postfix(loss=loss.item())



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mn5', action='store_true', help="To indicate te execution is done on a cluster")
    parser.add_argument("--data-path", type=str, default="/home/bsc/bsc048726/politics/cmp_encoder/datasets/congress/congress.xlsx", help="The path to the dataset file for training")
    parser.add_argument("--save-model-path", type=str, default="/home/bsc/bsc048726/politics/cmp_encoder/results/intermediate_pretrain/model_pretrained", help="The path to store the intermediate pretrained model")
    parser.add_argument("--checkpoint", type=str, default='/home/bsc/bsc048726/politics/cmp_encoder/models/tokenizers/bert-base-spanish-wwm-cased', help="The BERT tokenizer and pre-trained weights name (download) or path")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum number of tokens in an input sample after padding")
    parser.add_argument("--nsp-prob", type=float, default=0.5, help="Probability that the next sentence is the actual next sentence in NSP")
    parser.add_argument("--mlm-prob", type=float, default=0.15, help="Probability with which tokens are masked in MLM")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for training the model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train the model")
    parser.add_argument("--bs", type=int, default=16, help="Batch size")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for reproducibility")
    args = parser.parse_args(sys.argv[1:])

    if not args.mn5:
        args.checkpoint = "dccuchile/bert-base-spanish-wwm-cased" 
        args.data_path = "/home/sergi/Documents/projectes/politics/cmp_encoder/datasets/congress/congress.xlsx"
        args.save_model_path = "/home/sergi/Documents/projectes/politics/cmp_encoder/models/tokenizers/model_pretrained.pt"

    set_random_seed(args.seed)

    tokenizer = BertTokenizer.from_pretrained(args.checkpoint)
    model = BertForPreTraining.from_pretrained(args.checkpoint)

    data_loader = load_dataset(args, args.data_path, tokenizer)
    print(f"Length of Dataloader: {len(data_loader.dataset)}")
    
    pretrain(args, data_loader, model)

    model.save_pretrained(args.save_model_path)
    #torch.save(model.state_dict(), args.save_model_path)

    
    
    
    