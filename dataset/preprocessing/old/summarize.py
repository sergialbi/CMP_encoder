"""
File to summarize nearby sentences that have the same category into one.
@author: Sergi Albiach
@institution: BSC
@date: 22/11/2022
@version: 0.1
"""

import argparse
import sys
from transformers import pipeline
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def generate_indexes(df):
    all_codes = np.array(df['cmp_code'], dtype=str)
    indexes = []
    codes_simple = []
    actual = ''
    current_indexes = []
    for i,code in enumerate(all_codes):
        if code == actual:
            current_indexes.append(i)
        else:
            if len(current_indexes) != 0:
                indexes.append(current_indexes)
                codes_simple.append(actual)
            current_indexes = [i]
            actual = code
    return indexes, codes_simple


def generate_summary(model, tokenizer, text, handler, max_len=512, ):
    input_ids = tokenizer([handler(text)], return_tensors="pt", padding="max_length", truncation=True, max_length=max_len)["input_ids"]
    output_ids = model.generate(input_ids=input_ids, max_length=max_len, no_repeat_ngram_size=2, num_beams=4)[0]
    summary = tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return summary


def summarize(df, n_groups=10):
    WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))
    model_name = "csebuetnlp/mT5_multilingual_XLSum"
    #model_name = 'mrm8488/bert2bert_shared-spanish-finetuned-summarization'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    indexes, codes_simple = generate_indexes(df)
    summaries = np.empty(n_groups, dtype=object)
    rows = []

    for i in tqdm(range(n_groups)):
        df_same_code = df.iloc[indexes[i]]
        text = " ".join( map(str,df_same_code['text']))
        if len(df_same_code) > 1:
            summaries[i] = generate_summary(model, tokenizer, text, WHITESPACE_HANDLER)
        else:
            summaries[i] = text
        
        mi = np.unique(df_same_code["manifesto_id"])[0]
        c = np.unique(df_same_code["country"])[0]
        d = np.unique(df_same_code["date"])[0]
        l = np.unique(df_same_code["language"])[0]
        pc = np.unique(df_same_code["party_code"])[0]
        pn = np.unique(df_same_code["party_name"])[0]
        si = " ".join(map(str,list(df_same_code["sentence_id"])))
        rows.append([mi, c, d, l ,pc, pn, si, summaries[i], codes_simple[i]])

    new_df = pd.DataFrame(rows)
    new_df.columns = df.columns
    new_df.head()

    return new_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=str, default="/home/salbiach/Documents/projectes/politics/cmp_encoder/dataset/preprocessing/CMPD_Spain_modified.xlsx", help="The path to the dataset file for training")
    args = parser.parse_args(sys.argv[1:])

    df = pd.read_excel(args.data_path)
    df = summarize(df, len(df))
    file_name = args.data_path.split('.')[0]+"_summarized.xlsx"
    df.to_excel(file_name, index=False)


    
