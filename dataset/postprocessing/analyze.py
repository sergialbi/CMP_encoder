"""
Script to translate the sentences from one dataset into the desired language.

@author: Sergi Albiach
@institution: BSC
@date: 22/11/2022
@version: 1.0
"""

import pandas as pd
import numpy as np
import argparse
import sys

def print_stats_df(df):
    subcategories = list(df["cmp_code"])
    supercategories = list(df["supercategory"])

    table = pd.DataFrame(columns = ["frequency", "% frequency", "#subcategories", "mean +- std"],
        index=["(1) External Relations", "(2) Freedom, Democracy",
                "(3) Political System", "(4) Economy",
                "(5) Welfare, Quality of Life", "(6) Fabric of Society",
                "(7) Social Groups", "OVERALL"])

    super_dict = {c:supercategories.count(c) for c in np.unique(supercategories)}
    super_dict = dict(sorted(super_dict.items()))

    supercategories = np.asarray(supercategories)
    subcategories = np.asarray(subcategories)
    n_sub_dict = {}
    for i in range(len(np.unique(supercategories))):
        c = np.unique(supercategories)[i]

        subcats = subcategories[np.where(supercategories==str(c))[0]]
        subcats_unique = np.unique(subcats)
        n_sub_dict[c] = len(subcats_unique)
        frequencies = np.array([len(np.where(subcats==c)[0]) for c in subcats_unique], dtype=int)
        mean_freq = int(np.round(np.mean(frequencies)))
        sd = int(np.round(np.std(frequencies)))

        table_f = super_dict.get(c)
        table_f_p = str(np.round(super_dict.get(c)/np.sum(list(super_dict.values()))*100,2))+"%"
        table_num_sub = n_sub_dict.get(c)
        table_mean = f"{mean_freq} +- {sd}"
        table.iloc[i] = [table_f, table_f_p, table_num_sub, table_mean]
    table.iloc[7] = [np.sum(np.array(table["frequency"])[:-1]),"100%", np.sum(np.array(table["#subcategories"])[:-1]), None]
    print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", "-f", type=str, nargs='*', default=["/home/salbiach/Documents/projectes/politics/cmp_encoder/dataset/v5/CMPDa10_c2_train.xlsx", "/home/salbiach/Documents/projectes/politics/cmp_encoder/dataset/v5/CMPD_val.xlsx", "/home/salbiach/Documents/projectes/politics/cmp_encoder/dataset/v5/CMPD_test.xlsx"], help="The dataset files to analyze.")
    #parser.add_argument("--files", "-f", type=str, nargs='*', default=["/home/salbiach/Documents/projectes/politics/cmp_encoder/dataset/v5/CMPD_test.xlsx"], help="The dataset files to analyze.")
    args = parser.parse_args(sys.argv[1:])

    files = []
    for f in args.files:
        df = pd.read_excel(f)
        df = df[["text","cmp_code"]]
        codes = [str(c)[0] for c in df["cmp_code"]]
        df["supercategory"] = codes
        files.append(df)
    
    final_df = pd.DataFrame()
    for i in range(len(files)):
        df = files[i]
        final_df = pd.concat([final_df, df])
        print(args.files[i].split("/")[-1][:-5])
        print_stats_df(df)
        print("\n\n")
    
    if len(files)>1:
        print("TOTAL DATASET")
        print_stats_df(final_df)
    
    