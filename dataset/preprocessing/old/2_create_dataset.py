"""
File to process the Dataset: filter by language, add new samples from existing files, balance categories and apply data augmentation.

@author: Sergi Albiach
@institution: BSC
@date: 22/11/2022
@version: 1.0
"""

import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import os


def histogram(df:pd.DataFrame, category:str="cmp_code", save:bool=False, save_name:str="histogram", show_hist:bool=False):
    """
    Show and save (if desired) the histogram of a DataFrame's column
    @param df: DataFrame from where to extract the column
    return: nothing
    """
    df.groupby([category]).size().plot.bar(figsize=(15, 5))
    if save:
        plt.savefig(f"{save_name}.png", dpi=300)
    if show_hist:
        plt.show()
    else:
        plt.close()


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


def read_data(data_path:str, level_codes:int=1, show_histogram:bool=False, save_histogram:bool=False, type_:str="train") -> pd.DataFrame:
    """
    Function to read the data in data_path, transform its CMP_codes to a level of strictness and show its histogram if desired
    @param data_path: path of the dataset to read
    @param level_codes: level of strictness to be applied [0, 1, 2]
    @param show_histogram: flag to show or not the histogram of the cmp_codes
    return: the df with the 'cmp_code' category transformed by level_codes
    """
    df = pd.read_excel(data_path)
    cmp_codes_transformed = np.asarray([cmp_transform(code, level_codes) for code in df["cmp_code"]], dtype=str)
    df['cmp_code'] = cmp_codes_transformed
    
    histogram(df, category='cmp_code', save=save_histogram, save_name=f"histogram_original_{type_}", show_hist=show_histogram)

    return df

def read_groups(groups_path:str, condition=""):
    df = pd.read_excel(groups_path)
    parties = list(df.iloc[:, 0])
    labels = list(df.iloc[:, 4])
    parties_group = {p:g for p, g in zip(parties, labels)}
    groups_dict = {c:[] for c in np.unique(labels)}
    for party, group in zip(parties, labels):
        l = groups_dict.get(group,[])
        l.append(party)
        groups_dict[group] = l
    return groups_dict, parties_group

def combinations(n,k):
    if n-k >= 0:
        return int((math.factorial(n))/(math.factorial(k)*math.factorial(n-k)))
    else:
        return 1

def compute_quality(n,c):
    # 1 when n > c
    # 0 when n == c
    # -1 when n < c
    q = 1-(n/c)
    return np.round(q,2) if q > 0 else -1

def generate_dataset(df:pd.DataFrame, groups_members:dict, parties_group:dict, samples_per_code:int=5000, context:int=3, seed=123):
    rng = np.random.default_rng(seed)
    df_out = df
    party_codes = df_out["party_code"]
    cmp_codes = df_out["cmp_code"]
    sentences = df_out["text"]
    indexes = np.arange(len(party_codes))
    print("Generating indexes...")
    code_party_index = {c:{p:[indexes[np.array(np.intersect1d(np.where(party_codes==p),np.where(cmp_codes==c)))]] for p in np.unique(party_codes) if indexes[np.array(np.intersect1d(np.where(party_codes==p),np.where(cmp_codes==c)))].size > 0} for c in np.unique(cmp_codes) }

    code_group_index = {}

    print("Generating dictionaries of relations...")
    for c in np.unique(cmp_codes):
        for p in np.unique(party_codes):
            g = parties_group.get(p)
            inds = np.array(code_party_index.get(c).get(p,[]), dtype=int)
            if inds.size > 0:
                tmp = code_group_index.get(c,{})
                if tmp == {}:
                    code_group_index[c] = {}
                tmp2 = np.array(tmp.get(g,[]), dtype=int)
                tmp2 = np.append(tmp2,inds)  
                tmp2.sort()       
                code_group_index[c][g] = tmp2

    df_final = pd.DataFrame(columns=['text','cmp_code'])
    unique_codes = np.unique(cmp_codes)
    print("Merging sentences...")
    for icmp in range(len(unique_codes)):
        cmp_code = unique_codes[icmp]
        sentences_new = []
        candidates = code_group_index.get(str(cmp_code))
        conditions = list(candidates.keys())
        original_samples_per_condition = np.array([len(candidates.get(c)) for c in conditions], dtype=int)
        original_sum = int(np.sum(original_samples_per_condition))
        proportions = original_samples_per_condition/np.sum(original_samples_per_condition)

        target_samples_per_condition = np.array(np.round(proportions*samples_per_code),dtype=int)
        combinations_without_rep = [combinations(s,context) for s in original_samples_per_condition]
        qualities = [str(compute_quality(s,c)) for s,c in zip(target_samples_per_condition, combinations_without_rep)]
        
        for ic in range(len(candidates.keys())):
            c = list(candidates.keys())[ic]
            possible_inds = np.array(list(candidates.get(c)))
            
            if len(possible_inds) > context:
                if len(possible_inds)<500:
                    possible_comb = list(itertools.combinations(possible_inds, context))
                    rng.shuffle(possible_comb)
                    possible_comb = possible_comb[:target_samples_per_condition[ic]]

                else:
                    possible_comb = []
                    for _ in range(target_samples_per_condition[ic]):
                        indx = rng.integers(low=0,high=len(possible_inds),size=context)
                        possible_comb.append(possible_inds[indx])
            else:
                possible_comb = [possible_inds]

            s = [[" ".join(sentences[i] for i in possible_comb[j])] for j in range(len(possible_comb))]
            sentences_new.extend(s)
            
            
        sen = np.array(sentences_new, dtype=object).squeeze()
        lab = np.array([cmp_code]*len(sentences_new), dtype=object)
        mat = np.array([sen,lab]).T
        df_final = pd.concat([df_final, pd.DataFrame(mat, columns=['text','cmp_code'])])
        p = f"({icmp+1}/{len(unique_codes)})"
        qs = "["
        for qi in range(len(qualities)):
            q = qualities[qi]
            if q == 1.0:
                q = int(q)
            qs += f"{q:<4}"
            if qi < len(qualities)-1:
                qs += ", "
            else:
                qs = f"{qs:<17}]"

        s_out = f"{p:<7} {cmp_code:<3} - Quality: {qs} "
        if len(sen) < samples_per_code*0.85:
            s_out += f"- WARNING: significantly less samples than expected ({len(sentences_new)}/{samples_per_code})"
        print(s_out)
    return df_final

def generate_context_dataset(args, name, name_out, type_):
    # Read data
    df = read_data(os.path.join(args.data_path,name), show_histogram=False, save_histogram=args.save_histogram)
    num_original_classes = len(np.unique(list(df_train["cmp_code"])))
    print(f"Number of original classes: {num_original_classes} - Samples: {len(df)}")

    groups_members, parties_group = read_groups(args.groups_path)
    # Process the DataFrame
    df_out = generate_dataset(df, groups_members, parties_group, args.samples_per_code, args.context, seed=123)

    # Show the new dataset statistics
    histogram(df_out, category="cmp_code", save=args.save_histogram, save_name=f"histogram_final_{type_}")
    num_new_classes = len(np.unique(list(df_out["cmp_code"])))
    print(f"Number of new classes: {num_new_classes} - Samples: {len(df_out)}")
    
    out_name = os.path.join(args.data_path,name_out)
    df_out.to_excel(out_name, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=str, default="/home/salbiach/Documents/projectes/politics/cmp_encoder/dataset/preprocessing/", help="The path to the dataset file for training")
    parser.add_argument("--data-path-final", type=str, default="/home/salbiach/Documents/projectes/politics/cmp_encoder/dataset/", help="The final path to the dataset file for training")
    parser.add_argument("--groups-path", type=str, default="/home/salbiach/Documents/projectes/politics/cmp_encoder/proves/preprocessing/groups_parties.xlsx", help="The path to store the figures")
    parser.add_argument('--save-histogram','-sh', action='store_true', help="Save the histograms")
    parser.add_argument('--augment-testing','-at', action='store_true', help="Augment the testing dataset")
    parser.add_argument("--samples-per-code", "-s", type=int, default=5000, help="The number of target samples per code, not more, not less")
    parser.add_argument("--context", "-c", type=int, default=3, help="The number of samples that should be merged")
    args = parser.parse_args(sys.argv[1:])
    
    train_name = "CMPD_train.xlsx"
    train_name_out = "CMPDa_train.xlsx"
    test_name = "CMPD_test.xlsx"
    test_name_out = "CMPDa_test.xlsx"

    generate_context_dataset(args, train_name, train_name_out, type_="train")

    if args.augment_testing:
        generate_context_dataset(args, test_name, test_name_out, type_="test")
    


    
