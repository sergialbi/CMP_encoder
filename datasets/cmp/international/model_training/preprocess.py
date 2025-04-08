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
from tqdm import tqdm
import nlpaug.augmenter.word as naw
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
        return str(cmp)[0:3]
    elif level_codes == 2: #GENERAL CATEGORY (103.2 --> 1)
        return str(cmp)[0]
    else:
        print(f"[ERROR] An invalid level of strictness ({level_codes}) was specified! --> level_codes = [0, 1, 2]")
    return ""


def read_data(data_path:str, level_codes:int=1, show_histogram:bool=False, save_histogram:bool=False, type_:str="") -> pd.DataFrame:
    """
    Function to read the data in data_path, transform its CMP_codes to a level of strictness and show its histogram if desired
    @param data_path: path of the dataset to read as csv
    @param level_codes: level of strictness to be applied [0, 1, 2]
    @param show_histogram: flag to show or not the histogram of the cmp_codes
    return: the df with the 'cmp_code' category transformed by level_codes
    """
    df = pd.read_csv(data_path)
    delete = []
    cmp_codes_transformed = np.asarray([cmp_transform(code, level_codes) if str(code) != "nan" else delete.append(i) for i,code in enumerate(df["cmp_code"])], dtype=str)
    df['cmp_code'] = cmp_codes_transformed
    df.drop(delete, inplace=True)
    df.index = range(len(df.index))

    name = "histogram_original"
    if type_ != "":
        name += f"_{type_}"
    
    histogram(df, category='cmp_code', save=save_histogram, save_name=name, show_hist=show_histogram)

    return df


def remove_categories(df:pd.DataFrame, black_list:list=[], dimension:str="all", simplify_dimension:bool=False) -> pd.DataFrame:
    """
    Function to remove those categories that are present in 'black_list' and/or if those categories have less than low_bound samples.
    Also remove random instances from those categories having more than 'high_bound' samples.
    @param df: DataFrame to cut the categories from
    @param black_list: list of categories that will be removed from the DataFrame
    @param dimension: programatic dimension to select labels
    @param simplify_dimension: flag to simplify labels to supra-categories
    return: the clean DataFrame without categories in 'black_list'
    """
    black_list = [b for b in black_list if b != ""]

    # Categories in Black List
    if len(black_list)>0:
        print(f"[APPLIED] Removing categories: {black_list})")
        df = df[[(v not in black_list) for v in list(df['cmp_code'])]]
        df.index = range(len(df.index))

    df = df.copy()
    # Custom Programatic Dimension Selection
    if dimension == "rile":
        right_list = ["104", "201", "203", "305", "401", "402", "407", "414", "505", "601", "603", "605", "606"]
        left_list =  ["103", "105", "106", "107", "403", "404", "406", "412", "413", "504", "506", "701", "202"]
        white_list = right_list + left_list
        dimension_dict = {"right": right_list, "left": left_list}
        dimension_dict = {vn:k for k,v in dimension_dict.items() for vn in v}

    elif dimension == "all":
        pass

    if dimension != "all":
        print(f"[APPLIED] Selecting categories for '{dimension}'")
        df.loc[:, "cmp_code"] = list([c if c in white_list else "other" for c in list(df['cmp_code'])])
 
    if simplify_dimension:
        df.loc[:, "cmp_code"] = list(map(lambda c: dimension_dict.get(c,"other"), list(df['cmp_code'])))

    return df


def remove_short_sentences(df:pd.DataFrame, min_len:int=3) -> pd.DataFrame:
    """
    Function to keep those samples whose 'text' length is greather than 'min_len' words
    @param df: DataFrame from which to remove short sentences
    @param min_len: minimum length of the sample's 'text' field to be considered
    return: the DataFrame without samples with less than 'min_len' words
    """
    if min_len > 0:
        print(f"[APPLIED] Removing short sentences (<{min_len} words)")
        sentences = np.asarray([t.split() for t in df['text']], dtype=object)
        inds = []
        for i,s in enumerate(sentences):
            if len(s) >= min_len:
                inds.append(i)
        print(f"[INFO] Removed {len(df) - len(inds)} short sentences.")
        df = df.iloc[inds]
        df.index = range(len(df.index))
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

def augment_context(df:pd.DataFrame, groups_path:str, samples_per_code:int=5000, context:int=2, seed:int=123, verbose:bool=False, aug_perc:float=0.1, mixed:bool=True):
    print(f"[APPLIED] Generating new dataset by increasing sentence context.")
    
    _, parties_group = read_groups(groups_path)
    rng = np.random.default_rng(seed)
    df_out = df.copy()
    party_codes = df_out["party_code"]
    cmp_codes = df_out["cmp_code"]
    sentences = df_out["text"]
    indexes = np.arange(len(party_codes))

    # For each 'code' --> for each 'party' --> list of samples belonging to that party and that code
    code_party_index = {c:{p:[indexes[np.array(np.intersect1d(np.where(party_codes==p),np.where(cmp_codes==c)))]] for p in np.unique(party_codes) if indexes[np.array(np.intersect1d(np.where(party_codes==p),np.where(cmp_codes==c)))].size > 0} for c in np.unique(cmp_codes) }

    # For each 'code' --> for each 'group' --> list of samples belonging to that code and that group
    code_group_index = {}
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

    df_final = pd.DataFrame(columns=['text','cmp_code', "augmented"])
    unique_codes = np.unique(cmp_codes)

    if verbose:
        r = range(len(unique_codes))
    else:
        r = tqdm(range(len(unique_codes)))
    
    # FOR EACH UNIQUE CODE
    for icmp in r:
        cmp_code = unique_codes[icmp]
        sentences_new = []
        candidates = code_group_index.get(str(cmp_code)) # dictionary of 'groups' --> samples for the given 'code'
        conditions = list(candidates.keys()) # the names of the groups conditions (i.e. LEFT, CENTER, RIGHT)
        original_samples_per_condition = np.array([len(candidates.get(c)) for c in conditions], dtype=int) # number of samples per group
        proportions = original_samples_per_condition/np.sum(original_samples_per_condition) # proportions of samples per group relative to group total
        target_samples_per_condition = np.array(np.round(proportions*samples_per_code),dtype=int) # assign total 'code' samples proportionally to each group
        
        combinations_without_rep = [combinations(s,context) for s in original_samples_per_condition] #potential combinations without repetition
        qualities = [str(compute_quality(s,c)) for s,c in zip(target_samples_per_condition, combinations_without_rep)] #see if it is possible to generate it
        augmented = []
        # FOR EACH GROUP (condition)
        for ic in range(len(conditions)):
            c = conditions[ic]
            possible_inds = np.array(list(candidates.get(c))) # the indices for the given 'group'
            
            if len(possible_inds) > context: # (1) proceed if there are more samples than desired context | otherwise append just the 'possible_inds'
                if len(possible_inds)<500: # (2) if few samples, generate all possible combinations and randomly choose 
                    possible_comb = list(itertools.combinations(possible_inds, context))
                    rng.shuffle(possible_comb)
                    possible_comb = possible_comb[:target_samples_per_condition[ic]]
                else: # (2) too many 'possible_inds', randomly select samples (WARNING --> repetitions could therefore happen)
                    possible_comb = []
                    # FOR EVERY NEW_SAMPLE THAT SHOULD BE GENERATED
                    for _ in range(target_samples_per_condition[ic]):
                        indx = rng.integers(low=0,high=len(possible_inds),size=context) # select 'context' number of sentences
                        possible_comb.append(possible_inds[indx]) # append new sample for the group

                if mixed and aug_perc < 1:
                    n_current_samples = len(possible_comb)
                    num_original = len(possible_inds)
                    #num_original = int(n_current_samples*(1-aug_perc))
                    num_augmented = min(int(num_original*(aug_perc)/(1-aug_perc)),n_current_samples)

                    if num_augmented > 0:
                        indx = rng.integers(low=0,high=len(possible_comb),size=num_augmented) # select 'context' number of sentences
                        possible_comb = list(np.array(possible_comb)[indx])
                        tmp1 = np.ones(len(possible_comb))
                        possible_comb.extend([pi] for pi in possible_inds)
                        tmp0 = np.zeros(len(possible_inds))
                        augmented = np.concatenate([augmented, tmp1, tmp0])
                    else:
                        possible_comb = [[pi] for pi in possible_inds]
                        tmp0 = np.zeros(len(possible_inds))
                        augmented = np.concatenate([augmented, tmp0])

                if verbose:
                    print(f"Augmented: {num_augmented} - Original: {num_original}")

            else: # (1) very few samples to generate context, append them directly.
                possible_comb = [[pi] for pi in possible_inds]
                tmp0 = np.zeros(len(possible_inds))
                augmented = np.concatenate([augmented, tmp0])

            # Format new samples and append them to final dataset
            s = [[" ".join(sentences[i] for i in possible_comb[j])] for j in range(len(possible_comb))]
            sentences_new.extend(s)
            
        # Print the qualities for the 'code'
        sen = np.array(sentences_new, dtype=object).squeeze()
        lab = np.array([cmp_code]*len(sentences_new), dtype=object)
        mat = np.array([sen, lab, augmented]).T
        df_final = pd.concat([df_final, pd.DataFrame(mat, columns=['text','cmp_code', "augmented"])])
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
        if verbose:
            print(s_out)
    return df_final


def split(df, max_cat_size, balanced = False, train_perc=0.7, val_perc=0.2, seed=123, random_test=False):

    rng = np.random.default_rng(seed)
    test_perc = 1-train_perc-val_perc
    
    if balanced:
        max_num_test = int(max_cat_size*test_perc)
        max_num_val = int(max_cat_size*val_perc)
        max_num_train = int(max_cat_size*train_perc)

        # Dictionary with the indices of the samples per each category
        samples_per_code = dict(df['cmp_code'].value_counts())
        codes = np.asarray(df['cmp_code'])
        indices_per_category = {code: np.where(codes == code)[0] for code in samples_per_code.keys()}
        train_inds, val_inds, test_inds = [], [], []

        if not random_test:
            print("Missing information for spliting based on manifesto index")

        else:
            for code, inds in indices_per_category.items():
                n_samples = len(inds)
                if n_samples < max_cat_size:
                    num_test = int(n_samples*test_perc)
                    num_val = int(n_samples*val_perc)
                    num_train = n_samples - num_test - num_val
                else:
                    num_test = max_num_test
                    num_val = max_num_val
                    num_train = max_num_train
                
                rng.shuffle(inds)
                test_inds.extend(inds[:num_test])
                val_inds.extend(inds[num_test:num_test+num_val])
                train_inds.extend(inds[num_test+num_val:num_test+num_val+num_train])
        
            df_test = df.iloc[test_inds]
            df_val = df.iloc[val_inds]
            df_train = df.iloc[train_inds]
    else:
        num_test = int(len(df)*test_perc)
        num_val = int(len(df)*val_perc)
        num_train = int(len(df)*train_perc)
        inds = np.arange(len(df))
        rng.shuffle(inds)
        df_test = df.iloc[inds[:num_test]]
        df_val = df.iloc[inds[num_test:num_test+num_val]]
        df_train = df.iloc[inds[num_test+num_val:num_test+num_val+num_train]]

    total_length = len(df_train) + len(df_val) + len(df_test)

    print(f"[APPLIED] SPLITS: testing ({np.round(len(df_test)/total_length*100,2)}% - {len(df_test)}), validation ({np.round(len(df_val)/total_length*100,2)}% - {len(df_val)}) and training ({np.round(len(df_train)/total_length*100,2)}% - {len(df_train)}) data.")

    df_train.reset_index(drop=True, inplace=True)
    df_val.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    return df_train, df_val, df_test


def show_statistics(args, df, type_="",  hist_name = "histogram", save=True, save_out_name="CMPDa.csv"):
    histogram(df, category="cmp_code", save=args.save_histogram, save_name=hist_name)

    num_new_classes = len(np.unique(list(df["cmp_code"])))
    print(f"Number of classes in {type_}: {num_new_classes} - Samples: {len(df)}")

    if save:
        df.to_csv(save_out_name, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=str, default="/home/sergi/Documents/projectes/politics/new_cmp_encoder/datasets/model_training/", help="The path to the dataset file for training")
    parser.add_argument("--new-datasets-save-path", type=str, default="/home/sergi/Documents/projectes/politics/new_cmp_encoder/datasets/model_training/", help="The path to the dataset file for training")
    parser.add_argument('--save-histogram','-sh', action='store_true', help="Save the histograms")
    parser.add_argument("--remove-category", "-rc", type=str, nargs='*', default="H", help="The categories to leave out in the final data")
    parser.add_argument("--balanced", "-b", action='store_true', help="Balance categories by adding/removing samples or cutting categories if do not comply with minimum requirements.")
    parser.add_argument("--max-cat-size", "-mx", type=int, default=5000, help="Maximum number of samples per category")
    parser.add_argument("--min-cat-size", "-mn", type=int, default=500, help="Minimum number of samples per category")
    parser.add_argument("--groups-path", type=str, default="/home/sergi/Documents/projectes/politics/cmp_encoder/datasets/cmp/national/preprocessing/spectrum_data/groups_parties.xlsx", help="The path to store the figures")
    parser.add_argument("--context", "-c", type=int, default=1, help="The number of samples that should be merged")
    parser.add_argument('--verbose','-v', action='store_true', help="Output the dataset creation")
    parser.add_argument("--aug_perc", "-p", type=float, default=0.1, help="The percentage of augmented samples on training dataset")
    args = parser.parse_args(sys.argv[1:])


    args.results_path = args.new_datasets_save_path


    # Format the black list of categories to be removed
    black_list = []
    for c in args.remove_category:
        black_list.append(c)

    # Read data
    data_name = "2_CMPD_training.csv"
    data_name_out = f"CMPDb" if args.balanced else "CMPD"
    if args.context > 1:
        data_name_out = f"{data_name_out}a{int(args.aug_perc*100)}" 
    data_path = os.path.join(args.data_path, data_name)
    df = read_data(data_path, show_histogram=False, save_histogram=args.save_histogram)
    num_original_classes = len(np.unique(list(df["cmp_code"])))
    print(f"Number of original classes: {num_original_classes} - Samples: {len(df)}")
    
    
    # REMOVE SHORT SENTENCES, REMOVE CATEGORIES and SPLIT SETS
    df_out = df.copy()
    df_out = remove_short_sentences(df_out, min_len=5)
    df_out = remove_categories(df, black_list, dimension="all", simplify_dimension=False)

    train_per, val_per, test_per = 0.7, 0.2, 0.1
    df_train, df_val, df_test = split(df_out, args.max_cat_size, balanced = args.balanced, train_perc = train_per, val_perc = val_per, seed=123579, random_test=True)
    total_samples = len(df_train) + len(df_val) + len(df_test)
    
    
    # AUGMENTATION BY CONTEXT CONCATENATION
    if args.context > 1:
        mixed = args.aug_perc<1
        df_train = augment_context(df_train.copy(), args.groups_path, args.max_cat_size, args.context, seed=123, verbose=args.verbose, mixed=mixed, aug_perc=args.aug_perc)

    # Show the new dataset statistics
    total_samples = len(df_train) + len(df_val) + len(df_test)
    print(f"Train: {len(df_train)} ({np.round(len(df_train)/total_samples*100,2)}%) - Validation: {len(df_val)} ({np.round(len(df_val)/total_samples*100,2)}%) - Testing: {len(df_test)} ({np.round(len(df_test)/total_samples*100,2)}%)")
    show_statistics(args, df_train, type_="train", hist_name = data_name_out+"_histogram_train",save=True, save_out_name = os.path.join(args.new_datasets_save_path, data_name_out+f"_c{args.context}_train.csv"))
    if args.context == 1:
        show_statistics(args, df_val, type_="val", hist_name = data_name_out+"_histogram_val", save=True,  save_out_name = os.path.join(args.new_datasets_save_path, data_name_out+f"_val.csv"))
        show_statistics(args, df_test, type_="test", hist_name = data_name_out+"_histogram_test", save=True,  save_out_name = os.path.join(args.new_datasets_save_path, data_name_out+f"_test.csv"))
    
