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
        return str(cmp).split(".")[0]
    elif level_codes == 2: #GENERAL CATEGORY (103.2 --> 1)
        return str(cmp)[0]
    else:
        print(f"[ERROR] An invalid level of strictness ({level_codes}) was specified! --> level_codes = [0, 1, 2]")
    return ""


def read_data(data_path:str, level_codes:int=1, show_histogram:bool=False, save_histogram:bool=False, type_:str="") -> pd.DataFrame:
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

    name = "histogram_original"
    if type_ != "":
        name += f"_{type_}"
    
    histogram(df, category='cmp_code', save=save_histogram, save_name=name, show_hist=show_histogram)

    return df


def balance_by_cut(df:pd.DataFrame, balanced_flag:bool=False, low_bound:int=0, high_bound:int=np.inf, show_before_cut:bool=False, scale:float=0) -> pd.DataFrame:
    """
    Function to remove those categories that are present in 'black_list' and/or if those categories have less than low_bound samples.
    Also remove random instances from those categories having more than 'high_bound' samples.
    @param df: DataFrame to cut the categories from
    @param balanced_flag: wether to apply the high and low bound to cut categories
    @param low_bound: minimum number of samples to be considered a valid category
    @param high_bound: maximum number of samples that a category can have
    @param show_before_cut: show dictionary with count of samples per categories
    return: the clean DataFrame balanced if desired between 'low_bound' and 'high_bound' samples per category.
    """
    if balanced_flag:
        print(f"[APPLIED] Removing categories with less than {low_bound} samples")
        print(f"[APPLIED] Removing samples to fit {high_bound} samples per category")
        samples_per_code = dict(df['cmp_code'].value_counts())
        if scale != 0:
            high_bound = int(high_bound*scale)
        target_low_labels = [code for code, n_samples in samples_per_code.items() if n_samples < low_bound]
        target_high_labels = [code for code, n_samples in samples_per_code.items() if n_samples > high_bound]

        to_print = np.sort(np.asarray(target_low_labels))
        print("Labels with less than target samples: ", to_print)

        if show_before_cut:
            print(dict(sorted(df.groupby(['cmp_code']).size().items(), key=lambda item: item[1])))

        df = df[[(v not in target_low_labels) for v in list(df['cmp_code'])]]
        codes = np.asarray(df['cmp_code'])
        cut_indexes = []

        # Select randomly the indexes of the rows to be removed per category whose count was greater than 'high_bound'
        for high_label in target_high_labels:
            indexes = np.asarray(np.where(codes == high_label))[0]
            np.random.shuffle(indexes)
            cut_indexes.extend(indexes[high_bound:])

        # Drop rows from the DataFrame 
        df = df.drop(df.index[cut_indexes])
        df.index = range(len(df.index))

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


def filter_language(df:pd.DataFrame, language:str='spanish', output_foreign_langs:bool=False) -> pd.DataFrame:
    """
    Function that filters those samples that are not in the specified language
    @param df: DataFrame from which to extract sentences
    @param language: language to filter the samples
    @param output_foreign_langs: flag to export the samples that are not in the specified language to excel
    return: the filter dataset by language
    """
    print(f"[APPLIED] Extracting sentences in {language}")
    df_target = df[df['language']==language]
    df_target.index = range(len(df_target.index))

    if output_foreign_langs:
        print(f"[APPLIED] Writing to output those samples that are not in {language}")
        df_to_translate = df[df['language']!=language]
        df_to_translate.to_excel("foreign_langs.xlsx", index=False)
        
    return df_target


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


def data_augmentation(df:pd.DataFrame, size:int=0, model_path_own:str='dccuchile/bert-base-spanish-wwm-cased', balanced_flag:bool=False, low_bound:int=0, high_bound:int=np.inf) -> pd.DataFrame: 
    """
    Function to apply Data Augmentation to the DataFrame
    @param df: DataFrame to apply DA to
    @param size: number of augmented samples to add to the original DataFrame
    """
    if size>0:
        print(f"[APPLIED] Generating new samples by using Data Augmentation on the DataFrame. [WARNING] NEEDS REVISION!!!")
        rng = np.random.default_rng(12345)

        if balanced_flag:
            samples_per_code = dict(df['cmp_code'].value_counts())
            samples_per_code = dict(sorted(samples_per_code.items(), key=lambda item: item[1]))
            inds_per_category = dict(df.groupby(['cmp_code']).groups)
            target_low_labels = [code for code, n_samples in samples_per_code.items() if n_samples < low_bound]
            target_high_labels = [code for code, n_samples in samples_per_code.items() if n_samples > high_bound]
            
            new = 0
            size_low_bound = 0
            pass_cut = 0
            da_indx = []
            # From the most populated label that doesn't comply with minimum bound (to try to ensure they fit)
            for cat in reversed(target_low_labels):
                to_add = low_bound - samples_per_code[cat]
                size_low_bound += to_add
                if to_add <= (size-new): # Try to generate enough samples to fit the category in bound
                    new += to_add
                    pass_cut += 1
                    inds_cat = list(inds_per_category[cat])
                    rng.shuffle(inds_cat)
                    n_blocks = to_add // len(inds_cat)
                    da_indx.extend(inds_cat*n_blocks)
                    rest = to_add%len(inds_cat)
                    da_indx.extend(inds_cat[:rest])
                else:
                    break
            if new == 0:
                print(f"[WARNING] No samples were added in a balanced way. Generating them RANDOMLY.\n"+
                        "          Try reducing the 'low_bound' or increasing the 'size' of the new DA samples.")
            elif new < size_low_bound:
                print(f"[WARNING] Not all categories will fit the 'low_bound' cut after DA, only {pass_cut}/{len(target_low_labels)} (out of the already below bound categories)")

            if new < size: # If there are still samples to generate, do it randomly.
                inds = np.arange(0,len(df))
                rng.shuffle(inds)
                da_indx.extend(inds[:(size-new)])
        else:
            inds = np.arange(0,len(df))
            rng.shuffle(inds)
            da_indx = inds[:(size)]

        aug_ins = naw.ContextualWordEmbsAug(model_path='dccuchile/bert-base-spanish-wwm-cased', action="insert")
        aug_sub = naw.ContextualWordEmbsAug(model_path='dccuchile/bert-base-spanish-wwm-cased', action="substitute")
        dict_augs = {0:aug_ins, 1:aug_sub}

        for i in tqdm(da_indx):
            text = df.iloc[i]["text"]
            index_aug = rng.integers(0, len(dict_augs), 1)[0]
            augmented_text = dict_augs.get(index_aug).augment(text)
            row = df.iloc[i].copy()
            row['text'] = augmented_text
            df.loc[len(df.index)] = row
        augmented_col = np.zeros(len(df))
        augmented_col[-size:] = 1
        df["augmented"] = augmented_col
        df.index = range(len(df.index))
    return df


def import_from_files(df:pd.DataFrame, files_paths:list=[]) -> pd.DataFrame:
    """
    Add to the DataFrame samples from other datasets
    @param df: original DataFrame to append the new samples
    @param files_paths: paths to the external files to add samples from
    return: the df with extended samples
    """
    files_paths = [f for f in files_paths if f != ""]
    if len(files_paths)>0 and files_paths[0]!="":
        print(f"[APPLIED] Importing new samples from external files.")
        new_samples = 0
        original_size = len(df)
        for path in files_paths:
            if path != '':
                df_to_add = read_data(path, show_histogram=False)
                df_to_add[df_to_add['cmp_code']=='0']='000'
                print(f"[INFO] Adding {len(df_to_add)} samples from \'{path}\'")
                new_samples+=len(df_to_add)
                df = pd.concat([df, df_to_add], ignore_index=True)
        external_column = np.zeros(original_size+new_samples)
        external_column[original_size:] = 1
        df["external"] = external_column
        df = df.drop('index', axis=1)
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


def split(df, train_perc=0.7, val_perc=0.2, seed=123, random_test=False):

    rng = np.random.default_rng(seed)
    #df_train = df.copy()

    num_test = int(len(df)*(1-train_perc-val_perc))
    num_val = int(len(df)*val_perc)
    num_train = int(len(df)*train_perc)

    if not random_test:
        testing_manifesto_ids = ["33610_200803", "33220_201111", "33210_201606", "33710_201911", "33611_200003", "33610_201904", "33907_201904"] # 2 extra
        manifesto_ids = list(df["manifesto_id"])
        test_inds = [i for i in range(len(manifesto_ids)) if (manifesto_ids[i] in testing_manifesto_ids)]
        num_test_new = int(len(test_inds))
        df_test = df.iloc[test_inds]
        df_train = df.drop(df.index[test_inds])
        inds = np.arange(len(df_train))
        rng.shuffle(inds)
        #num_val = num_val + (num_test-num_test_new)
        val_inds = inds[:num_val]
        df_val = df_train.iloc[val_inds]
        df_train = df_train.drop(df_train.index[val_inds])

    else:
        inds = np.arange(len(df))
        rng.shuffle(inds)
        test_inds = inds[:num_test]
        val_inds = inds[num_test:num_test+num_val]
        train_inds = inds[num_test+num_val:]
        df_test = df.iloc[test_inds]
        df_val = df.iloc[val_inds]
        df_train = df.iloc[train_inds]

    total_length = len(df_train) + len(df_val) + len(df_test)

    print(f"[APPLIED] Splitting testing ({np.round(len(df_test)/total_length*100,2)}%), validation ({np.round(len(df_val)/total_length*100,2)}%) and training ({np.round(len(df_train)/total_length*100,2)}%) data.")

    df_train.reset_index(drop=True, inplace=True)
    df_val.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    return df_train, df_val, df_test


def show_statistics(args, df, type_="", save=True, out_name="CMPDa.xlsx"):
    histogram(df, category="cmp_code", save=args.save_histogram, save_name=f"histogram_{type_}")

    num_new_classes = len(np.unique(list(df["cmp_code"])))
    print(f"Number of new classes in {type_}: {num_new_classes} - Samples: {len(df)}")
    if args.dimension != "all":
        dim_name = args.dimension
        if args.simplify_dimension:
            dim_name += "_simp"
        out_name = out_name.split('.')[0]+f"_{dim_name}.xlsx"

    if save:
        df.to_excel(out_name, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=str, default="/home/salbiach/Documents/projectes/politics/cmp_encoder/dataset/preprocessing/CMPD_raw/", help="The path to the dataset file for training")
    parser.add_argument("--data-path-final", type=str, default="/home/salbiach/Documents/projectes/politics/cmp_encoder/dataset/", help="The path to the dataset file for training")
    parser.add_argument("--results-path", type=str, default="/home/salbiach/Documents/projectes/politics/cmp_encoder/results/", help="The path to store the figures")
    parser.add_argument('--save-histogram','-sh', action='store_true', help="Save the histograms")
    parser.add_argument("--language", "-l", type=str, default="spanish", help="The language of the output sentences (either to translate or to filter)")
    parser.add_argument("--remove-category", "-rc", type=str, nargs='*', default="H", help="The categories to leave out in the final data")
    parser.add_argument("--balanced", "-b", action='store_true', help="Balance categories by adding/removing samples or cutting categories if do not comply with minimum requirements.")
    parser.add_argument("--max-cat-size", "-mx", type=int, default=5000, help="Maximum number of samples per category")
    parser.add_argument("--min-cat-size", "-mn", type=int, default=50, help="Minimum number of samples per category")
    parser.add_argument("--import-from-files","-if", type=str, nargs='*', default=["CMPD_raw/CMPD_Spain_foreign_translated.xlsx"], help="The dataset files from which samples will be loaded.")
    parser.add_argument("--augmented-samples", "-as", type=int, default=0, help="Numer of samples to apply data augmentation (insertions, deletions, substitutions)")
    parser.add_argument("--dimension", "-d", type=str, default="all", choices=["all", "rile"], help="The programatic dimension to select")
    parser.add_argument("--simplify-dimension", "-sd", action='store_true', help="Reduce the programatic dimensions to big categories")
    parser.add_argument("--groups-path", type=str, default="/home/salbiach/Documents/projectes/politics/cmp_encoder/dataset/preprocessing/groups_parties.xlsx", help="The path to store the figures")
    parser.add_argument('--augment-validation','-av', action='store_true', help="Augment the validation dataset")
    parser.add_argument('--augment-testing','-at', action='store_true', help="Augment the testing dataset")
    parser.add_argument("--samples-per-code", "-s", type=int, default=5000, help="The number of target samples per code, not more, not less")
    parser.add_argument("--context", "-c", type=int, default=2, help="The number of samples that should be merged")
    parser.add_argument('--verbose','-v', action='store_true', help="Output the dataset creation")
    parser.add_argument("--aug_perc", "-p", type=float, default=0.1, help="The percentage of augmented samples on training dataset")
    parser.add_argument('--match-test','-mt', action='store_true', help="Filter validation and training datasetes to match test labels only")
    parser.add_argument('--random-testing','-rt', action='store_true', help="Random split instead of predefined test.")
    args = parser.parse_args(sys.argv[1:])

    # Format the black list of categories to be removed
    black_list = []
    for c in args.remove_category:
        black_list.append(c)

    # Read data
    data_name = "CMPD_Spain_raw.xlsx"
    data_name_out = f"CMPDa{int(args.aug_perc*100)}" if args.context > 1 else "CMPD"
    data_path = os.path.join(args.data_path, data_name)
    df = read_data(data_path, show_histogram=False, save_histogram=args.save_histogram)
    num_original_classes = len(np.unique(list(df["cmp_code"])))
    print(f"Number of original classes: {num_original_classes} - Samples: {len(df)}")

    # IMPORT --> FILTER LANGUAGE --> REMOVE SHORT SENTENCES
    df_out = df.copy()
    df_out = import_from_files(df_out, args.import_from_files)
    df_out = filter_language(df_out, args.language, output_foreign_langs=False)
    df_out = remove_short_sentences(df_out, min_len=3)
    #df_out = data_augmentation(df_out, size = args.augmented_samples, balanced_flag=args.balanced, low_bound = args.min_cat_size, high_bound = args.max_cat_size)
    

    # SPLIT IN TRAINING, VALIDATION AND TESTING
    train_per, val_per, test_per = 0.7, 0.2, 0.1
    df_train, df_val, df_test = split(df_out, train_perc = train_per, val_perc = val_per, seed=123579, random_test=args.random_testing)

    total_samples = len(df_train) + len(df_val) + len(df_test)
    train_samples_per_category = min(args.samples_per_code, args.max_cat_size)
    val_samples_per_category = int(train_samples_per_category*val_per)
    test_samples_per_category = int(train_samples_per_category*test_per)
    

    # AUGMENT CONTEXT
    if args.context > 1:
        mixed = args.aug_perc<1
        df_train = augment_context(df_train.copy(), args.groups_path, train_samples_per_category, args.context, seed=123, verbose=args.verbose, mixed=mixed, aug_perc=args.aug_perc)
        if args.augment_validation:
            df_val = augment_context(df_test.copy(), args.groups_path, val_samples_per_category, args.context, seed=123, verbose=args.verbose, mixed=mixed, aug_perc=args.aug_perc)
        if args.augment_testing:
            df_test = augment_context(df_test.copy(), args.groups_path, test_samples_per_category, args.context, seed=123, verbose=args.verbose, mixed=mixed, aug_perc=args.aug_perc)

    print(f"\nLENGTH OF TRAIN DATASET: {len(df_train)}")
    # REMOVE SPECIFIC CATEGORIES + FILTER BY DIMENSION --> CUT THE CATEGORIES TO FIT RANGE
    print("[TRAIN]")
    df_train = remove_categories(df_train, black_list, dimension=args.dimension, simplify_dimension=args.simplify_dimension)
    df_train = balance_by_cut(df_train, args.balanced, args.min_cat_size, train_samples_per_category)

    print(f"\nLENGTH OF VALIDATION DATASET: {len(df_val)}")
    print("[VALIDATION]")
    df_val = remove_categories(df_val, black_list, dimension=args.dimension, simplify_dimension=args.simplify_dimension)
    df_val = balance_by_cut(df_val, args.balanced, args.min_cat_size, val_samples_per_category)

    print(f"\nLENGTH OF TESTING DATASET: {len(df_test)}")
    print("[TESTING]")
    df_test = remove_categories(df_test, black_list, dimension=args.dimension, simplify_dimension=args.simplify_dimension)
    df_test = balance_by_cut(df_test, args.balanced, args.min_cat_size, test_samples_per_category)

    if args.match_test:
        # FILTER DATASETS TO FIT THE TESTING LABELS ONLY
        test_labels = list(df_test["cmp_code"])
        train_labels = list(df_train["cmp_code"])
        val_labels = list(df_val["cmp_code"])

        print(f"[APPLIED] Filtering train dataset to fit testing labels")
        wrong_inds = [iv for iv in tqdm(range(len(train_labels))) if train_labels[iv] not in np.unique(test_labels)]
        df_train = df_train.drop(df_train.index[wrong_inds])
        print(f"[APPLIED] Filtering validation dataset to fit testing labels")
        wrong_inds = [iv for iv in tqdm(range(len(val_labels))) if val_labels[iv] not in np.unique(test_labels)]
        df_val= df_val.drop(df_val.index[wrong_inds])

        df_train.reset_index(drop=True, inplace=True)
        df_val.reset_index(drop=True, inplace=True)
        df_test.reset_index(drop=True, inplace=True)

        new_train_labels_u = np.unique(list(df_train["cmp_code"]))
        new_val_labels_u = np.unique(list(df_val["cmp_code"]))
        if len(np.unique(train_labels))!=len(new_train_labels_u): #{len(np.unique(list(df_train["cmp_code"]))}
            print(f"[WARNING] Train classes have been reduced to fit test classes. Keeping {len(new_train_labels_u)}/{len(np.unique(train_labels))}")
        if len(np.unique(val_labels))!=len(new_val_labels_u): #{len(np.unique(list(df_train["cmp_code"]))}
            print(f"[WARNING] Validation classes have been reduced to fit test classes. Keeping {len(new_val_labels_u)}/{len(np.unique(val_labels))}")

    
    # Show the new dataset statistics
    total_samples = len(df_train) + len(df_val) + len(df_test)
    print(f"Train: {len(df_train)} ({np.round(len(df_train)/total_samples*100,2)}%) - Validation: {len(df_val)} ({np.round(len(df_val)/total_samples*100,2)}%) - Testing: {len(df_test)} ({np.round(len(df_test)/total_samples*100,2)}%)")
    show_statistics(args, df_train, type_="train", save=True, out_name = os.path.join(args.data_path_final, data_name_out+f"_c{args.context}_train.xlsx"))
    show_statistics(args, df_val, type_="val", save=True,  out_name = os.path.join(args.data_path_final, data_name_out+f"_val.xlsx"))
    show_statistics(args, df_test, type_="test", save=True,  out_name = os.path.join(args.data_path_final, data_name_out+f"_test.xlsx"))


    
