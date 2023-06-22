"""
Script to translate the sentences from one dataset into the desired language.

@author: Sergi Albiach
@institution: BSC
@date: 22/11/2022
@version: 1.0
"""

from transformers import MarianMTModel, MarianTokenizer
from typing import Sequence
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import gc
import argparse
import sys


class Translator:
    def __init__(self, source_lang: str, dest_lang: str) -> None:
        """
        Initialization of the translator from language A to language B
        @param source_lang: source language of the translator (A)
        @param dest_lang: destination language of the translator (B)
        return: nothing
        """
        __slots__ = ('model_name', 'model', 'tokenizer')
        self.model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{dest_lang}'
        self.model = MarianMTModel.from_pretrained(self.model_name)
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        
    def translate(self, texts: Sequence[str], max_size:int=512) -> Sequence[str]:
        """
        Function to translate a set of texts from language A to language B previsouly defined
        @param texts: one or more texts to translate
        @param max_size: maximum size of the sentences for the model (in characters)
        return: the translated texts
        """
        tokens = self.tokenizer(list(texts), return_tensors="pt", padding=True)
        translate_tokens = self.model.generate(**tokens, max_new_tokens=max_size)
        res = [self.tokenizer.decode(t, skip_special_tokens=True) for t in translate_tokens]
        gc.collect()
        return res

def translate(df: pd.DataFrame, target_lang:str="spanish", remove_unavail:bool=True, bs:int=50, max_size:int=512) -> pd.DataFrame:
    """
    Function to translate the texts from a DataFrame to target language
    @param df: DataFrame containing the texts to translate
    @param target_lang: language in which the sentences will be translated
    @param remove_unavail: flag to remove or not the sentences from the final dataframe that could not be translated
    @param bs: batch size, amount of samples that will be fed to the model at the same time
    @param max_size: maximum size of the sentences for the model (in characters)
    return: the translated DataFrame
    """
    avail_codes = ["ca", "en"] # Available target languages for the translation (there are more)
    langs_dict = {'af': 'afrikaans', 'sq': 'albanian', 'am': 'amharic', 'ar': 'arabic', 'hy': 'armenian', 'az': 'azerbaijani', 'eu': 'basque', 'be': 'belarusian', 'bn': 'bengali', 'bs': 'bosnian', 'bg': 'bulgarian', 'ca': 'catalan', 'ceb': 'cebuano', 'ny': 'chichewa', 'zh-cn': 'chinese (simplified)', 'zh-tw': 'chinese (traditional)', 'co': 'corsican', 'hr': 'croatian', 'cs': 'czech', 'da': 'danish', 'nl': 'dutch', 'en': 'english', 'eo': 'esperanto', 'et': 'estonian', 'tl': 'filipino', 'fi': 'finnish', 'fr': 'french', 'fy': 'frisian', 'gl': 'galician', 'ka': 'georgian', 'de': 'german', 'el': 'greek', 'gu': 'gujarati', 'ht': 'haitian creole', 'ha': 'hausa', 'haw': 'hawaiian', 'iw': 'hebrew', 'hi': 'hindi', 'hmn': 'hmong', 'hu': 'hungarian', 'is': 'icelandic', 'ig': 'igbo', 'id': 'indonesian', 'ga': 'irish', 'it': 'italian', 'ja': 'japanese', 'jw': 'javanese', 'kn': 'kannada', 'kk': 'kazakh', 'km': 'khmer', 'ko': 'korean', 'ku': 'kurdish (kurmanji)', 'ky': 'kyrgyz', 'lo': 'lao', 'la': 'latin', 'lv': 'latvian', 'lt': 'lithuanian', 'lb': 'luxembourgish', 'mk': 'macedonian', 'mg': 'malagasy', 'ms': 'malay', 'ml': 'malayalam', 'mt': 'maltese', 'mi': 'maori', 'mr': 'marathi', 'mn': 'mongolian', 'my': 'myanmar (burmese)', 'ne': 'nepali', 'no': 'norwegian', 'ps': 'pashto', 'fa': 'persian', 'pl': 'polish', 'pt': 'portuguese', 'pa': 'punjabi', 'ro': 'romanian', 'ru': 'russian', 'sm': 'samoan', 'gd': 'scots gaelic', 'sr': 'serbian', 'st': 'sesotho', 'sn': 'shona', 'sd': 'sindhi', 'si': 'sinhala', 'sk': 'slovak', 'sl': 'slovenian', 'so': 'somali', 'es': 'spanish', 'su': 'sundanese', 'sw': 'swahili', 'sv': 'swedish', 'tg': 'tajik', 'ta': 'tamil', 'te': 'telugu', 'th': 'thai', 'tr': 'turkish', 'uk': 'ukrainian', 'ur': 'urdu', 'uz': 'uzbek', 'vi': 'vietnamese', 'cy': 'welsh', 'xh': 'xhosa', 'yi': 'yiddish', 'yo': 'yoruba', 'zu': 'zulu', 'fil': 'Filipino', 'he': 'Hebrew'}
    langs_dict_inv = {v: k for k, v in langs_dict.items()} # full language to code dictionary

    target_code = langs_dict_inv.get(target_lang, '') # code of the target language
    source_langs = np.unique(df["language"]) # languages to be translated to target language
    
    final_df = pd.DataFrame({k : [] for k in df.columns}) # init empty DataFrame

    # For every language to translte, get its code and a df with the sentences from that language
    for src_lang in source_langs:
        src_code = langs_dict_inv.get(src_lang, '')
        target_df = df[df["language"]==src_lang]
        target_df = target_df.reset_index()

        if src_code != '' and src_code in avail_codes: # if the language is available to translate

            # Filter out sentences bigger than max_size (512) characters
            inds = [i for i in range(len(target_df)) if len(target_df["text"].iloc[i]) > max_size] 
            if len(inds) > 0:
                print(f"Removing very long sentences (> {max_size} characters) for {src_lang} ({len(inds)})")
            target_df = target_df.drop(inds)
            sentences = target_df["text"]

            translator = Translator(src_code, target_code)
            
            print(f"Translating [{len(target_df)}] sentences from [{src_lang}] to [{target_lang}] with a batch size of [{bs}]")
            
            start = time.time()
            trans_sents = np.empty((len(sentences)), dtype=object)
            start_end_set = [((i * bs),((i + 1) * bs)) for i in range((len(sentences) + bs - 1) // bs )]
            for start_end_subset in tqdm(start_end_set):
                s = sentences.iloc[start_end_subset[0]:start_end_subset[1]]
                t = translator.translate(s)
                trans_sents[start_end_subset[0]:start_end_subset[1]] = t
                gc.collect()
            end = time.time()

            print(f"Translation from {src_lang} took {np.round(end-start, 2)} seconds")

            target_df["language"] = [target_lang for _ in range(len(target_df))] # update 'language' column
            target_df["text"] = trans_sents # update 'text' column with translation
            final_df = pd.concat([final_df, target_df], axis=0) # add to final DF

        else:
            if not remove_unavail:
                print(f"Could not translate {src_lang} ({len(target_df)}). ADDED TO FINAL DATA FRAME!")
                final_df = pd.concat([final_df, target_df], axis=0)
            else:
                print(f"Could not translate {src_lang} ({len(target_df)}). NOT ADDED TO FINAL DATA FRAME!")
                
    return final_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", "-bs", type=int, default=100, help="Batch size to load the sentences into the translator")
    parser.add_argument("--file-to-translate", "-f", type=str, default="/home/salbiach/Documents/projectes/politics/cmp_encoder/dataset/preprocessing/CMPD_raw/CMPD_Spain_foreign.xlsx", help="Dataset to translate")
    args = parser.parse_args(sys.argv[1:])

    df = pd.read_excel(args.file_to_translate)
    final_df = translate(df, bs=args.batch_size)
    final_df.to_excel(f"{args.file_to_translate.split('.')[0]}_translated.xlsx", index=False)