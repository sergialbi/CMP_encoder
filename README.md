# CMP_Encoder
Creation of an encoder to transform a sentence into a CMP code



## Project status
This project is in a development stage. Software is provided as is without warranty as to its features, functionality, performance or integrity.
<br/><br/>


## Description
This project aims to build a neural network that is able to assign to a given sentence, a CMP code from [Manifesto Project](https://manifesto-project.wzb.eu/). We want to automize the process of sentence classification so we can later study how the discourse of a party changes over years and contexts.

In order to do that, we use a pretrained Bert transformer and finetune it with sentences of the Manifesto Corpus.

For the moment being, only spanish text is considered and accordingly, tokenization and pretrained weigths are loaded from ['dccuchile/bert-base-spanish-wwm-cased'](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased).

Those sentences in catalan are translated to spanish also using a transformer.

Sentences from other countries are left out for the sake of simplicity. Once the model is correctly trained and tested we will fine tune it for other countries and languages.
<br/><br/>


## Data
The data used corresponds to the annotated manifestos of the political parties in Spain from 2000 to 2019.

The labels for the chosen data subset distribute as follows:

<img src="results/histogram_spain_raw.png"  width="1000" height="350">

As it can be seen, subcategories have been truncated to supracategories, that is 103.2 is transformed into 103

Other datasets are being created and tested. For example by limiting the maximum amount of samples per category and taking out those categories with less than enough samples.
<br/><br/>


## Usage
The train notebook is prepared to be launched in a Google Colab environment, just run every cell in order, modifying the necessary paths.


The train python file is prepared to be launched in a Supercomputer infrastructure via the launcher.sh or in a common computer using the command contained in the 'setup' folder.
In that file you can find the calls used.
<br/><br/>


## Folder structure and files

```
├── dataset
│   ├── CMPD_model.xlsx
│   ├── meta
│   ├── official_docs
│   └── preprocessing
│       ├── get_data.Rmd
│       ├── preprocess.py
│       ├── summarize.py
│       └── translate
│           └── translator.py
│   ├── v0
│   ├── v1
├── launcher.sh
├── lib
├── predict.py
├── proves
├── results
├── setup
├── train_encoder.py
└── train_encoder_notebook.ipynb
```


### **Train files**
Training files can be found under cmp_encoder directly as a notebook and as a script ready to launch: ***train_encoder_notebook.ipynb*** - ***train_encoder.py***


### **lib**
Folder containing useful files to the training process like the model to build and some other functions.

### **proves**
Containig files in testing process, not finished, containing bugs, etc.

### **dataset**
#### **preprocessing**
Contains the R script to download the data (***get_data.Rmd***),
MISSING: the API. Register in the Manifesto Project page to obtain it.

#### **official_docs**
Contains the documentation of the used Codebook as well as the codification of the political parties in pdf and in excel format.

#### **meta**
Meta information of the selected manifestos to use to train

***CMPD_model.xlsx***: set of sentences with associated CMP code ready to train. Obtained after the call to 'preprocess.py' in order to balance the dataset, add sentences from other languages, remove short sentences, etc.

***CMPD_raw.xlsx***: set of sentences with associated CMP code with a minimal preprocessing (strange token removal, language correction, etc.) but still raw directly extracted from the Manifesto Corpus.
<br/><br/>



## Authors and acknowledgment
Sergi Albiach (BSC), Eduardo Quiñones (BSC), Marc Guinjoan (UOC), Jordi Mas (UOC) and Xavier Roura (UB).
<br/><br/>


## Gitlab Usage
```
cd cmp_encoder
git status #Check which files are updated
git add . #Add all files to commit
git commit -m "the_comment" #Create a commit with a comment
git push # Push the files to Gitlab
```
<br/><br/>
