# CMP_Encoder
Classify sentences from Party Manifestos into CMP codes using a BERT-based Sentence Classifier. In particular, EuroBert.


## Project status
This project is in a development stage. Software is provided as is without warranty as to its features, functionality, performance or integrity.
<br/><br/>


## Description
This project aims to build a neural network that is able to correctly assign a CMP code from [Manifesto Project](https://manifesto-project.wzb.eu/) to a given sentence from a Party Manifesto. We want to automize the process of political sentence classification so we can later study how the discourse and ideology of a party changes over years and contexts.

We finetune [EuroBERT](https://huggingface.co/EuroBERT/EuroBERT-210m) with sentences in several languages from the Manifesto Corpus. We have made a selection to even out the number of instances per class.

We are working on using extra sentences from other sources to increase the accuracy of the classes that have less instances in the dataset.
<br/><br/>


## Data
The data used corresponds to the annotated manifestos of several political parties from 2000 to 2019 for the finetuning task.

The labels for the chosen data subset distribute as follows:

<img src="results/finetuning/previous_testing/histogram_spain_raw.png"  width="1000" height="350">

As it can be seen, subcategories have been truncated to supracategories, that is 103.2 is transformed into 103

<br/><br/>


## Usage
The train notebook is prepared to be launched in a Google Colab environment, just run every cell in order, modifying the necessary paths.


The train python file is prepared to be launched in a Supercomputer infrastructure via the launcher.sh or in a common computer using the command contained in the 'setup' folder.
In that file you can find the calls used.
<br/><br/>


## Folder structure and files

```
├── datasets
│   ├── cmp
│   │   ├── national
│   │   └── regional
│   ├── congress
│   │   ├── congress.xlsx
│   │   ├── old
│   │   ├── original_data
│   │   └── process_congress_sessions.py
│   └── congress_parsing.zip
├── models
├── README.md
├── results
│   ├── finetuning
│   │   └── previous_testing
│   └── intermediate_pretraining
└── src
    ├── finetuning
    │   ├── launcher_debug.sh
    │   ├── launcher.sh
    │   ├── lib
    │   ├── predict.py
    │   ├── results
    │   ├── setup
    │   └── train_encoder.py
    └── intermediate_pretrain
        ├── intermediate_pretrain.py
        └── launcher.sh

```


### **src**
Training files can be found under the src folder for each of the tasks together with a launch script for MareNostrum5 Supercomputer cluster. For example ***train_encoder.py*** and ***launcher.sh*** under the finetuning folder.

### **results**
This folders contains execution results, tables and images corresponding to the training files in **src**.

### **models**
It contains the models to execute the finetuning on as well as final models that have already been finetuned or pretrained.

### **datasets**
This folder contains all the datasets used in the training files. Each dataset contains subfolder with preprocessing files or annotations.

<br/><br/>



## Authors and acknowledgment
Sergi Albiach (BSC), Eduardo Quiñones (BSC), Marc Guinjoan (UOC), Jordi Mas (UOC) and Xavier Roura (UB).
<br/><br/>

