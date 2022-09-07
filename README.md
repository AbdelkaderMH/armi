
## Deep Multi-Task Models for Misogyny Identification and Categorization on Arabic Social Media


### Code for ArMI 2021 paper [Deep Multi-Task Models for Misogyny Identification and Categorization on Arabic Social Media](http://ceur-ws.org/Vol-3159/T5-5.pdf)

## Requirements
- PyTorch
- emojis
- scikit-learn
- barbar

## Results

Our official submission are located in the `results/` folder

## Datasets
The `data/` folder contains the ArMI dataset:
#### ArMI at FIRE2021: Overview of the First Shared Task on Arabic Misogyny Identification

Please fill this form: [ArMI data](https://forms.gle/TgpYhdJBETW2caGj7), to have access to the full corpus. 

To reproduce the results achieved in paper please use the following datasets:

## Training and Evaluation

### Single task models

#### Mysogyny detection

```
python train_misogyny.py --lm marbert --cls 1 --lr 1e-5 --epochs 5 --batch_size 16 

python eval_misogyny.py --lm marbert --cls 1 --lr 1e-5 --epochs 5 --batch_size 16 

```
arguments:
- lm: the pretrained language model marber|camel|qarib|arbert|larabert
- cls: 1 for ST_CLS, 2 for ST_CLS and 3 for ST_VHATT

#### Mysogyny categorization 

```
python train_cat.py --lm marbert --cls 1 --lr 1e-5 --epochs 5 --batch_size 16 

python eval_category.py --lm marbert --cls 1 --lr 1e-5 --epochs 5 --batch_size 16 

```
arguments:
- lm: the pretrained language model marber|camel|qarib|arbert|larabert
- cls: 1 for ST_CLS, 2 for ST_CLS and 3 for ST_VHATT

### Multi-task learning models

```
python train_mtl.py --lm marbert --cls 1 --lr 1e-5 --epochs 5 --batch_size 16 

python eval_mtl.py --lm marbert --cls 1 --lr 1e-5 --epochs 5 --batch_size 16 

```
arguments:
- lm: the pretrained language model marber|camel|qarib|arbert|larabert
- cls: 1 for MT_CLS, 2 for MT_CLS and 3 for MT_VHATT

## Citation 
If you use this code, please cite this paper
```
inproceedings{El-Mahdaouy-Deep,
  author    = {Abdelkader El Mahdaouy and
               Abdellah El Mekki and
               Ahmed Oumar and
               Hajar Mousannif and
               Ismail Berrada},
  editor    = {Parth Mehta and
               Thomas Mandl and
               Prasenjit Majumder and
               Mandar Mitra},
  title     = {Deep Multi-Task Models for Misogyny Identification and Categorization
               on Arabic Social Media},
  booktitle = {Working Notes of {FIRE} 2021 - Forum for Information Retrieval Evaluation,
               Gandhinagar, India, December 13-17, 2021},
  series    = {{CEUR} Workshop Proceedings},
  volume    = {3159},
  pages     = {852--860},
  publisher = {CEUR-WS.org},
  year      = {2021},
  url       = {http://ceur-ws.org/Vol-3159/T5-5.pdf},
  abstract = {The prevalence of toxic content on social media platforms, such as hate speech, offensive language, and misogyny, presents serious challenges to our interconnected society. These challenging issues have attracted widespread attention in Natural Language Processing (NLP) community. In this paper, we present the submitted systems to the first Arabic Misogyny Identification shared task. We investigate three multi-task learning models as well as their single-task counterparts. In order to encode the input text, our models rely on the pre-trained MARBERT language model. The overall obtained results show that all our submitted models have achieved the best performances (top three ranked submissions) in both misogyny identification and categorization tasks.}
}
```

