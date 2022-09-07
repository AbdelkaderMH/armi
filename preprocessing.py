import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import Dataset
import os
import text_normalization
import numpy as np


def augment_data(df_train):
    df_others = df_train[(df_train['category'] != 'none') & (df_train['category'] != 'discredit')]
    df_aug = pd.DataFrame(columns=['tweet_id', 'text', 'misogyny', 'category'])
    dic_dup = {'damning': 4,
               'derailing': 26,
               'dominance': 17,
               'sexual harassment': 50,
               'stereotyping & objectification': 4,
               'threat of violence': 17}
    for i in range(df_others.shape[0]):
        current = df_others.iloc[i]
        tweet_id = current['tweet_id']
        text = current['text']
        label_mis = current['misogyny']
        label_cat = current['category']

        aug_ratio = dic_dup[label_cat]
        for k in range(aug_ratio):
            tokens = text.split(' ')
            l = len(tokens)
            n = int(0.15 * l)
            indices = np.random.choice(l, n, replace=False)
            for j in range(len(indices)):
                tokens[indices[j]] = '[MASK]'
            new_text = ' '.join(tokens)
            entry = {'tweet_id': tweet_id, 'text': new_text, 'misogyny': label_mis, 'category': label_cat}
            df_aug = df_aug.append(entry, ignore_index=True)
    df_aug.drop_duplicates(subset=['text'], keep='first', inplace=True)
    df = pd.concat([df_train,df_aug])
    print(df.category.value_counts())
    return df



def loadTrainValData(path='data/ArMI2021_training.tsv', size=0.2, batchsize=16, num_worker=25, pretraine_path="xlm-roberta-base", seed=0):

    data = pd.read_csv(path, encoding='utf-8', sep='\t')
    print(data.shape)
    df_train, df_test = train_test_split(data, test_size=size, stratify=data[['misogyny', 'category']], random_state=seed, shuffle=True)#
    #df_train = augment_data(df_train)
    df_train.dropna(axis=0, inplace=True)
    df_test.dropna(axis=0, inplace=True)

    df_train['text'] = df_train['text'].apply(lambda x: text_normalization.preprocess(x))
    df_test['text'] = df_test['text'].apply(lambda x: text_normalization.preprocess(x))


    DF_train = Dataset.TrainDataset(df_train, pretraine_path)
    DF_test = Dataset.TrainDataset(df_test, pretraine_path)

    DF_train_loader = DataLoader(dataset=DF_train, batch_size=batchsize, shuffle=True,
                                 num_workers=num_worker)
    DF_test_loader = DataLoader(dataset=DF_test, batch_size=batchsize, shuffle=False,
                                num_workers=num_worker)
    return DF_train_loader, DF_test_loader

def loadTestData(path='data/ArMI2021_test.tsv', batchsize=16, num_worker=2, pretraine_path="xlm-roberta-base"):
    data = pd.read_csv(path, encoding='utf-8', sep='\t')
    data['text'] = data['text'].apply(lambda x:text_normalization.preprocess(x))
    ids = data['tweet_id']

    DF_test = Dataset.TestDataset(data, pretraine_path)

    DF_test_loader = DataLoader(dataset=DF_test, batch_size=batchsize, shuffle=False,
                                num_workers=num_worker)
    return DF_test_loader, ids



