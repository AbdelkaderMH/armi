import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer



class TrainDataset(Dataset):
    def __init__(self, df, pretraine_path='xlm-roberta-base', max_length=128):
        self.df = df
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(pretraine_path)

    def __getitem__(self, index):
        text = self.df.iloc[index]['text']
        l_misogyny = self.df.iloc[index]["misogyny"]
        l_category = self.df.iloc[index]["category"]
        dict_misogyny = {
            'none' : 0,
             'misogyny' : 1,
        }
        dict_category = {
            'none': 0,
            'damning':1,
            'derailing':2,
            'discredit': 3,
            'dominance': 4,
            'sexual harassment': 5,
            'stereotyping & objectification':6,
            'threat of violence':7
        }


        misogyny = dict_misogyny[l_misogyny]
        category = dict_category[l_category]
        encoded_input = self.tokenizer(
                text,
                max_length = self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt",
            )

        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"] if "attention_mask" in encoded_input else None

        data_input = {
            "input_ids":input_ids.flatten(),
            "attention_mask": attention_mask.flatten()
        }

        label_input ={
            "misogyny": torch.tensor(misogyny, dtype=torch.float),
            "category": torch.tensor(category, dtype=torch.long)


        }

        return data_input, label_input

    def __len__(self):
        return self.df.shape[0]


class TestDataset(Dataset):
    def __init__(self, df, pretraine_path='xlm-roberta-base', max_length=128):
        self.df = df
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(pretraine_path)

    def __getitem__(self, index):
        text = self.df.iloc[index]["text"]

        encoded_input = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"] if "attention_mask" in encoded_input else None

        data_input = {
            "input_ids": input_ids.flatten(),
            "attention_mask": attention_mask.flatten()
        }

        return data_input

    def __len__(self):
        return self.df.shape[0]
