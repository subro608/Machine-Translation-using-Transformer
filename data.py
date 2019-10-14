import spacy
from spacy.lang.cs import Czech
from torchtext.data import Field
import pandas as pd
from sklearn.model_selection import train_test_split
from torchtext.data import TabularDataset, BucketIterator
import torch

en = spacy.load("en")
cs = Czech()
cs = spacy.blank("cs")


def tokenize_en(sentence):
    return [tok.text for tok in en.tokenizer(sentence)]


def tokenize_cs(sentence):
    return [tok.text for tok in cs.tokenizer(sentence)]


def getdata():
    europarl_en = open('europarl-v7.cs-en.en', encoding='utf-8').read().split('\n')
    europarl_cs = open('europarl-v7.cs-en.cs', encoding='utf-8').read().split('\n')

    EN_TEXT = Field(tokenize=tokenize_en)
    CS_TEXT = Field(tokenize=tokenize_cs, init_token="<sos>", eos_token="<eos>")

    raw_data = {'English': [line for line in europarl_en], 'Czech': [line for line in europarl_cs]}
    df = pd.DataFrame(raw_data, columns=["English", "Czech"])
    # remove very long sentences and sentences where translations are
    df['en_len'] = df['English'].str.count(' ')
    df['cs_len'] = df['Czech'].str.count(' ')
    df = df.query('cs_len < 80 & en_len < 80')
    df = df.query('cs_len < en_len * 1.5 & cs_len * 1.5 > en_len')
    train, val = train_test_split(df, test_size=0.1)
    train.to_csv("train.csv", index=False)
    val.to_csv("val.csv", index=False)
    data_fields = [('English', EN_TEXT), ('Czech', CS_TEXT)]
    train, val = TabularDataset.splits(path='./', train='train.csv', validation='val.csv', format='csv',
                                       fields=data_fields)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EN_TEXT.build_vocab(train, min_freq=2)
    CS_TEXT.build_vocab(train, min_freq=2)
    BATCH_SIZE = 16
    INPUT_DIM = len(EN_TEXT.vocab)
    OUTPUT_DIM = len(CS_TEXT.vocab)
    PAD_IDX = EN_TEXT.vocab.stoi['<pad>']

    train_iterator, valid_iterator = BucketIterator.splits(
        (train, val),
        batch_size=BATCH_SIZE,
        device=device)
    return train_iterator, valid_iterator, INPUT_DIM, OUTPUT_DIM, PAD_IDX


