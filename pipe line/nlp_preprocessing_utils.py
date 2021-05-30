import os, sys
from tqdm import tqdm
import gc, time, random, math
import numpy as np
import pandas as pd
from pandarallel import pandarallel

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import TfidfVectorizer, CountVectorizer
from joblib import Parallel, delayed

import torch
import transformers
from transformers import BertTokenizer
tqdm.pandas()

import googletrans
from googletrans import Translator

import pycld2 as cld2
import re
import textstat
from nltk.corpus import stopwords
import nltk, string
from nltk.stem.porter import PorterStemmer
from nltk import WordNetLemmatizer


class GoogleTranslate:
    def __init__(self, translate_cols, dest='en'):
        self.col = translate_cols
        self.dest = dest

    def translate_text_util(self, comment, src, dest, i):
        if type(comment)==str:
            translator = Translator()
            translator.raise_Exception = True
            text = ''
            comments = comment.split('.')
            for comment in comments:
                comment+='.'
                try:
                    text += translator.translate(comment, src=src, dest=dest).text
                    
                except:
                    text += translator.translate(comment, dest=dest).text
                gc.collect()
                time.sleep(0.5)
            return str(text), i
        else:
            return comment, i
    def translate_text(self, df):
        parallel = Parallel(n_jobs=-1, backend="threading", verbose=5)
        comments_list = df[self.col].values
        comments_lang_list = df[f"{self.col}_lang"].values
        print(f'Translate comments using {self.dest} language')
        translated_data = parallel(
            delayed(self.translate_text_util)(comment, src, self.dest, i) for i, (comment, src)  in enumerate(zip(comments_list, comments_lang_list))
            )    
        translated_data = pd.DataFrame(translated_data).sort_values(1)[[0]]
        translated_data.columns=[f'translate_{self.dest}']
        df[f'translate_{self.dest}_{self.col}'] = translated_data[f'translate_{self.dest}'].values
        return df

class BertSequenceVectorizer:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.bert_model = transformers.BertModel.from_pretrained(self.model_name)
        self.bert_model = self.bert_model.to(self.device)
        self.max_len = 128


    def vectorize(self, sentence : str) -> np.array:
        inp = self.tokenizer.encode(sentence)
        len_inp = len(inp)

        if len_inp >= self.max_len:
            inputs = inp[:self.max_len]
            masks = [1] * self.max_len
        else:
            inputs = inp + [0] * (self.max_len - len_inp)
            masks = [1] * len_inp + [0] * (self.max_len - len_inp)

        inputs_tensor = torch.tensor([inputs], dtype=torch.long).to(self.device)
        masks_tensor = torch.tensor([masks], dtype=torch.long).to(self.device)

        bert_out = self.bert_model(inputs_tensor, masks_tensor)
        seq_out, pooled_out = bert_out['last_hidden_state'], bert_out['pooler_output']

        if torch.cuda.is_available():    
            return seq_out[0][0].cpu().detach().numpy() # 0番目は [CLS] token, 768 dim の文章特徴量
        else:
            return seq_out[0][0].detach().numpy()

class BaseTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self
        
class TextstatProcessing(BaseTransformer):
    def __init__(self, col_name):
        self.col_name = col_name
    
    def transform(self, X):
        X[f'{self.col_name}_len'] = X[self.col_name].str.len()
        X[f'{self.col_name}_word_len_mean'] = X[self.col_name].apply(lambda x: [len(s) for s in x.split()]).map(np.mean)
        X[f'{self.col_name}_word_len_std'] = X[self.col_name].apply(lambda x: [len(s) for s in x.split()]).map(np.std)
        X[f'{self.col_name}_word_len_max'] = X[self.col_name].apply(lambda x: [len(s) for s in x.split()]).map(np.max)
        
        X[f'{self.col_name}_char_count'] = X[self.col_name].map(textstat.char_count)
        X[f'{self.col_name}_word_count'] = X[self.col_name].map(textstat.lexicon_count)
        X[f'{self.col_name}_sentence_count'] = X[self.col_name].map(textstat.sentence_count)
        X[f'{self.col_name}_syllable_count'] = X[self.col_name].apply(textstat.syllable_count)
        X[f'{self.col_name}_smog_index'] = X[self.col_name].apply(textstat.smog_index)
        X[f'{self.col_name}_automated_readability_index'] = X[self.col_name].apply(textstat.automated_readability_index)
        X[f'{self.col_name}_coleman_liau_index'] = X[self.col_name].apply(textstat.coleman_liau_index)
        X[f'{self.col_name}_linsear_write_formula'] = X[self.col_name].apply(textstat.linsear_write_formula)
        return X
    
class LanguageDetectFeatureExtractor(BaseTransformer):
    def __init__(self, cols):
        self.cols = cols

    def language_detect(self, s):
        _, _, detail = cld2.detect(s)
        lang = detail[0][1]
        return lang

    def transform(self, X):
        for col in self.cols:
            X[f"{col}_languagefeature"] = X[col].fillna("").map(lambda x: self.language_detect(x))
        return X

class TfidfVectorFeatureExtractor(BaseTransformer):
    def __init__(self, col):
        stop_words = list(stopwords.words("english"))
        self.stop_words = stop_words+[' ', '  ','']
        self.porter = PorterStemmer()
        self.lemma = WordNetLemmatizer()  # NOTE: 複数形の単語を単数形に変換する
        self.vec_tfidf = TfidfVectorizer()
        self.col = col
        
    def change_text(self, text):
        #usage
        #train['text'] = train['text'].parallel_apply(self.change_text)
        text = re.sub("[^a-zA-Z]", " ", text).lower()
        text = nltk.word_tokenize(text)  # NOTE: 英文を単語分割する
        text = [word for word in text if not word in self.stop_words]
        text =  " ".join([self.porter.stem(word) for word in text])
        #text =  " ".join([lemma.lemmatize(word) for word in text])
        return text
    
    def fit(self, df):
        self.vec_tfidf.fit(df[self.col].values)
        
    def transform(self, df):
        X = self.vec_tfidf.transform(df[self.col].values)
        X = pd.DataFrame(X.toarray(), columns=self.vec_tfidf.get_feature_names())
        return X


class CountVectorFeatureExtractor(BaseTransformer):
    def __init__(self, col):
        stop_words = list(stopwords.words("english"))
        self.stop_words = stop_words+[' ', '  ','']
        self.porter = PorterStemmer()
        self.lemma = WordNetLemmatizer()  # NOTE: 複数形の単語を単数形に変換する
        self.vec_count = CountVectorizer()
        self.col = col
        
    def change_text(self, text):
        #usage
        #train['text'] = train['text'].parallel_apply(self.change_text)
        text = re.sub("[^a-zA-Z]", " ", text).lower()
        text = nltk.word_tokenize(text)  # NOTE: 英文を単語分割する
        text = [word for word in text if not word in self.stop_words]
        text =  " ".join([self.porter.stem(word) for word in text])
        #text =  " ".join([lemma.lemmatize(word) for word in text])
        return text

    def fit(self, df):
        self.vec_count.fit(df[self.col].values)
        
    def transform(self, df):
        X = self.vec_count.transform(df[self.col].values)
        X = pd.DataFrame(X.toarray(), columns=self.vec_count.get_feature_names())
        return X
