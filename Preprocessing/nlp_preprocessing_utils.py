import os, sys
from tqdm import tqdm
import gc, time, random, math
import numpy as np
import pandas as pd
from pandarallel import pandarallel

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from joblib import Parallel, delayed

import torch
import transformers
from transformers import BertTokenizer
tqdm.pandas()

import googletrans
from googletrans import Translator

import re
import textstat
import string, re

import nltk
from nltk.corpus import wordnet
from nltk.stem.porter import PorterStemmer
from nltk import WordNetLemmatizer
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords as nltk_stopwords

import emoji
from janome.tokenizer import Tokenizer

#単語の定義を取ってくる
def get_definition(s):
    definitions = []
    for syn in wordnet.synsets(s):
        definitions.append(syn.definition())
    return definitions

#翻訳APIを叩く
class GoogleTranslate:
    """
    g_translate = GoogleTranslate('text', dest='en')
    df = g_translate.translate_text(df)
    """
    def __init__(self, col, dest='en', sleep_time=0.1):
        self.col = col
        #変換後の言語
        self.dest = dest
        self.sleep_time = sleep_time
        
    def show_language(self):
        print(googletrans.LANGUAGES)

    def translate_text_util(self, comment, src, dest, i):
        if type(comment)==str:
            translator = Translator()
            translator.raise_Exception = True
            text = ''
            comments = comment.split('.')
            comments = [c for c in comments if c!="" ]
            for comment in comments:
                comment+='.'
                try:
                    text += translator.translate(comment, src=src, dest=dest).text
                    time.sleep(self.sleep_time)
                except:
                    try:
                        text += translator.translate(comment, dest=dest).text
                        time.sleep(self.sleep_time)
                    except:
                        pass
            gc.collect()
            return str(text), i
        else:
            return comment, i

    def __call__(self, df):
        parallel = Parallel(n_jobs=-1, backend="threading", verbose=5)
        #変換元のtext
        comments_list = df[self.col].values
        #変換元textの言語
        comments_lang_list = df[f"{self.col}_lang"].values
        print(f'Translate comments using {self.dest} language')
        translated_data = parallel(
            delayed(self.translate_text_util)(comment, src, self.dest, i) for i, (comment, src)  in enumerate(zip(comments_list, comments_lang_list))
            )    
        translated_data = pd.DataFrame(translated_data).sort_values(1)[[0]]
        translated_data.columns=[f'translate_{self.dest}']
        df[f'translate_{self.dest}_{self.col}'] = translated_data[f'translate_{self.dest}'].values
        return df

# Bertの埋め込み表現
class BertSequenceVectorizer:
    def __init__(self, model_name='bert-base-uncased'):
        """
        cl-tohoku/bert-base-japanese-whole-word-masking
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.tokenizer = transformers.BertTokenizer.from_pretrained(self.model_name)
        self.bert_model = transformers.BertModel.from_pretrained(self.model_name)
        self.bert_model = self.bert_model.to(self.device)
        self.max_len = 128
        self.output_vec_size = 768

    def vectorize(self, sentence : str) -> np.array:
        inputs = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens = True, # [CLS],[SEP]を入れるか
            max_length = self.max_len, # paddingとtrancation(切り出し)を使って、単語数をそろえる
            padding = True, # ブランク箇所に[PAD]を入れる
            truncation = True, # 切り出し機能。例えばmax_length10とかにすると、最初の10文字だけにしてくれる機能。入れないと怒られたので、入れておく
            return_tensors = 'pt',

        )

        for key, value in inputs.items():
            inputs[key] = value.to(self.device)
            
        bert_out = self.bert_model(**inputs)
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

#文字の長さ特徴量     
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

# 言語判別
class LanguageDetectFeatureExtractor(BaseTransformer):
    def __init__(self, cols):
        import pycld2 as cld2
        self.cols = cols

    def language_detect(self, s):
        _, _, detail = cld2.detect(s)
        lang = detail[0][1]
        return lang

    def transform(self, X):
        for col in self.cols:
            X[f"{col}_languagefeature"] = X[col].fillna("").map(lambda x: self.language_detect(x))
        return X

#日本語の分かち書き
def get_wakati_ja(s):
    """
    ==usagge==
    df["ja_new_text"] = df["ja_text"].apply(get_wakati_ja)
    """
    tknz = Tokenizer()

    UNICODE_EMOJI = []
    for lang, value in emoji.UNICODE_EMOJI.items():
        UNICODE_EMOJI += list(value.keys())
    UNICODE_EMOJI = list(set(UNICODE_EMOJI))

    l = []
    speech = ["名詞", "動詞", "形容詞" ]
    code_regex = re.compile('[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％]')
    s = code_regex.sub(' ', s)
    for token in tknz.tokenize(s):
        if token.part_of_speech.split(",")[0] in speech:
            t = token.base_form
            t = t.replace(' ', '')
            if len(t)==0:
                continue
            if len(t)==1:
                if len(re.findall('[\u3041-\u309F]+', t))>0:
                    continue
            if t in UNICODE_EMOJI:
                continue
            #t = token.surface
            l.append(t)
    l = [s.replace(' ', '') for s in l]
    l = [s for s in l if len(s)>0]
    s = " ".join(l)
    return s

def change_text_en(text):
    """
    ==usagge==
    df["english_new_text"] = df["english_text"].apply(change_text_en)
    """
    stop_words = list(nltk_stopwords.words("english"))+[' '*i for i in range(10)]
    porter = PorterStemmer()
    lemma = WordNetLemmatizer()  # NOTE: 複数形の単語を単数形に変換する
    #usage
    #train['text'] = train['text'].parallel_apply(self.change_text)
    text = re.sub("[^a-zA-Z]", " ", text).lower()
    text = nltk.word_tokenize(text)  # NOTE: 英文を単語分割する
    text = [word for word in text if not word in stop_words]
    text = [porter.stem(word) for word in text]
    text = [s.replace(' ', '') for s in text]
    text = [s for s in text if len(s)>0]
    text =  " ".join(text)
    #text =  " ".join([lemma.lemmatize(word) for word in text])
    return text

class change_text:
    """
    ==usagge==
    change_fr_text = change_text('fr')
    df["new_text"] = df["text"].apply(change_fr_text)
    """
    
    @staticmethod
    def get_language():
        print(nltk_stopwords.fileids())
        print(list(SnowballStemmer.languages))
        
    def __init__(self, language="english", stem=False):
        assert language in nltk_stopwords.fileids() and language in list(SnowballStemmer.languages), "language is not in stopwords fileids or SnowballStemmer languages"
        
        UNICODE_EMOJI = []
        for lang, value in emoji.UNICODE_EMOJI.items():
            UNICODE_EMOJI += list(value.keys())
        UNICODE_EMOJI = list(set(UNICODE_EMOJI))

        self. code_regex = re.compile('[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％]')
        self.stemmer = SnowballStemmer(language)
        self.stopwords = [" "*i for i in range(10)] + nltk_stopwords.words(language)+ UNICODE_EMOJI
        self.stem = stem
    
    def set_stem(self, stem):
        self.stem = stem
        
    def add_stopwords(self, add_words):
        if type(add_words)==str:
            add_words = [add_words]
        self.stopwords += add_words
        
    def __call__(self, text):
        text = text.lower()
        text = self.code_regex.sub(' ', text)
        text = word_tokenize(text)
        if self.stem:
            text = [self.stemmer.stem(token) for token in text]
        text = [word for word in text if not word in self.stopwords]
        text = [s.replace(' ', '') for s in text]
        text = [s for s in text if len(s)>0]
        text =  " ".join(text)
        return text
    
class TfidfVectorFeatureExtractor(BaseTransformer):
    def __init__(self, col):
        self.vec_tfidf = TfidfVectorizer()
        self.col = col
    
    def fit(self, df):
        self.vec_tfidf.fit(df[self.col].values)
        
    def transform(self, df):
        X = self.vec_tfidf.transform(df[self.col].values)
        X = pd.DataFrame(X.toarray(), columns=self.vec_tfidf.get_feature_names())
        return X


class CountVectorFeatureExtractor(BaseTransformer):
    def __init__(self, col):
        self.vec_count = CountVectorizer()
        self.col = col
        
    def fit(self, df):
        self.vec_count.fit(df[self.col].values)
        
    def transform(self, df):
        X = self.vec_count.transform(df[self.col].values)
        X = pd.DataFrame(X.toarray(), columns=self.vec_count.get_feature_names_out())
        return X

    def fit_transform(self, df, y=None, **fit_params):
        self.vec_count.fit(df[self.col].values)
        X = self.vec_count.transform(df[self.col].values)
        X = pd.DataFrame(X.toarray(), columns=self.vec_count.get_feature_names_out())
        return X