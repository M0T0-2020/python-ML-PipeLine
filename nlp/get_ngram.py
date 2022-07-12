import imp


import pandas as pd
import numpy as np
from collections import Counter

def get_ngram(text_list, n_gram=1):
    """
    ==usage==
    text_list = df["text"] <=分ち書き済み
    ngram_df = get_ngram(text_list, n_gram=2)
    """
    text_split = [text.split(" ") for text in text_list]
    text_split = [[s.replace(' ', '') for s in text] for text in text_split]
    new_text_split = []
    if n_gram>1:
        for i, text in enumerate(text_split):
            length = len(text)
            if length>=n_gram:
                a = []
                for l in range(length-n_gram+1):
                    a.append(" ".join([text[l+k] for k in range(n_gram)]))
                new_text_split.append(a)
    else:
        new_text_split = text_split
    ngram = []
    for text in new_text_split:
        text = [s for s in text if len(s)>0]
        ngram += text
    ngram = dict( Counter(ngram) )
    ngram = dict(sorted(ngram.items(), key=lambda x:x[1])[::-1])
    ngram = [kv for kv in ngram.items()]
    ngram_df = pd.DataFrame(ngram, columns=["word", "word_count"])
    return ngram_df
