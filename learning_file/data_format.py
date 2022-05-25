import os
import json
import glob
import itertools
import pprint
import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=150)

# __大文字アルファベット_や__数字_

# jsonファイルからMizar本文のデータを取得

def input_data_formatting(df):
    # os.chdir('/home/shotaro0310/research/learning_data')
    dir = './learning_data/'
    word = []

    for file_name in df['file_name']:
        word_list_a = []
        try:
            json_open = open(f'{dir}{file_name}.json', 'r')
        except FileNotFoundError:
            continue
        json_load = json.load(json_open)
        word_list_a.append(json_load["contents"])
        json_open.close()
        word_list_b = list(itertools.chain.from_iterable(word_list_a))
        word_list_c = list(itertools.chain.from_iterable(word_list_b))
        word_list_d = list(itertools.chain.from_iterable(word_list_c))
        word1 = []
        word2 = []
        for i in range(len(word_list_d)):
            if len(word_list_d) > 0:
                if word_list_d[0] == word_list_d[1]:
                    del word_list_d[0:2]
                else:
                    word1.append(word_list_d[0])
                    word_list_d.remove(word_list_d[0])
                    word1.append(word_list_d[0])
                    word_list_d.remove(word_list_d[0])
            else:
                break

        for i in range(len(word1)):
            if len(word1) > 0:
                if word1[1] == '__number_' or word1[1] == '__variable_' or word1[1] == '__label_':
                    del word1[0:2]
                else:
                    word2.append(word1[0])
                    del word1[0:2]
            else:
                break

        word.append(" ".join(word2))
    
    return word

if __name__ == '__main__':
    df = pd.read_csv('分類表.csv')
    word = input_data_formatting(df)
    print(len(word))