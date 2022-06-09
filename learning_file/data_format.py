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

def input_data_formatting(df, dir):
    # os.chdir('/home/shotaro0310/research/learning_data')
    # dir = './learning_data/'
    word = []
    words_list = []

    for file_name in df['file_name']:
        # word_list_a = []
        # try:
        #     json_open = open(f'{dir}{file_name}.json', 'r')
        # except FileNotFoundError:
        #     continue
        # json_load = json.load(json_open)
        # word_list_a.append(json_load["contents"])
        # json_open.close()
        # word_list_b = list(itertools.chain.from_iterable(word_list_a))
        # word_list_c = list(itertools.chain.from_iterable(word_list_b))
        # word_list_d = list(itertools.chain.from_iterable(word_list_c))
        json_path = f'{dir}{file_name}.json'
        if not os.path.exists(json_path):
            continue
        
        word_list = []
        with open(f'{dir}{file_name}.json', 'r') as json_file:
            json_load = json.load(json_file)
            word_list.append(json_load["contents"])
        
        def flatten_list(l):
            for el in l:
                if isinstance(el, list):
                    yield from flatten_list(el)
                else:
                    yield el
        
        word_list = list(flatten_list(word_list))
        

        # word1 = []
        # word2 = []
        # for i in range(len(word_list_d)):
        #     if len(word_list_d) > 0:
        #         if word_list_d[0] == word_list_d[1]:
        #             del word_list_d[0:2]
        #         else:
        #             word1.append(word_list_d[0])
        #             word_list_d.remove(word_list_d[0])
        #             word1.append(word_list_d[0])
        #             word_list_d.remove(word_list_d[0])
        #     else:
        #         break

        # for i in range(len(word1)):
        #     if len(word1) > 0:
        #         if word1[1] == '__number_' or word1[1] == '__variable_' or word1[1] == '__label_':
        #             del word1[0:2]
        #         else:
        #             word2.append(word1[0])
        #             del word1[0:2]
        #     else:
        #         break

        # word.append(" ".join(word2))
        ww_list = []
        i = 0
        while i < len(word_list):
            ww_list.append( (word_list[i], word_list[i+1]) )
            i = i + 2

        
        words = []
        for ww in ww_list:
            if ww[0] != ww[1] and ww[1] not in ['__number_', '__variable_', '__label_']:
                words.append(ww[0])
        
        words_list.append(" ".join(words))
        
    
    # return word
    return words_list


if __name__ == '__main__':
    df = pd.read_csv('分類表.csv')
    word = input_data_formatting(df)
    print(len(word))