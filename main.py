import os
import json
import itertools
import pprint
import pandas as pd 
import numpy as np
import pickle
from tqdm import tqdm
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=100)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# from learning_file.data_format import input_data_formatting
from sklearn.linear_model import LogisticRegression


def main():
    df = pd.read_csv('分類表.csv')
    dir = './learning_data/'

    # input_data_formattingが重いので，words_list.pklにデータを保存するようにした．
    # words_listが変更されたらカレントディレクトリのwords_list.pklを消す
    if os.path.exists('words_list.pkl'):
        with open('words_list.pkl', 'rb') as f:
            words_list = pickle.load(f)
    else:
        words_list = input_data_formatting(df, dir)
        with open('words_list.pkl', 'wb') as f:
            pickle.dump(words_list, f)

    targets = []

    for label in df['label_number']:
        targets.append(int(label))

    # テキスト内の単語の出現頻度を数えて、結果を素性ベクトル化する(Bag of words)
    tf_idf_vectorizer = TfidfVectorizer(lowercase=False, token_pattern=r"\S+")

    # csr_matrix(疎行列)にしてndarray(多次元配列)に変形
    feature_vectors = tf_idf_vectorizer.fit_transform(words_list).toarray()

    # train_test_splitでラベル位置が変わらないようにしている
    feature_vectors = np.insert(feature_vectors, 0, list(range(len(feature_vectors))), axis=1)
    input_train, input_test, output_train, output_test = train_test_split(feature_vectors, targets, test_size=0.2, random_state=0, stratify=targets)
    
    train_indices = input_train[:, 0]
    test_indices = input_test[:, 0]
    input_train = input_train[:, 1:]
    input_test = input_test[:, 1:]
    
    sc = StandardScaler()
    sc.fit(input_train)
    input_train_std = sc.transform(input_train)
    input_test_std = sc.transform(input_test)

    # 学習インスタンス生成
    svc_model = SVC(kernel="linear", random_state=None)
    # svc_model = LogisticRegression(random_state=None)

    # 学習
    svc_model.fit(input_train_std, output_train)

    #traning dataのaccuracy
    pred_train = svc_model.predict(input_train_std)
    accuracy_train = accuracy_score(output_train, pred_train)
    print('traning data accuracy： %.2f' % accuracy_train)

    #test dataのaccuracy
    pred_test = svc_model.predict(input_test_std)
    # pred_testを結果出力時のためにリスト化
    pred_test_tolist = pred_test.tolist()
    accuracy_test = accuracy_score(output_test, pred_test_tolist)
    print('test data accuracy： %.2f' % accuracy_test)

    list1 = []
    list2 = []
    list3 = []
    list4 = []

        
    for i in range(len(pred_test_tolist)):
        if pred_test_tolist[i] != output_test[i]:
            list1.append(i)
            list2.append(test_indices[i])
            list3.append(pred_test_tolist[i])
            list4.append(output_test[i])
    
    print("\n")

    print("testして予測が失敗したものを以下に示す。")
    print("output_testのindex番号")
    print(list1)
    print("分類表のファイル番号")
    print(list2)
    print("testデータの予測結果")
    print(list3)
    print("予測してほしかった実際の結果")
    print(list4)


def input_data_formatting(df, dir):
    words_list = []

    for file_name in tqdm(df['file_name']):
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
        
    
    # return words_list
    return words_list
    

if __name__ == '__main__':
    main()