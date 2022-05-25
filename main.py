import os
import json
import itertools
import pprint
import pandas as pd 
import numpy as np
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=150)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from learning_file.data_format import input_data_formatting
from sklearn.linear_model import LogisticRegression


def main():
    df = pd.read_csv('分類表.csv')
    word = input_data_formatting(df)

    targets = []

    for label in df['label_number']:
        targets.append(int(label))

    # テキスト内の単語の出現頻度を数えて、結果を素性ベクトル化する(Bag of words)
    tf_idf_vectorizer = TfidfVectorizer(lowercase=False, token_pattern=r"\S+")

    # csr_matrix(疎行列)にしてndarray(多次元配列)に変形
    feature_vectors = tf_idf_vectorizer.fit_transform(word).toarray()

    input_train, input_test, output_train, output_test = train_test_split(feature_vectors, targets, test_size=0.2, random_state=0, stratify=targets)

    sc = StandardScaler()
    sc.fit(input_train)
    input_train_std = sc.transform(input_train)
    input_test_std = sc.transform(input_test)

    # 学習インスタンス生成
    svc_model = SVC(kernel='linear', random_state=None)
    # svc_model = LogisticRegression(random_state=None)

    # 学習
    svc_model.fit(input_train_std, output_train)

    #traning dataのaccuracy
    pred_train = svc_model.predict(input_train_std)
    accuracy_train = accuracy_score(output_train, pred_train)
    print('traning data accuracy： %.2f' % accuracy_train)

    #test dataのaccuracy
    pred_test = svc_model.predict(input_test_std)
    ppp = pred_test.tolist()
    accuracy_test = accuracy_score(output_test, ppp)
    print('test data accuracy： %.2f' % accuracy_test)

    list1 = []
    list2 = []
    list4 = []

        
    for i in range(len(ppp)):
        if ppp[i] != output_test[i]:
            list1.append(ppp[i])
            list2.append(output_test[i])
    
    print(list1)
    print(list2)
    
    # for j in range(len(list1)):
    #     list3 = []
    #     for _ in range(len(list1)):
    #         list3.append(0)
        
    #     list3[j] = "F"
        
    #     list4.append(list3)
    
    # # 予測と結果が一致しなかったデータのmatrix(行が予測，列が結果)
    # df1 = pd.DataFrame(list4, index = list1, columns = list2)
    # print(df1)

# データが一つしかないため避難
# descip_1,cryptography,5
# arrow,economics,6


if __name__ == '__main__':
    main()