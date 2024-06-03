#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data_preprocessing.py
@Time    :   2024/06/04 00:12:31
@Author  :   YuFanWenShu 
@Contact :   1365240381@qq.com

'''

# here put the import lib
from config import DATA_PATH
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
import torch



def load_data():
    df = pd.read_csv(DATA_PATH, encoding='GBK', low_memory=False)
    selected_columns = ['年龄(女)', '年龄(男)', '周期序号', '不孕年限[年]', '孕[次]', '产[次]', '足月产[次]',
                        '存活子女[个]', '流产[次]', '既往移植失败周期[次]', '既往无可用胚胎周期',
                        '体重指数(女)', '内膜促排方案', '移植日E2', '移植日P', '移植日内膜厚度',
                        '临床妊娠', 'FET治疗方案']
    filtered_df = df[selected_columns]

    X = filtered_df.drop('临床妊娠', axis=1)
    y = filtered_df['临床妊娠']

    categorical_features = ['内膜促排方案', 'FET治疗方案']
    numeric_features = [x for x in X.select_dtypes(include=['int64', 'float64']).columns.tolist() if x not in categorical_features]

    numeric_transformer = MinMaxScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    transformers = [
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]

    preprocessor = ColumnTransformer(transformers=transformers)
    X_processed = preprocessor.fit_transform(X).toarray()

    X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X_processed, y, test_size=0.1, random_state=42)
    X_train_df, X_val_df, y_train_df, y_val_df = train_test_split(X_train_df, y_train_df, test_size=0.1, random_state=42)

    X_train = torch.tensor(X_train_df, dtype=torch.float32)
    X_val = torch.tensor(X_val_df, dtype=torch.float32)
    X_test = torch.tensor(X_test_df, dtype=torch.float32)

    y_train = torch.tensor(y_train_df.values, dtype=torch.long)
    y_val = torch.tensor(y_val_df.values, dtype=torch.long)
    y_test = torch.tensor(y_test_df.values, dtype=torch.long)

    encoder = LabelEncoder()
    y_train = torch.tensor(encoder.fit_transform(y_train))
    y_val = torch.tensor(encoder.fit_transform(y_val))
    y_test = torch.tensor(encoder.transform(y_test))

    return X_train, X_val, X_test, y_train, y_val, y_test
