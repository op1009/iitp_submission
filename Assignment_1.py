# -*- coding: utf-8 -*-

!huggingface-cli login

from datasets import load_dataset

ds = load_dataset("ainlpml/english-hindi")

ds

import pandas as pd

train_df = ds["train"].to_pandas()

train_df

col_eng = train_df.iloc[:10000].reset_index(drop=True)
col_hin = train_df.iloc[10000:].reset_index(drop=True)

df_trans = pd.concat([col_eng, col_hin], axis=1)

df_trans.columns = ['eng', 'hin']

# print(df_trans)
df_trans

df_trans['count_eng'] = df_trans['eng'].str.len()
df_trans['count_hin'] = df_trans['hin'].str.len()

df_trans

df_filtered = df_trans[((df_trans['count_eng'] >= 5) & (df_trans['count_eng'] <= 50)) & ((df_trans['count_hin'] >= 5) & (df_trans['count_hin'] <= 50))]

df_filtered

df_filtered['diff'] = df_filtered['count_eng'] - df_filtered['count_hin']

df_filtered

df_filtered = df_filtered[((df_filtered['diff'] >= -10) & (df_filtered['diff'] <= 10))]
df_filtered

df_filtered.columns = ["English Sentences", "Hindi Sentences", "Word Count (English)", "Word Count (Hindi)", "Difference between Word Count (English) and Word Count (Hindi)"]

df_filtered

df_filtered.to_csv('cleaned_dataset.csv')

