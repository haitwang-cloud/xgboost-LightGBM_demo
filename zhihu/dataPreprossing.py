import pandas as pd
from sklearn.model_selection import train_test_split
df=pd.read_csv('data.csv')

col=['follower','following','agree_num',
    'appreciate_num','star_num','share_num',
    'ask_num','answer_num','article_num',
    'collect_num','public_edit_num']
df_Blog2=df[col]
print(df_Blog2.shape)
df_Blog2=df_Blog2[pd.notnull(df_Blog2['follower'])]
print(df_Blog2.shape)
df_Blog2=df_Blog2[pd.notnull(df_Blog2['following'])]
print(df_Blog2.shape)
df_Blog2=df_Blog2[pd.notnull(df_Blog2['agree_num'])]
print(df_Blog2.shape)
df_Blog2=df_Blog2[pd.notnull(df_Blog2['appreciate_num'])]
print(df_Blog2.shape)
df_Blog2=df_Blog2[pd.notnull(df_Blog2['share_num'])]
print(df_Blog2.shape)
df_Blog2=df_Blog2[pd.notnull(df_Blog2['ask_num'])]
print(df_Blog2.shape)
df_Blog2=df_Blog2[pd.notnull(df_Blog2['answer_num'])]
print(df_Blog2.shape)
df_Blog2=df_Blog2[pd.notnull(df_Blog2['article_num'])]
print(df_Blog2.shape)
df_Blog2=df_Blog2[pd.notnull(df_Blog2['collect_num'])]
print(df_Blog2.shape)
df_Blog2=df_Blog2[pd.notnull(df_Blog2['public_edit_num'])]
print(df_Blog2.shape)
df_Blog2.to_csv('dataset.csv',header=col,index=False)

# train,test=train_test_split(df_Blog2,test_size=0.25)
# print(train.shape,test.shape)
# train.to_csv('train.csv',header=col,index=False)
# test.to_csv('test.csv',header=col,index=False)