import pandas as pd
import collections
import os

df1 = pd.read_csv('efficientnetb4_output.csv')
df2 = pd.read_csv('efficientnetb6_output.csv')
df3 = pd.read_csv('resnet18.csv')
df4 = pd.read_csv('resnet152.csv')
df5 = pd.read_csv('vit_small_patch16_384.csv')
df6 = pd.read_csv('vgg19.csv')
df7 = pd.read_csv('swin_base_patch4_window7_224.csv')
df8 = pd.read_csv('efficient_b4.csv')

test_dir = '/opt/ml/input/data/eval'
submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))

answer = [0] * len(df1)
for i in range(len(df1)):
    tmp = []
    tmp.append([df1['ans'][i],df2['ans'][i],df3['ans'][i],df4['ans'][i],df5['ans'][i],df6['ans'][i],df7['ans'][i],df8['ans'][i]])
    most = collections.Counter(tmp[0])
    answer[i] = most.most_common(n=1)[0][0]

submission['ans'] = answer
submission.to_csv('submission_ensemble.csv', index=False)