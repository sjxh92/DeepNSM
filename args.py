import pandas as pd

data = {'证券名称': ['格力电器', '视觉中国', '成都银行', '中国联通', '格力电器', '视觉中国', '成都银行', '中国联通'],
        '摘要': ['证券买入', '证券买入', '证券买入', '证券买入', '证券卖出', '证券卖出', '证券卖出', '证券卖出'],
        '成交数量': [500, 1000, 1500, 2000, 500, 500, 1000, 1500],
        '成交金额': [-5000, -10000, -15000, -20000, 5500, 5500, 11000, 15000]}

df = pd.DataFrame(data, index=['2018-2-1', '2018-2-1', '2018-2-1', '2018-2-1', '2018-2-2', '2018-2-2', '2018-2-2',
                               '2018-2-3'])

print(df)
print()

# df.drop(df['成交金额'].isin([11000]), inplace=True)
df = df[~df['成交金额'].isin([15000])]

print(df)

