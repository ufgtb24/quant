import numpy as np
import pandas as pd
''':arg
直接索引是 先 column 后 row
loc 索引是 先 row 后 column


'''

a=np.arange(0,30).reshape(10,3)
b=pd.DataFrame(a,index=range(10),columns=['A','B','C'])
print(b)
print(b.A.pct_change())
# print(round(b['A'].rolling(window=2).mean(), 2))

