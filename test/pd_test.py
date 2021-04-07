import numpy as np
import pandas as pd
''':arg
直接索引是 先 column 后 row
loc 索引是 先 row 后 column


'''

a=np.arange(1,10).reshape(3,3)
b=pd.DataFrame(a,index=[0,1,2],columns=['A','B','C'])
b.set_index()
print(b)
print('=============')
# b.index=[-1,-2,-3]
# print(b.B)

print(b[['A','B']] )
