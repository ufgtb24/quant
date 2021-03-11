import math

import numpy as np
import matplotlib.pyplot as plt

def moving_average(ma,value,n):
    return ma*(n-1)/n+value/n


def experiment(start_money, proportion, gain_rate, loss_rate, results):
    test=np.sum(results)
    for result in results:
        if result:
            start_money= start_money* (1+proportion * gain_rate)
        else:
            start_money= start_money*(1-proportion * loss_rate)
    return start_money



def repeat_exp(ex_time,init_money,proportions,trade_time,win_rate,gain_rate,loss_rate,random):
    average=np.ones_like(proportions)
    for i in range(ex_time):
        prop_sum = []
        for proportion in proportions:
            if random:
                results = np.random.choice([1,0], trade_time, p=[win_rate, 1-win_rate])
            else:
                win_time=math.ceil(trade_time*win_rate)
                results = np.concatenate([np.ones(win_time),np.zeros(trade_time-win_time)])
            money=experiment(init_money,proportion,gain_rate,loss_rate,results)
            prop_sum.append(money)
        average=moving_average(average,np.array(prop_sum),i+1)
    return average
    
proportions=np.linspace(0, 1, 101)


# ##### 期权
p=0.5  # 胜率
b=1.5  # 赚的比例（除去本金）
c=0.6  # 亏的比例（关于本金）

##### 股票
# p=0.6  # 胜率
# b=0.2  # 赚的比例（除去本金）
# c=0.1  # 亏的比例（关于本金）

sum=repeat_exp(ex_time=1,
               init_money=1,
               proportions=proportions,
               trade_time = 5,
               win_rate=p,
               gain_rate=b,
               loss_rate=c,
               random=False
               )

f=(p*b-c*(1-p))/(b*c)
print('投注比例=',f)
fig, ax = plt.subplots()
ax.plot(proportions, sum)
plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)
# ax.ticklabel_format(useOffset=False)
plt.show()
    
    

