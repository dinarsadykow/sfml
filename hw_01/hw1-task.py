
# coding: utf-8

# In[1]:


import pandas as pd


# ### (1) Используя параметры pandas прочитать красиво пандас 

# In[87]:


df = pd.read_csv('UCI_Credit_Card.csv') #TODO

columns = []
for line in list(df.columns):
    columns.append(line.strip().replace('.',' '))
df.columns = columns

df.head()


# ### (2) выведите, что за типы переменных, сколько пропусков,
# ### для численных значений посчитайте пару статистик (в свободной форме)

# In[28]:


df.info()


# In[29]:


df.describe()


# ### (3) посчитать число женщин с университетским образованием
# ### SEX (1 = male; 2 = female). 
# ### EDUCATION (1 = graduate school; 2 = university; 3 = high school; 4 = others). 

# In[52]:


df[df['EDUCATION'] == 3][['SEX','EDUCATION']].pivot_table(index=['SEX'], aggfunc='count')


# ### (4) Сгрупировать по "default payment next month" и посчитать медиану для всех показателей начинающихся на BILL_ и PAY_

# In[79]:


bill_pay_columns = []
for line in list(df.columns):
    if 'BILL' in line or 'PAY' in line:
        bill_pay_columns.append(line)

df.groupby(['default payment next month'], as_index=False)[bill_pay_columns].mean().T


# ### (5) постройте сводную таблицу (pivot table) по SEX, EDUCATION, MARRIAGE

# In[91]:


df.pivot_table(index = ['SEX','EDUCATION','MARRIAGE'],aggfunc='count').head()


# ### (6) Создать новый строковый столбец в data frame-е, который:
# ### принимает значение A, если значение LIMIT_BAL <=10000
# ### принимает значение B, если значение LIMIT_BAL <=100000 и >10000
# ### принимает значение C, если значение LIMIT_BAL <=200000 и >100000
# ### принимает значение D, если значение LIMIT_BAL <=400000 и >200000
# ### принимает значение E, если значение LIMIT_BAL <=700000 и >400000
# ### принимает значение F, если значение LIMIT_BAL >700000

# In[104]:


def ABCDEF_def(LIMIT_BAL):
    if LIMIT_BAL <=10000:
        return 'A'
    if LIMIT_BAL <=100000 and LIMIT_BAL>10000:
        return 'B'
    if LIMIT_BAL <=200000 and LIMIT_BAL>100000:
        return 'C'
    if LIMIT_BAL <=400000 and LIMIT_BAL>200000:
        return 'D'
    if LIMIT_BAL <=700000 and LIMIT_BAL>400000:
        return 'E'
    if LIMIT_BAL >700000:
        return 'F'
    
df['ABCDEF'] = df['LIMIT_BAL'].map(lambda x: ABCDEF_def(x))
df.head()


# In[106]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### (7) постироить распределение LIMIT_BAL (гистрограмму)

# In[107]:


df['LIMIT_BAL'].hist(alpha=0.6)


# ### (8) построить среднее значение кредитного лимита для каждого вида образования 
# ### и для каждого пола
# ### график необходимо сделать очень широким (на весь экран)

# In[111]:


_, ax = plt.subplots(figsize=(20,8))

df.pivot_table(values='LIMIT_BAL', index='SEX', columns='EDUCATION', aggfunc='mean').plot(
    kind='bar', stacked=False, ax=ax)
plt.show()


# ### (9) построить зависимость кредитного лимита и образования только для одного из полов
# 

# In[166]:


df[df.SEX == 1][['EDUCATION','LIMIT_BAL']].pivot_table(values='LIMIT_BAL', index='EDUCATION',  aggfunc='mean').plot(
    kind='bar', stacked=True)
plt.title("SEX = 1");


# ### (10) построить большой график (подсказка - используя seaborn) для построения завимисости всех возможных пар параметров
# ### разным цветом выделить разные значение "default payment next month"
# ### (но так как столбцов много - картинка может получиться "монструозной")
# ### (поэкспериментируйте над тем как построить подобное сравнение параметров)
# ### (подсказка - ответ может состоять из несколькольких графиков)
# ### (если не выйдет - программа минимум - построить один график со всеми параметрами)
# 

# In[178]:


import seaborn as sns

sns_plot = sns.pairplot(df)
sns_plot.savefig('10.png')

