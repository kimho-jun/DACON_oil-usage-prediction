#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from scipy import sparse
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb



train=pd.read_csv(r"C:\Users\yjhut\OneDrive\바탕 화면\데이콘 제주시 공모전 csv 파일\train.csv")
test=pd.read_csv(r"C:\Users\yjhut\OneDrive\바탕 화면\데이콘 제주시 공모전 csv 파일\test.csv")
sample_submission=pd.read_csv(r"C:\Users\yjhut\OneDrive\바탕 화면\데이콘 제주시 공모전 csv 파일\sample_submission.csv")


# In[3]:


pip install --upgrade pandas


# In[ ]:





# In[6]:


import pandas


# In[5]:


pandas.__version__


# In[28]:


def resumtable(df):
  print( f'데이터 형상: ',{df.shape})
  summary=pd.DataFrame(df.dtypes,columns=['데이터 타입'])
  summary=summary.reset_index()
  summary=summary.rename(columns={'index':'피처'})
  summary['결측값 개수']=df.isnull().sum().values
  summary['고유값 개수']=df.nunique().values
  summary['첫 번째 값']=df.loc[0].values
  summary['두 번째 값']=df.loc[1].values
  summary['세 번째 값']=df.loc[2].values

  return summary


# In[29]:


resumtable(train)


# In[30]:


# 벡터화는 road_name /  start_node_name  / end_node_name / start_turn_restricted / end_turn_restricted  / day_of_week을 진행


# In[31]:


#원본 데이터에서 삭제하려는 col중에 target 값과 상관관계가 더 높은 변수를 남겨둔다.


# In[32]:


resumtable(test)


# In[33]:


all_data=pd.concat([train,test])
all_data=all_data.drop('target',axis=1)
all_data.shape


# In[34]:


all_data=all_data.set_index('id')


# In[35]:


all_data['day_of_week']=all_data['day_of_week'].map({'일':0,'월':1,'화':2,'수':3,'목':4,'금':5,'토':6})
# 범주형 변수 인코딩 


# In[36]:


from sklearn.preprocessing import OneHotEncoder

oh_encode=OneHotEncoder()

encoded_road_matrix=oh_encode.fit_transform(all_data[['road_name']])

all_data=all_data.drop('road_name',axis=1)

encoded_road_matrix

#효율, 연산시간 save 위해  -> CSR 형식



# In[37]:


encoded_end_matrix=oh_encode.fit_transform(all_data[['end_node_name']])
all_data=all_data.drop('end_node_name',axis=1)


encoded_end_matrix

# 변환할 범주형 뱐수들 모두 csr_matrix 형태로 인코딩


# In[38]:


encoded_start_matrix=oh_encode.fit_transform(all_data[['start_node_name']])
all_data=all_data.drop('start_node_name',axis=1)

encoded_start_matrix

# 변환할 범주형 뱐수들 모두 csr_matrix 형태로 인코딩


# In[40]:


all_data['start_turn_restricted']=all_data['start_turn_restricted'].map({'있음':1,'없음':0})
all_data['end_turn_restricted']=all_data['end_turn_restricted'].map({'있음':1,'없음':0})

# start_turn_restricted , end_turn_restricted는 map 함수로 진행 


# In[41]:


all_data.describe()


# In[42]:


from sklearn.preprocessing import StandardScaler

s_scaler=StandardScaler()
all_data[['weight_restricted']]=s_scaler.fit_transform(all_data[['weight_restricted']])


# In[43]:


all_data=all_data.drop('connect_code',axis=1)
all_data=all_data.drop('multi_linked',axis=1)
all_data=all_data.drop('day_of_week',axis=1)


# In[44]:


all_data_sprs2=sparse.hstack([sparse.csr_matrix(all_data),
                            encoded_end_matrix,
                            encoded_start_matrix,
                            encoded_road_matrix],
                            format='csr')

# 이전 모든 결과 하나의 matrix로 생성


# In[45]:


all_data_sprs2


# In[46]:


from sklearn.model_selection import train_test_split

num_train=len(train)

X_train=all_data_sprs2[:num_train]
X_test=all_data_sprs2[num_train:]

y=train['target']

X_tr,X_val,y_tr,y_val=train_test_split(X_train,y,
                                      test_size=0.2,
                                      random_state=43)


# In[47]:


lgbr_model=lgb.LGBMRegressor(random_state=43, learning_rate=0.1, num_iterations=5000 ,max_depth=12, boosting_type='goss')
                                                                 
lgbr_model.fit(X_tr,y_tr)

lgb_preds=lgbr_model.predict(X_val)

print(f'lgbr모델의 MAE 값:  {mean_absolute_error(y_val,lgb_preds):.4f}')

# LGBMRgressor 모델의 MAE 값:  3.0808  goss  학습률:0.1   depth=10
# LGBMRgressor 모델의 MAE 값:  3.1926  goss  학습률:0.05  depth=8
# LGBMRgressor 모델의 MAE 값:  3.1796  goss  학습률:0.05  depth=10


#  gbr모델의 MAE 값:  3.1426   gbdt


#lgbr모델의 MAE 값:   3.0771    test_size= 0.2 나머지는 맨 윗줄과 동일하게 진행

#lgbr모델의 MAE 값:   3.0790    test_size=0.25로 놓고 나머지는 바로 윗줄과 동일하게 진행

#lgbr모델의 MAE 값:    3.0796    w.r를 frobustscaler가 아닌 제거하고 진행

# 3.0606

#lgbr모델의 MAE 값:  3.0560  max_depth=11

#lgbr모델의 MAE 값:  3.0538  max_depth=12

#lgbr모델의 MAE 값:  3.0579  test_size=0.25로 하니까 오히려 오차가 증가함  max_depth=12 그대로 했음 


# In[56]:


lgbr_model2=lgb.LGBMRegressor(random_state=43)
ddff=all_data[:num_train]
pred3=lgbr_model2.fit(ddff,y)




# In[57]:


from lightgbm import plot_importance
import matplotlib.pyplot as plt

fig,ax=plt.subplots(figsize=(10,10))
plot_importance(lgbr_model2,ax=ax)


# In[ ]:


lgbr_preds2=lgbr_model.predict(X_test)


# In[ ]:


sample_submission['target']=lgbr_preds2


# In[ ]:


sample_submission.to_csv('submission_sample1101(1).csv',index=False)


# In[ ]:


# all_data=all_data.drop(['vehicle_restricted','height_restricted'],axis=1)


# In[ ]:


from sklearn.model_selection import GridSearchCV

param_grid={'learning_rate':[0.1,0.05], 'max_depth':[10]}

lgb_grid_model=GridSearchCV(estimator=lgb.LGBMRegressor(random_state=43, 
                                                        num_iterations=5000 , 
                                                        boosting_type='goss'),
                             param_grid=param_grid,
                             cv=2)

lgb_grid_model.fit(X_tr,y_tr)
print(lgb_grid_model.best_params_)

lgb_grid_model_preds=lgb_grid_model.best_params_.predict(X_val)

print(f'lgb + gridS의  MAE값: {mean_absolute_error(y_val,lgb_grid_model_preds):.4f}')


# In[ ]:


# from sklearn.model_selection import RandomizedSearchCV

# param={
#     'max_depth':[6,7,8,9,10],
#     'num_iterations':[3000,4000,5000],
#     'random_state':[43],
#     'learning_rate':[0.01,0.05,0.25,0.1]
# }

# n_iter=30

# # lgbr_model=lgb.LGBMRegressor(random_state=43, learning_rate=0.1, num_iterations=5000 ,max_depth=8, boosting_type='goss')

# lgb_random_model=RandomizedSearchCV(estimator=lgb.LGBMRegressor(),
#                                 param_distributions=param,
#                                 cv=5,
#                                 n_iter=n_iter)


# lgb_random_model.fit(X_tr,y_tr)

# print(lgb_random_model.best_params_)

# random_model_preds=lgb_random_model.best_params_.predict(X_val)

# print(f'lgb + 랜덤서치 MAE값: {mean_absolute_error(y_val,random_model_preds):.4f}')



# In[ ]:





# In[ ]:





# In[ ]:




