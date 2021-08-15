#%%
import pandas as pd 
import numpy as np
train_df = pd.read_csv('train.csv')
train_df
test_df = pd.read_csv('test.csv')
submission_df = pd.read_csv('random_forest_sample_submission.csv')
# train_df.info()
# train_df.isna().sum()
train_df1 = train_df.copy()
train_df1.dropna(inplace=True)
train_df.isna().sum()
def correlation(data,col1,col2):
    colm1 = data[col1]
    colm2 = data[col2]
    return colm1.corr(colm2)
test_df1 = test_df.copy()   
correlation(train_df,'trade_price','weight')  #0.086  
correlation(train_df,'trade_price','current_coupon') #0.044   
correlation(train_df,'trade_price','time_to_maturity')#0.12   
correlation(train_df,'trade_price','trade_size')   #0.55 
train_df
def null_values(data):
    null_vales = data.isna().sum()
    return null_vales
null_vales = null_values(train_df1).sort_values(ascending=False)
selected_columns = ['trade_price','weight','current_coupon','time_to_maturity','is_callable',
'reporting_delay','trade_size','trade_type','curve_based_price']
train_df.columns
train_data = train_df1[selected_columns]
train_data
test_data = test_df1[selected_columns[1:]]
null_vales_ts  =null_values(test_data).sort_values(ascending=False)
null_vales_ts
train_data
test_data
input_col = list(train_data.columns)[1:]
input_col
target_col  = 'trade_price'
train_input = train_data[input_col].copy()
train_target = train_data[target_col].copy()
train_target
from sklearn.model_selection import train_test_split
x_train, x_vald, y_train, y_vald = train_test_split(train_input, train_target, test_size=0.25, random_state=42)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train
numeric_colms = train_input.select_dtypes(include=np.number).columns.tolist()
numeric_colms
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
fit1 = scaler.fit(train_input[numeric_colms])
x_train[numeric_colms] = scaler.transform(x_train[numeric_colms])
x_train
x_vald[numeric_colms] = scaler.transform(x_vald[numeric_colms])
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(x_train,y_train)
trian_preds = model.predict(x_train)
train_acc = model.score(x_train,y_train)
train_acc
vald_acc = model.score(x_vald,y_vald)
vald_acc

weights_df = pd.DataFrame({
    'feature': np.append(numeric_colms, 1),
    'weight': np.append(model.coef_, model.intercept_)
})
weights_df.sort_values('weight', ascending=False)
test_data
scaler1 = MinMaxScaler()
fit2 = scaler.fit(test_data[numeric_colms])
test_data[numeric_colms] = scaler.transform(test_data[numeric_colms])
test_data
X_test =test_data
final_predictions = model.predict(X_test)
final_predictions
predictions = (final_predictions)
test_id = submission_df['id']
submission = pd.DataFrame({"Id": test_id, "risk_flag_predict": predictions})
submission.to_csv('submission.csv')
#%%













































# trade_type =  train_df['trade_type'] 
# # trade_type
# def tarde_type(data,new_colm,old_colm,number):
#     data[new_colm]=data[old_colm] == number

# # train_df['sale'] =( train_df['trade_type'] ==3 )
# # train_df
# # for trade in range(2,11):
# #     tarde_type(train_df,'sale{}'.format(trade),' trade_type_last{}'.format(trade),2)
# #     tarde_type(train_df,'buy{}'.format(trade),' trade_type_last{}'.format(trade),3)
# #     tarde_type(train_df,'dealers{}'.format(trade),' trade_type_last{}'.format(trade),4)
# #     # print('trade_type_last{}'.format(trade))
# # train_df
# # trade_type_last2

# #another way 
# selected_colms =['id','bond_id','trade_price','weight','current_coupon','time_to_maturity','is_callable','reporting_delay','trade_size','trade_type','curve_based_price','received_time_diff_last1']
# selected_colms2=['received_time_diff_last2','trade_price_last1','trade_size_last2','curve_based_price_last2','received_time_diff_last3','trade_size_last3','curve_based_price_last3','trade_size_last4']
# # # train_df.info()
# train_data = train_df[selected_colms].iloc[:61146].copy() 
# vald_data = train_df[selected_colms2].copy()
# # train_data
# # submission_df   
# # train_data
# # tarde_type(train_data,'sale','trade_type',2)
# # tarde_type(train_data,'buy','trade_type',3)
# # tarde_type(train_data,'dealers','trade_type',4) 
# del train_data['trade_type']
# # train_data    
# numeric = list(train_data)[3:12] 
# numeric_v = list(vald_data) [0:]   
# # numeric            
# # categorical =list(train_data)[11:]   
# # print(categorical)
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# fit_process = scaler.fit(train_data[numeric])
# train_data[numeric] = scaler.transform(train_data[numeric])
# train_data[numeric] 
# vald_data[numeric_v] = scaler.transform(vald_data[numeric_v])


# #categorical 
# # from sklearn.preprocessing import OneHotEncoder
# # encoder = OneHotEncoder(sparse=False,handle_unknown ='ignore')
# # fit_process = encoder.fit(train_data[categorical])
# # train_data[categorical]= encoder.transform(train_data[categorical])
# # train_data[categorical]
# target = 'trade_price'
# target2 = 'trade_price_last1'
# train_target = train_data[target]
# vald_targets = vald_data[target2]
# X_train = train_data[numeric] 
# X_val = vald_data[numeric_v]
# from sklearn.linear_model import LinearRegression
# model = LinearRegression()
# fit_model = model.fit(X_train,train_target)
# pred_train = model.predict(X_train)
# score_train =model.score(X_train, train_target)
# score_train
# score_train =model.score(X_train, train_target)
# score_vald = model.score(X_val,vald_targets)
# score_vald
# # from sklearn.model_selection import train_test_split
# # train_val_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42)
# # trains_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)
# # val_df
# # trains_df
# # input_colms = list(trains_df)[3:]
# # input_colms
