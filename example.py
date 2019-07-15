from var_bin import qcut_stable
import pandas as pd
#包含标签的训练数据
test_data = pd.read_csv('test_data.csv')
classvar = [] #类别型变量
selected_features  = []#所有要分析的变量，包括标签
htlab = qcut_stable(data=var_data,features=selected_features, target='label', bin_pct=0.05)
htlab.features_std(exile, classvar)
htlab.stable_cacule()
# 描述性分析
htlab.var_desc()
htlab.tbbinresult['varbin']=htlab.tbbinresult['varbin'].reset_index(level=1)
# 结果保存excel
htlab.save_result('xxx.xlsx')
