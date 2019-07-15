import var_bin
exile=['asqbh','bill_month','yq_flag','oot2','oot']
classvar = ['']
tmp_f = 'mak'
htlab = qcut_stable(data=var_data,features=tmp_f, target='yq_flag', bin_pct=0.05)
htlab.features_std(exile, classvar)
htlab.stable_cacule()
# 描述性分析
htlab.var_desc()
htlab.tbbinresult['varbin']=htlab.tbbinresult['varbin'].reset_index(level=1)
# 结果保存excel
htlab.save_result('repay_feature_desc_woe.xlsx')
