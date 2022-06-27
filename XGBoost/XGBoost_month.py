"""
    本策略中将部分数据整理后存储至本地文档，便于查看与使用，故运行此程序前，请查看代码并建立好相关文件夹
    需建立4个文件夹：data用于储存数据处理前截面期数据(329行)，redata用于储存数据处理后截面期数据(340行），
        section用于储存4个子区间样本内数据（245行），result用于储存最后结果数据（402行）
    另有热力图图片与净值曲线图片储存（372行。459行）
"""
from causis_api.const import get_version
from causis_api.const import login

login.username = 'haochen.feng'
login.password = 'wtmjsnbb136122'
login.version = get_version()
import calendar
import xgboost as xgb
from causis_api.factor import *
from sklearn.metrics import accuracy_score
from matplotlib import ticker

# 使图片中中文与负数正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

start_date = '2014-01-01'
end_date = '2021-12-31'





"""
函数部分
"""

# 市值因子 in_capital=ln(总市值)
def marketcap_factor(stock_valuation_info):
    size_df = stock_valuation_info.xs('MarketCap', level=1) * 1e8  # 总市值（元）
    size_df = pd.DataFrame(size_df, dtype=float)
    ln_size_df = size_df.apply(np.log)
    return ln_size_df


#  EP因子 EP=Quant_code/pe
def ep_factor(stock_valuation_info):
    pe_df = stock_valuation_info.xs('PeRatio', level=1)
    pe_df = pd.DataFrame(pe_df, dtype=float)
    ep_df = 1 / pe_df
    return ep_df


# BP因子 BP=Quant_code/pb
def bp_factor(stock_valuation_info):
    pb_df = stock_valuation_info.xs('PBRatio', level=1)
    pb_df = pd.DataFrame(pb_df, dtype=float)
    bp_df = 1 / pb_df
    return bp_df


# 价格因子 in_price=ln(price)
def price_factor(stock_price_info):
    price_df = stock_price_info.xs('Open', level=1)
    price_df = pd.DataFrame(price_df, dtype=float)
    ln_price_df = price_df.apply(np.log)
    return ln_price_df


# 换手率因子 turn_1m=1月内日换手率的平均值
def turn_1m(stock_valuation_info, time_list):
    turn_1m_df = []
    turn = stock_valuation_info.xs('TurnoverRatio', level=1)
    for i in time_list:
        result = np.mean(turn.loc[:, i[0]:i[1]], axis=1).tolist()
        turn_1m_df.append(result)
    turn_1m_df = np.array(turn_1m_df).T
    turn_1m_df = pd.DataFrame(turn_1m_df, columns=[i[-1] for i in time_list], index=stock)
    return turn_1m_df


# 动量反转因子 wgt_turn=1月内每日换手率乘每日收益率的算术平均值
def wgt_turn(stock_price_info, stock_valuation_info, time_list):
    wgt_turn_df = []
    pct_change = stock_price_info.xs('Close', level=1).pct_change(axis=1)
    turnover = stock_valuation_info.xs('TurnoverRatio', level=1)
    wgt_return = pct_change * turnover
    for i in time_list:
        result = np.mean(wgt_return.loc[:, i[0]:i[1]], axis=1).tolist()
        wgt_turn_df.append(result)
    wgt_turn_df = np.array(wgt_turn_df).T
    wgt_turn_df = pd.DataFrame(wgt_turn_df, columns=[i[-1] for i in time_list], index=stock)
    return abs(wgt_turn_df)


# 波动率因子 std_1m=1月内日收益率序列标准差
def std_1m(stock_price_info, time_list):
    std_1_df = []
    stock_price_info = stock_price_info
    pct_change = stock_price_info.xs('Close', level=1).pct_change(axis=1)

    for i in time_list:
        result = np.std(pct_change.loc[:, i[0]:i[1]], axis=1).tolist()
        std_1_df.append(result)
    std_1_df = np.array(std_1_df).T
    std_1_df = pd.DataFrame(std_1_df, columns=[i[-1] for i in time_list], index=stock)
    return std_1_df


# 获取ROE，Sales_G_q,gross_profit_margin_q因子
def factor_compilation(stock_indicator_info, last_day_lst):
    factor_compilation_lst = []
    t_stock_indicator_info = pd.concat([stock_indicator_info, stock_indicator_info, stock_indicator_info], axis=1)
    t_stock_indicator_info = t_stock_indicator_info.sort_index(axis=1)
    t_stock_indicator_info.columns = last_day_lst
    factor_compilation_lst.append(t_stock_indicator_info.xs('ROE', level=1))
    factor_compilation_lst.append(t_stock_indicator_info.xs('GrossProfitMargin', level=1))
    factor_compilation_lst.append(t_stock_indicator_info.xs('IncTotalRevenueYearOnYear', level=1))
    return factor_compilation_lst


# 获取股票超额收益率，以沪深300为基准
def yield_rate(stock_price_info, CSI300_price_info, new_time_lst):
    stock_price_rate_df = []
    csi300_price_rate_df = []
    stock_price_df = stock_price_info.xs('Close', level=1)
    CSI300_price_df = CSI300_price_info.xs('Close', level=1)
    for i in range(0, len(new_time_lst) - 1):
        stock_month_rate = (stock_price_df.loc[:, new_time_lst[i + 1][-1]] - stock_price_df.loc[:,
                                                                             new_time_lst[i][-1]]) / \
                           stock_price_df.loc[:, new_time_lst[i][-1]].tolist()
        CSI300_price_rate = (CSI300_price_df.loc[:, new_time_lst[i + 1][-1]] - CSI300_price_df.loc[:,
                                                                               new_time_lst[i][-1]]) / \
                            CSI300_price_df.loc[:, new_time_lst[i][-1]].tolist()
        stock_price_rate_df.append(stock_month_rate)
        csi300_price_rate_df.append(CSI300_price_rate)
    stock_price_rate_df = np.array(stock_price_rate_df).T
    stock_price_rate_df = pd.DataFrame(stock_price_rate_df, index=stock, columns=[i[-1] for i in time_lst])
    stock_price_rate_df.to_csv('D:/check.csv')
    csi300_price_rate_df = pd.DataFrame(csi300_price_rate_df)
    csi300_price_rate_se = pd.Series(csi300_price_rate_df['S.CN.SSE.000300'])
    csi300_price_rate_se.index = [i[-1] for i in time_lst]
    stock_overturn_rate_df = stock_price_rate_df - csi300_price_rate_se
    return stock_overturn_rate_df


# 获取股票当月策略收益
def strategy_return(stock_price_info, time_lst):
    stock_price_rate_df = []
    stock_price_df = stock_price_info.xs('Open', level=1)
    for i in range(0, len(time_lst)):
        stock_month_rate = (stock_price_df.loc[:, time_lst[i][1]] - stock_price_df.loc[:,
                                                                    time_lst[i][0]]) / \
                           stock_price_df.loc[:, time_lst[i][0]].tolist()
        stock_price_rate_df.append(stock_month_rate)
    stock_price_rate_df = np.array(stock_price_rate_df).T
    stock_price_rate_df = pd.DataFrame(stock_price_rate_df, index=stock, columns=[i[-1] for i in time_lst])
    stock_overturn_rate_df = stock_price_rate_df
    return stock_overturn_rate_df


# 获取股票当月超额收益
def strategy_yield_rate(stock_price_info, CSI300_price_info, time_lst):
    stock_price_rate_df = []
    csi300_price_rate_df = []
    stock_price_df = stock_price_info.xs('Open', level=1)
    CSI300_price_df = CSI300_price_info.xs('Open', level=1)

    for i in range(0, len(time_lst)):
        stock_month_rate = (stock_price_df.loc[:, time_lst[i][1]] - stock_price_df.loc[:,
                                                                    time_lst[i][0]]) / \
                           stock_price_df.loc[:, time_lst[i][0]].tolist()

        CSI300_price_rate = (CSI300_price_df.loc[:, time_lst[i][1]] - CSI300_price_df.loc[:,
                                                                      time_lst[i][0]]) / \
                            CSI300_price_df.loc[:, new_time_lst[i][0]].tolist()
        stock_price_rate_df.append(stock_month_rate)
        csi300_price_rate_df.append(CSI300_price_rate)
    stock_price_rate_df = np.array(stock_price_rate_df).T
    stock_price_rate_df = pd.DataFrame(stock_price_rate_df, index=stock, columns=[i[-1] for i in time_lst])
    csi300_price_rate_df = pd.DataFrame(csi300_price_rate_df)
    csi300_price_rate_se = pd.Series(csi300_price_rate_df['S.CN.SSE.000300'])
    csi300_price_rate_se.index = [i[-1] for i in time_lst]
    stock_overturn_rate_df = stock_price_rate_df - csi300_price_rate_se
    return stock_overturn_rate_df


# 获取时间参数
def get_time_range_list(startdate, enddate):
    """
    获取每月股票交易首尾日期
    """
    section_time = []
    date_range_list = []
    startdate = datetime.datetime.strptime(startdate, '%Y-%m-%d')
    enddate = datetime.datetime.strptime(enddate, '%Y-%m-%d')
    while 1:
        next_month = startdate + datetime.timedelta(days=calendar.monthrange(startdate.year, startdate.month)[1])
        month_end = next_month - datetime.timedelta(days=1)
        if month_end < enddate:
            date_range_list.append((datetime.datetime.strftime(startdate, '%Y-%m-%d'),
                                    datetime.datetime.strftime(month_end, '%Y-%m-%d')))
            startdate = next_month
        else:
            break
    for date in date_range_list:
        section_time.append([get_trading_dates(date[0], date[1])[0], get_trading_dates(date[0], date[1])[-1]])
    return section_time


# 中位数去极值
def factor_mad(factor):
    dm = np.median(factor)
    dm1 = np.median(abs(factor - dm))
    up = dm + (5 * dm1)
    down = dm - (5 * dm1)
    factor = np.where(factor > up, up, factor)
    factor = np.where(factor < down, down, factor)
    return factor


# 标准化
def standardize(factor):
    factor_data = (factor - factor.mean()) / factor.std()
    return factor_data


# 合并所有样本作调参使用
def factor_train():
    path = 'D:/redata/'
    files = os.listdir(path)
    model_dt = pd.DataFrame()
    for files_name in files:
        files_path = path + files_name
        dt = pd.read_csv(files_path, index_col=0, header=0).dropna()
        dt.sort_values('excess_return', inplace=True, ascending=True)
        half_point = int(len(dt.index) / 2)
        score_excess_return = [0] * half_point + [1] * int(len(dt.index) - half_point)
        dt['excess_return'] = score_excess_return
        model_dt = model_dt.append(dt)
    model_dt.to_csv('D:/model_data.csv')


# 合并每个子区间样本内数据并保存本地
def section_train_test():
    path = 'D:/redata/'
    files = os.listdir(path)
    for i in range(0, 5):
        section_train = pd.DataFrame()
        for files_name in files[i * 12:(i + 3) * 12]:
            files_path = path + files_name
            dt = pd.read_csv(files_path, index_col=0, header=0).dropna()
            dt.sort_values('excess_return', inplace=True, ascending=True)
            half_point = int(len(dt.index) / 2)
            score_excess_return = [0] * half_point + [1] * int(len(dt.index) - half_point)
            dt['excess_return'] = score_excess_return
            section_train = section_train.append(dt)
        section_train.to_csv('D:/section/section_' + str(i + 2017) + '_train.csv')


# 获取基准收益
def base_return_ratio(CSI300_price_info, new_time_lst):
    csi300_price_rate_df = []
    CSI300_price_df = CSI300_price_info.xs('Close', level=1)
    for i in range(0, len(new_time_lst) - 1):
        CSI300_price_rate = (CSI300_price_df.loc[:, new_time_lst[i + 1][1]] - CSI300_price_df.loc[:,
                                                                              new_time_lst[i][1]]) / \
                            CSI300_price_df.loc[:, new_time_lst[i][1]].tolist()
        csi300_price_rate_df.append(CSI300_price_rate)
    csi300_price_rate_df = pd.DataFrame(csi300_price_rate_df)
    csi300_price_rate_lst = csi300_price_rate_df.loc[:, 'S.CN.SSE.000300'].tolist()
    return csi300_price_rate_lst[36:-1]


# 计算最大回撤
def maxdrawdown(return_list):
    i = np.argmax((np.maximum.accumulate(return_list) - return_list) / np.maximum.accumulate(return_list))
    if i == 0:
        return 0
    j = np.argmax(return_list[:i])
    return (return_list[j] - return_list[i]) / (return_list[j])


# 获取时间列表
time_lst = get_time_range_list('2014-01-01', '2022-01-01')
# 获取交易截面日
last_day_lst = pd.to_datetime([i[-1] for i in time_lst])
last_day_lst_str = [i[-1] for i in time_lst]
new_time_lst = get_time_range_list('2014-01-01', '2022-02-01')

# 获取股票估值Dataframe
valuation_df = pd.read_pickle('stock_valuation_standard_')
stock_valuation_standard = valuation_df.loc[:, '2013-12-31':'2022-01-01'].replace(0, np.nan)

# 获取股票列表
stock = []
for el in valuation_df.index:
    if stock.count(el[0]) < 1:
        stock.append(el[0])

# 获取股票价格Dataframe
price_df = pd.read_pickle('stock_price_standard_')
stock_price_standard = price_df.loc[:, '2013-12-31':'2022-01-01']
special_price_standard = price_df.loc[:, '2013-12-31':'2022-01-31'].replace(0, np.nan)

# 获取指数priceDataframe并标准化
CSI300_price_ori = get_price('S.CN.SSE.000300', '2013-12-31', '2022-01-31')
CSI300_price_standard = price_standard(CSI300_price_ori).replace(0, np.nan)

# 获取股票财务信息并标准化
indicator_info = get_indicator(stock, start_date, end_date)
stock_indicator_standard = stock_finance_standard(indicator_info).replace(0, np.nan)

# 生成因子Dataframe
marketcap_factor_df = marketcap_factor(stock_valuation_standard)
ep_factor_df = ep_factor(stock_valuation_standard)
bp_factor_df = bp_factor(stock_valuation_standard)
price_factor_df = price_factor(stock_price_standard)
turn_factor_df = turn_1m(stock_valuation_standard, time_lst)
std_1m_df = std_1m(stock_price_standard, time_lst)
wgt_turn_df = wgt_turn(stock_price_standard, stock_valuation_standard, time_lst)
factor_compilation_df = factor_compilation(stock_indicator_standard, last_day_lst)
roe_factor_df = factor_compilation_df[0]
sales_G_q_factor_df = factor_compilation_df[1]
grossprofitmargin_factor_df = factor_compilation_df[2]
yield_rate_df = yield_rate(special_price_standard, CSI300_price_standard, new_time_lst)
strategy_return_df = strategy_return(stock_price_standard, time_lst)
strategy_yield_rate_df = strategy_yield_rate(stock_price_standard, CSI300_price_standard, time_lst)

# 结果汇总至本地文档。
column = ['in_capital', 'EP', 'BP', 'in_price', 'turn_1m', 'wgt_turn', 'std_1m', 'ROE', 'Sales_G_q',
          'Grossprofitmargin', 'excess_return', 'strategy_return', 'strategy_excess_return']
for day_time in last_day_lst_str:
    one_day_result = pd.concat(
        [marketcap_factor_df[day_time], ep_factor_df[day_time], bp_factor_df[day_time], price_factor_df[day_time],
         turn_factor_df[day_time], wgt_turn_df[day_time], std_1m_df[day_time], roe_factor_df[day_time],
         sales_G_q_factor_df[day_time],
         grossprofitmargin_factor_df[day_time], yield_rate_df[day_time], strategy_return_df[day_time],
         strategy_yield_rate_df[day_time]], axis=1)
    one_day_result.columns = column
    one_day_result.to_csv('D:/data/' + day_time + '.csv')

# 初始文件因子处理
first_path = 'D:/data/'
first_files = os.listdir(first_path)
for files_name in first_files:
    files_path = first_path + files_name
    dt = pd.read_csv(files_path, index_col=0, header=0).dropna()  # 直接删去存在缺失项股票
    for factor_name in dt.columns[:-3]:
        dt[factor_name] = factor_mad(dt[factor_name])
        dt[factor_name] = standardize(dt[factor_name])
    dt.to_csv('D:/redata/' + files_name)

# 模型文件处理
factor_train()
section_train_test()

"""
# 调参
model_dt = pd.read_csv('D:/model_data.csv', index_col=0, header=0)
x = model_dt.iloc[:, :-3]
y = model_dt.excess_return
X_train, X_test, y_train, y_test = train_test_split(np.array(x), np.array(y), test_size=0.9, random_state=10)
model = xgb.XGBClassifier(learning_rate=0.05, n_estimators=100)
# 需调整的参数设置
# 学习率 learning_rate = [0.05,0.Quant_code,0.15,0.2]
# 树的个数 n_estimator = [100,500,1000]
max_depth = [3, 4, 5, 6]
subsample = [0.8, 0.85, 0.9, 0.95, Quant_code]
param_grid = dict(max_depth=max_depth, subsample=subsample)
grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=10) # 十折验证
grid_result = grid_search.fit(X_train, y_train)

# 生成参数热力图
means = grid_result.cv_results_['mean_test_score']
means = means.reshape(5, 4) * 100
print(means)
means = means.tolist()
ax = sns.heatmap(means, annot=True, xticklabels=subsample, yticklabels=max_depth, fmt='.4f')
ax.set_xlabel('subsample')
ax.set_ylabel('max_depth')
ax.set_title('交叉验证集正确率')
plt.savefig('D:/调参正确率.jpg', dpi=400)
plt.show()
"""
#  调参后模型训练
train_path = 'D:/section/'
test_path = 'D:/redata/'
year = 2016
model_score_lst = []
for section_train in os.listdir(train_path):
    dataframe = pd.read_csv(train_path + section_train, index_col=0, header=0)
    x_train = dataframe.iloc[:, :-3]
    y_train = dataframe.excess_return
    X_train = np.array(x_train)
    y_train = np.array(y_train)
    model = xgb.XGBClassifier(learning_rate=0.05, n_estimators=100, max_depth=3, subsample=0.9)
    model = model.fit(X_train, y_train)
    year += 1
    for section_test in os.listdir(test_path)[(year - 2014) * 12:(year - 2013) * 12]:
        test_dataframe = pd.read_csv(test_path + section_test, index_col=0, header=0)
        test_dataframe.sort_values('excess_return', inplace=True, ascending=True)
        thirty = int(len(test_dataframe.index) / 2)
        score_excess_return = [0] * thirty + [1] * int(len(test_dataframe.index) - thirty)
        test_dataframe['excess_return_code'] = score_excess_return
        x_test_df = test_dataframe.iloc[:, :-4]
        y_test_df = test_dataframe.excess_return_code
        X_test = np.array(x_test_df)
        y_test = np.array(y_test_df)
        y_pred = model.predict(X_test)
        model_score = accuracy_score(y_test, y_pred)
        model_score_lst.append(model_score)
        test_dataframe['pred_excess_return_code'] = pd.Series(y_pred, index=test_dataframe.index)
        test_dataframe.to_csv('D:/result/' + section_test)

# 回测

result_path = 'D:/result/'
files_lst = os.listdir(result_path)
month_return = []
month_yield_return = []

#  计算每个截面期收益
for files_num in range(0, len(files_lst) - 1):
    today_files = result_path + files_lst[files_num]
    nextday_files = result_path + files_lst[files_num + 1]
    today_dataframe = pd.read_csv(today_files, index_col=0, header=0)
    nextday_dataframe = pd.read_csv(nextday_files, index_col=0, header=0)
    today_stock_lst = today_dataframe.index.tolist()
    nextday_stock_lst = nextday_dataframe.index.tolist()
    # 取既能在该截面期买入又能在下个截面期卖出的股票
    deal_stock_lst = []
    for stock_code in today_stock_lst:
        if stock_code in nextday_stock_lst:
            deal_stock_lst.append(stock_code)
    deal_today_df = today_dataframe.loc[deal_stock_lst]
    # 取该截面期交易信号为1的股票列表
    deal_stock_df = deal_today_df[deal_today_df['pred_excess_return_code'].isin([1])]
    deal_stock_lst = deal_stock_df.index.tolist()
    return_df = nextday_dataframe.loc[deal_stock_lst]  # 取下个截面期表格中交易股票的dataframe
    return_ = np.mean(return_df['strategy_return'])  # 获取该月收益率
    yield_return_ = np.mean(return_df['strategy_excess_return'])  # 获取该月超额收益率
    month_return.append(return_)
    month_yield_return.append(yield_return_)

# 获取收益率变化曲线

base_return_lst_1 = base_return_ratio(CSI300_price_standard, new_time_lst)
base_return_lst = [1]  # 基准收益率
for i in range(len(base_return_lst_1)):
    base_return_lst.append((base_return_lst_1[i] + 1) * base_return_lst[i])
return_change_lst = [1]  # 策略收益率
for i in range(len(month_return)):
    return_change_lst.append(return_change_lst[i] * (month_return[i] + 1))

# 绘制回测结果图

# 净值曲线
fig, ax = plt.subplots()
y1 = return_change_lst
y2 = base_return_lst
x = last_day_lst_str[36:]
tick_spacing = 9  # 通过修改tick_spacing的值可以修改x轴的密度
ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax.plot(x, y1, label='策略组合')
ax.plot(x, y2, label='沪深300')
ax.legend()
plt.xticks(rotation=20)
plt.title('XGBoost分类模型回测净值')
plt.savefig('D:/XGBoost分类模型回测净值.jpg', dpi=400)
plt.show()

# 计算回测年收益率，最大回撤，夏普比率，胜率。
year_return = (return_change_lst[-1] - 1) / 4
year_yield_return = (np.array(return_change_lst) - np.array(base_return_lst))[-1] / 4
maxdown = maxdrawdown(return_change_lst)
shape_ratio = ((return_change_lst[-1] - 1) / 4) / (np.std(month_return) * np.sqrt(12))
win = 0
for i in month_return:
    if i > 0:
        win += 1
win_rate = win / len(month_return)
