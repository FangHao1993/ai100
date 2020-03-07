MAE : 將兩個陣列相減後, 取絕對值(abs), 再將整個陣列加總成一個數字(sum), 最後除以y的長度(len), 因此稱為"平均絕對誤差"
MAE = sum(abs(y - yp)) / len(y)

MSE : 均方誤差(Mean square error，MSE)
MSE = sum((y-yp)**2)/len(y)

機器學習是甚麼?
白話文 : 讓機器從資料中找尋規律與趨勢而不需要給定特殊規則
數學 : 給定目標函數與訓練資料，學習出能讓目標函數最佳的模型參數

EDA = Exploratory Data Analysis = '探索式資料分析'
數據分析流程 = (資料收集 - 數據清理 - 特徵萃取 - 資料視覺化 - 建立模型 - 驗證模型 - 決策應用)