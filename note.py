MAE : 將兩個陣列相減後, 取絕對值(abs), 再將整個陣列加總成一個數字(sum), 最後除以y的長度(len), 因此稱為"平均絕對誤差"
MAE = sum(abs(y - yp)) / len(y)

MSE : 均方誤差(Mean square error，MSE)
MSE = sum((y-yp)**2)/len(y)

機器學習是甚麼?
白話文 : 讓機器從資料中找尋規律與趨勢而不需要給定特殊規則
數學 : 給定目標函數與訓練資料，學習出能讓目標函數最佳的模型參數

EDA = Exploratory Data Analysis = '探索式資料分析'
數據分析流程 = (資料收集 - 數據清理 - 特徵萃取 - 資料視覺化 - 建立模型 - 驗證模型 - 決策應用)

資料原來是字串/類別的話，如果要做進⼀步的分析時（如訓練模型），⼀般需要轉為數值的資料類型，轉換的⽅式通常有兩種
• Label encoding：使⽤時機通常是該資料的不同類別是有序的，例如該資料是年齡分組，類別有⼩孩、年輕⼈、老⼈，表⽰為 0, 1, 2 是合理的，因為年齡上老⼈ > 年輕⼈、年輕⼈ > ⼩孩
• One Hot encoding：使⽤時機通常是該資料的不同類別是無序的，例如國家

資料特徵
    數值型特徵：最容易易轉成特徵，但需要注意很多細節
    類別型特徵：通常⼀一種類別對應⼀一種分數，問題在如何對應
    時間型特徵：特殊之處在於有週期性