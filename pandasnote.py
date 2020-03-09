row數 = data.shape[0]
column數 = data.shape[1]
列出所有欄位 = data.columns

讀檔
    txt:
        with open(‘example.txt’, ‘r’) as f:
            data = f.readlines()
        print(data)
    json:
        import json
        with open(‘example.json’, ‘r’) as f:
            data = json.load(f)
        print(data)
    矩陣檔 (mat):
        import scipy.io as sio
        data = sio.loadmat(‘example.mat’)
    圖像檔 (PNG / JPG …):
        Import cv2
        image = cv2.imread(...) # 注意 cv2 會以 BGR 讀入
        image = cv2.cvtcolor(image, cv2.COLOR_BGR2RGB)
        from PIL import Image
        image = Image.read(...)
        import skimage.io as skio
        image = skio.imread(...)
    Python npy :
        import numpy as np
        rr = np.load(example.npy)
    Pickle (pkl):
        import pickle
        with open(‘example.pkl’, ‘rb’) as f:
            arr = pickle.load(f)

查看檔案類型
    type()	    返回引數的資料型別
    dtype	    返回陣列中元素的資料型別
    astype()	對資料型別進行轉換


假設你想知道如果利用 pandas 計算上述資料中，每個 weekday 的平均 visitor 數量
visitors_1.groupby(by="weekday")['visitor'].mean()

pd.Series.nunique(dropna=True)
    dropna 默认参数设置为True，因此在计算唯一值时排除了NULL值。
    统计“Team”列中不同值的个数，不包括null值，類似取set計數
    unique_value = data["Team"].nunique()
np.unique(b,return_index=True)
    對於一維數組或者列表，unique函數去除其中重複的元素，並按元素由大到小返回一個新的無元素重複的元組或者列表
    類似取set做大到小排序。
    return_index=True表示返回新列表元素在舊列表中的位置，並以列表形式儲存在s中。
one hot enconding
    pd.get_dummies()  
pd.concat([df1,df2,df3]),預設axis=0，在0軸上合併。  
.reset_index()
    DataFrame取index值
.aggregate('COUNT', 'MAX', 'MIN', 'SUM', 'AVG') 
    functions (COUNT, MAX, MIN, SUM, AVG)