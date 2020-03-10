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
.quantile(0.25,0.5,0.75,0.99)
    分位數
    np.percentile()與分位數類似
.describe()  
    對數據中每一列數進行統計分析
    一列數據全是'number'
        count：一列的元素個數；
        意思是：一列數據的預期；
        std：一列數據的均方差；（方差的算術平方根，反映一個數據集的離散程度：變量，數據間的差異偏差，數據集中數據的離散程度不斷；越小，數據間的大小差異越 小，數據集中的數據離散程度越低）
        min：一列數據中的預設；
        max：一列數中的高度；
        25％：一列數據中，前25％的數據的先前；
        50％：一列數據中，前50％的數據的先前；
        75％：一列數據中，前75％的數據的先前；
    一列數據：'類別'，'類別' +'數字'
        count：一列數據的元素個數；
        unique：一列數據中元素的種類；
        top：一列數據中出現頻率最高的元素；
        freq：一列數據中出現頻率最高的元素的個數；
    一列數據：object（如時間序列）
        第一：開始時間；
        上一篇：結束時間；
.replace({365243: np.nan}, inplace = True)
    {?: np.nan} 單值用np.nan替換
np.log1p(x)
    ln(x)
.fillna(x,method = 'backfill'，'bfill'，'pad'，'ffill'，'None')
    method預設'None'
    將nan替換為x
scipy.stats.mode()
    尋找出現次數最多的
.clip(800,2500)
    將數值壓在'800'~'2500'之間，小於'800'為'800'，大於'2500'為'2500'。
mode()
    眾數，回傳出現最多的項目(9000)及次數(6385)。
    ModeResult(mode=array([9000.]), count=array([6385]))
defaultdict()
    相較於dict()，如無key會回傳空值而不是Error
    defaultdict(['a']) = []
    dict(['a']) = Error
