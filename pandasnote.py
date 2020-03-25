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

.loc[[row],[columns]]

pd.cut(x,bins,right=True,labels=None,retbins=False,precision=3,include_lowest=False)
    X：進行劃分的一維數組
    bins : 1,整數---將x劃分為多少個等間距的區間
    right : 是否包含右端點 
    labels : 是否用標記來代替返回的bins
    retbins: 是否返回間距bins
    precision: 精度
    include_lowest:是否包含左端點
    等寬劃分 (對應 pandas 中的cut) 
pd.qcut()
    等頻劃分 (對應 pandas 中的 qcut) 可以依實際需求來自己定義離散化的方式

np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
    在指定的間隔內返回均勻間隔的數字。
    返回num均勻分佈的樣本，在[start, stop]。
    這個區間的端點可以任意的被排除在外。

np.inf
    numpy用IEEE-754格式，一般numpy.inf是float64格式，能表示最大的数是±(1−2−53)∗21024=±1.79769×10308。

np.corrcoef(x,y)
    皮爾遜相關係數的變化範圍為-1到1。 係數的值為1意味著X 和 Y可以很好的由直線方程式來描述，所有的數據點都很好的落在一條 直線上，且 Y 隨著 X 的增加而增加。係數的值為−1意味著所有的數據點都落在直線上，且 Y 隨著 X 的增加而減少。係數的值為0意味著兩個變數之間沒有線性關係。
    corrcoef是對兩個列向量，或者一個矩陣的每列進行的，用的是pearson相關
    np.corr()
        corr可以對兩個矩陣的每列進行，也可以對一個矩陣的每列進行，相關的類型可以是pearson或者Kendall或者Spearman
        例如corr（X，Y，'type'，'Pearson'）
np.random
    numpy.random.randn(row,col)
        （d0，d1，…，dn）是從標準正態分佈中返回一個或多個樣本值。
    numpy.random.rand()
        （d0，d1，…，dn）的隨機樣本放置[0，1）中。
    
    sample(a,n)

        從序列a中隨機抽取n個元素，連接n個元素生以列表形式返回。
    np.random.choice([0,1,2], size=3)
        隨機回傳[0,1,2]到list，list長度為3

ax.annotate("r = {:.2f}".format(r),
            xy=(.2, .8), xycoords=ax.transAxes,
            size = 20)
    xy=(.2, .8) => 標註點在 xy 的座標在 (0.2, 0.8) 的位置
    xycoords=ax.transAxes => 標注點的定位方式，transAxes 是以 圖表的座標定位

app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)
    調整欄位數, 移除出現在 training data 而沒有出現 testing data 中的欄位
    只留下交集的columns

plt.scatter(x,y)
    散點圖


for index,key  in enumerate(keys)
    做index

def corr_func(x, y, **kwargs):
    **kwargs = dictionary變等號
    {a:1,b:2,c:3} => a=1,b=2,c=3
