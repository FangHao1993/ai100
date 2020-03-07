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

假設你想知道如果利用 pandas 計算上述資料中，每個 weekday 的平均 visitor 數量
visitors_1.groupby(by="weekday")['visitor'].mean()