from sklearn.model_selection  import train_test_split
train_test_split(X, Y, test_size =0.3, random_state=1212)
# X,Y arrays：可以是lists,np.array,pd.DataFrame
# test_size：float类型，必须是[0,1]之间 test_size =0.3, train : valid = 7 : 3
# accuracy : 準確性
DecisionTree : 決策樹(
# Entropy: 不確定性的量度。Entropy=0為最穩定的狀態，Entropy=1為最混亂無序的狀態。
#資訊增益(Information Gain): 不同狀態Entropy的差值，資訊增益越多越好。
)
RandomForest : 隨機森林(
    
)
SVM(support vector machine) 主要用途：
classification(分类)、regression(回归)、outliers detection(异常检测)

normalization : 標準化
initialization : 初始化
regularization : 正規化
tf.reduce_mean()
tf.truncated_normal(預設平均為0，標準差為stddev的亂數)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_predict))
#labels = 實際值，logits = 預測值
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#tf.train.GradientDescentOptimizer(learning_rate)
correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y, 1))
#tf.argmax(vector, 1)：返回的是vector中的最大值的索引号，如果vector是一个向量，那就返回一个值，如果是一个矩阵，那就返回一个向量
#tf.equal(A, B)是对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True，反正返回False，返回的值的矩阵维度和A是一样的
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#tf.cast：用于改变某个张量的数据类型

tf.nn.conv2d(
    tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    第一個引數input：指需要做卷積的輸入影象，它要求是一個Tensor，具有[batch, in_height, in_width, in_channels]這樣的shape
    第二個引數filter：相當於CNN中的卷積核，它要求是一個Tensor，具有[filter_height, filter_width, in_channels, out_channels]這樣的shape
    第三個引數strides：卷積時在影象每一維的步長，這是一個一維的向量，長度4；
    第四個引數padding：string型別的量，只能是"SAME","VALID"其中之一，這個值決定了不同的卷積方式；
    第五個引數use_cudnn_on_gpu:bool型別，是否使用cudnn加速，預設為true
)
cell = tf.contrib.rnn.BasicRNNCell(num_hidden(隱藏神經元個數、memory))
cell.zero_state(batch_size, dtype=tf.float32)
#初始化cell狀態
MS COCO dataset 資料集主要解決3個問題：目標檢測，目標之間的上下文關係，目標的2維上的精確定位。



sftp
10.124.131.87
10.124.131.81