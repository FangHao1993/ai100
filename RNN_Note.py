normalization : 標準化
initialization : 初始化
regularization : 正規化

cell = tf.contrib.rnn.BasicRNNCell(num_hidden(隱藏神經元個數、memory))
cell.zero_state(batch_size, dtype=tf.float32)
#初始化cell狀態

tf.nn.dynamic_rnn(
    cell,
    inputs,
    sequence_length=None,
    initial_state=None,
    dtype=None,
    parallel_iterations=None,
    swap_memory=False,
    time_major=False,
    scope=None
)
cell 必须是RNNCell的一个instance，例如BasicLSTMCell
inputs RNN的输入，如果time_major == False, 那么shape为( batch size, max time, input size); 如果time_major == True, 那么shape 为(max time, batch size, input size)
sequence_length 是可选的，是一个int32/int64 [batch_size]的向量，主要用于控制batch的长度，如果超过那么就将状态设置为0
initial_state，给RNN的初始状态 [公式] ，例如cell.state_size是一个int，那么就必须是一个shape = (batch size, cell.state_size)的Tensor
time_major, 表示inputs和outputs的shape格式，如果True，那么就是(max time, batch size, input size); 否则就是(batch size, max time, depth) 。这里默认为False，一般不需要改动
Return:(outputs, state)

outputs 是每个时刻的输出，shape为( batch size, max time, output size)
state 表示最后时刻的输出， (batch size, state size) 表示batch中每条数据最后的状态输出 [公式]
outputs, final_state = tf.nn.dynamic_rnn(cell, x, initial_state=init_state, time_major=False)

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

