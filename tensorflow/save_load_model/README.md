
# tensorflow: 保存和加载模型, 参数；以及使用预训练参数方法
## 1. 保存与加载模型参数


```python
import tensorflow as tf
```

tensorflow的保存和加载是通过tf.train.Saver类实现的, 她的构造函数是
```python
def __init__(self,
               var_list=None,
               reshape=False,
               sharded=False,
               max_to_keep=5,
               keep_checkpoint_every_n_hours=10000.0,
               name=None,
               restore_sequentially=False,
               saver_def=None,
               builder=None,
               defer_build=False,
               allow_empty=False,
               write_version=saver_pb2.SaverDef.V2,
               pad_step_number=False,
               save_relative_paths=False):
```
其中
- var_list: 要保存/恢复的变量列表，or dict of names to variables. 如果为空，则默认所有的变量
- reshape: 当shape不一样时是否允许恢复参数
- sharded: if True, instructs the saver to shard checkpoints per device.
- max_to_keep: 最多保存的checkpoints. Defaults to 5. checkpoints的区分在save时传递的**global_step**参数，用来表示第几次迭代。
- keep_checkpoint_every_n_hours: How often to keep checkpoints. 每几个hour保留一个checkpoint
- .....

**保存：saver.save(session, saved_path, global_step=None)**

**恢复：saver.restore(session, saved_path)**
保存的文件有四种：
1. checkpoint，保存最近保存的模型的文件名，因此我们能够知道最近的模型名，可以通过调用tf.train.latest_checkpoint(dir)获知
2. .meta 图的结构，变量等信息
3. .data 参数值
4. .index 索引文件

使用示例：


```python
tf.reset_default_graph()
v1=tf.get_variable('v1', shape=[6], dtype=tf.float32, initializer=tf.truncated_normal_initializer())
v2=tf.get_variable('v2', shape=[6], dtype=tf.float32, initializer=tf.random_normal_initializer())

saver=tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('v1', sess.run(v1))
    print('v2', sess.run(v2))
    saver.save(sess, 'save/model')
```

    v1 [ 1.88005733 -0.99327284 -1.19482517 -0.46593472 -0.1329312  -1.63472843]
    v2 [-1.03660548 -1.61874151 -1.5886656   0.45553902 -1.24812245 -0.90952981]
    


```python
tf.reset_default_graph()
v1=tf.get_variable('v1', shape=[6], dtype=tf.float32, initializer=tf.truncated_normal_initializer())
v2=tf.get_variable('v2', shape=[6], dtype=tf.float32, initializer=tf.random_normal_initializer())

saver=tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('save/'))
    print('v1', sess.run(v1))
    print('v2', sess.run(v2))
    
```

    INFO:tensorflow:Restoring parameters from save/model
    

    INFO:tensorflow:Restoring parameters from save/model
    

    v1 [ 1.88005733 -0.99327284 -1.19482517 -0.46593472 -0.1329312  -1.63472843]
    v2 [-1.03660548 -1.61874151 -1.5886656   0.45553902 -1.24812245 -0.90952981]
    

至此，我们可以简单使用。但是当我们需要加载模型的时候呢，可以使用tf.train.import_meta_graph(), tensor获取通过tf.get_default_graph().get_tensor_by_name('')  （不过一般我们都会有原来图的代码，所以一般不会用到。）


```python
tf.reset_default_graph()
tf.train.import_meta_graph('save/model.meta')
saver=tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, 'save/model')
    print(sess.run(tf.get_default_graph().get_tensor_by_name("v1:0")))
```

    INFO:tensorflow:Restoring parameters from save/model
    [ 1.88005733 -0.99327284 -1.19482517 -0.46593472 -0.1329312  -1.63472843]
    

介绍完基本使用，接下来如果我们的模型只有一部分是要加载参数时


```python
tf.reset_default_graph()
v1=tf.get_variable('v1', shape=[6], dtype=tf.float32, initializer=tf.truncated_normal_initializer())
v2=tf.get_variable('v2', shape=[6], dtype=tf.float32, initializer=tf.random_normal_initializer())
v2=tf.get_variable('v3', shape=[6], dtype=tf.float32, initializer=tf.random_normal_initializer())

saver=tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('save/'))
    print('v1', sess.run(v1))
    print('v2', sess.run(v2))
```

这个时候会报NotFoundError (see above for traceback): Key v3 not found in checkpoint
解决办法就是saver初始化要加上要保存的参数列表：


```python
tf.reset_default_graph()
v1=tf.get_variable('v1', shape=[6], dtype=tf.float32, initializer=tf.truncated_normal_initializer())
v2=tf.get_variable('v2', shape=[6], dtype=tf.float32, initializer=tf.random_normal_initializer())
v2=tf.get_variable('v3', shape=[6], dtype=tf.float32, initializer=tf.random_normal_initializer())

saver=tf.train.Saver([v1, v2])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('v1', sess.run(v1))
    print('v2', sess.run(v2))
    saver.save(sess, 'save/model')
    saver.restore(sess, tf.train.latest_checkpoint('save/'))
    print('v1', sess.run(v1))
    print('v2', sess.run(v2))
```

    v1 [ 0.39415926  0.24765804  1.26394165  0.62132704  1.0527215   1.55297732]
    v2 [ 0.56525308  1.07240736 -0.15881526 -1.1062392  -0.76180184  1.05873036]
    INFO:tensorflow:Restoring parameters from save/model
    v1 [ 0.39415926  0.24765804  1.26394165  0.62132704  1.0527215   1.55297732]
    v2 [ 0.56525308  1.07240736 -0.15881526 -1.1062392  -0.76180184  1.05873036]
    

现在，当我们想要预训练的时候，就可以将预训练的相关参数放在列表中，然后保存。
