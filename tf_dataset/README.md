### `tf.train.Example`

Example是对一条训练数据的抽象。一条训练数据可以有多个feature，如大小，长度；每个feature都有值，可以是int，float，甚至是string。
> 如果一个feature的值是二维的数值，该如何用`tf.train.Int64List(value=value)`来表示呢？这里value的值只能是list of int类型，不能是二维的数组。

因此，Example类似于一个dict，key是feature的名字，value是feature的值。Example采用的是Google的Protocol Buffer，因此在实例化的时候，不接受positional变量。

![tf_example](https://image-1300946842.cos.ap-beijing.myqcloud.com/tf_example.jpg)

```python
tokens = ['we', 'are', 'the', 'best']
example = tf.train.Example(
    features=tf.train.Features(
        feature={
            'input_ids': tf.train.Feature(
                int64_list=tf.train.Int64List(value=[12, 34, 29, 87, 43])
            ),
            'input_tokens': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[i.encode() for i in tokens])
            )
        }
    )
)
in_bytes = example.SerializeToString()

# 将其写入到文件
writer = tf.data.experimental.TFRecordWriter('test.tfrec')
writer.write(in_bytes)
# 将example写入到文件中，python版
with tf.io.TFRecordWriter('test.tfrec') as writer:
    writer.write(in_bytes)

# 被序列化的example可以转化回来
exp = tf.train.Example.FromString(in_bytes)

# 读取example的某一个feature
input_ids = exp.features.feature['input_ids'].int64_list.value[:]

# Example的序列化，其实就是dict的序列化，因此，多个序列化之后的bytes相加，仍然可以
# 反序列化回去，对两个Example的结构没有要求（因为都是dict，就相当于dict相加一样，
# key完全一样都没问题）
new = tf.train.Example.FromString(first_bytes + second_bytes)

# 如何知道一个example对象的keys, 记得要加上list，不然得到的KeyView
keys = list(new.features.feature.keys())
```
上面展示了如何将源数据转换成`tf.train.Example`，以及如何序列化example和如何反序列化。
这里我们引入`tf.data.TFRecordDataset`，一般以`.tfrec`结尾的文件，为tf record file文件。这类文件存储的就是序列化的`tf.train.Example`文件。

TFRecord文件由一系列的record组成，该文件只能串行读取。每一个record包含数据本身的bytes，数据的长度，和一个hash值用于检查数据的完整性的。

```python
# 如何从tf recordfile文件构造`tf.data.Dataset`
ds = tf.data.TFRecordDataset(filenames=["first.tfrec", 'second.tfrec'])
```

> 关于Example，tf record file，RecordFileDataset的关系：
>
> 构造了Example之后，可以序列化成bytes；bytes可以写入到tf recordfile文件中。
>
> TFRecord file是一种存储格式，并不是一定要跟`tf.train.Example`组合使用，后者只是提供了一种将dict序列化成bytes的便捷方式。
>
> 在解析TFRecord file的时候，如果每一个record刚好是Example序列化的数据，那么可以用`tf.io.parse_single_example()`来解析
>
> 需要注意的是，`tf.data.RecordFileDataset`在获取到dataset之后，其实每一个record都是bytes，我们需要对每一个record进行反序列化。

> Example.FromString()和Example.ParseFromString()有什么区别？

> 既然TFRecord文件只能串行读取，那怎么能保证效率呢？如我需要对一个Dataset进行shuffle，然后读取

> TFRecord文件能不跟`tf.train.Example`连用吗？如果可以的话，怎么直接写入和读取呢？

> `tf.io.parse_example()`和`tf.io.parse_single_example()`的区别？
>
> 二者的返回值都是dict，key是特征的名称，value是`tf.Tensor`。他们最常用的场景都是在TFRecordDataset中，解析读出来的二进制。Dataset可以进行batch，如果先对TFRecordDataset进行batch的话，那么得到的`tf.Tensor`其shape为`(batch_size,)`类型为`tf.string`，此时我们是不能用`parse_single_example`来parse的，只能用`parse_example`。其他大部分情况没有区别。

### `tf.data.Dataset`

> `tf.data.Dataset`和`tf.train.Example`有什么关联？

#### 如何构造Dataset

`from_tensor()`, `from_tensor_slices()`, `from_generator()`

```python
# 只读取前n条数据
for data in dataset.take(10):
	print(repr(data))
    
# 从TFRecord文件中构造一个dataset（下面的操作要求必须是Example序列化生成的record文件）
rec_dataset = tf.data.RecordFileDataset(filename=['first.tfrec'])
# 构造映射关系，为什么需要这个呢？不是可以直接用Example.FromString()反序列化吗？
# 这是因为Dataset是一个graph-execution的过程，即在搭建计算图的时候，数据还没有生成。
# 因为数据还没有被消费，我们无法知道其结构，即key、datatype、shape之类的参数。
# 因此，我们需要通过feature_description来提供这些参数。
feature_description = {
    'first_feature': tf.io.FixedLenFeature([], tf.int64, default_value=0)
    'raw': tf.io.FixedLenFeature([], tf.string)
}

def parse_example(example_protobuff):
    return tf.io.parse_single_example(example_protobuff, feature_description)

rec_dataset.map(parse_example)
```




### `tf.Tensor` & `tf.EagerTensor`

> 这两者的区别？前者只能在图计算中使用，后者可以通过其`tf.EagerTensor.numpy()`方法，转换成numpy中的数据。

> 如何enable eager execution？

### 一个例子

读两张图片，并写入到TFRecord文件中

```python
# 下载图片
urls = [
"https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg", 
'https://storage.googleapis.com/download.tensorflow.org/example_images/194pxNew_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg'
]
cat  = tf.keras.utils.get_file('cat.jpg', urls[0])
bridge = tf.keras.utils.get_file('bridge.jpg',urls[1])
# 返回的是文件的路径

# 给每一个文件一个唯一的编号，方便区分
image_labels = {cat:0, bridge:1}

# 二进制读取文件
image_string = open(cat, 'rb').read()
# 获取图片大小数据
image_shape = tf.image.decode_jpeg(image_string).shape
# 构造feature的dict
feature = {}
feature['height'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[image_shape[0]]))
feature['width'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[image_shape[1]]))
feature['depth'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[image_shape[2]]))
feature['label'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[image_labels[cat]]))
feature['raw'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string]))
# 实例化Example
cat_example = tf.train.Example(features=tf.Features(feature=feature))

# 写入到文件
with tf.io.TFRecordWriter('test.tfrec') as writer:
    for example in examples:
    	writer.write(example.SerializeToString())
```



> 如何从TFRecord文件中读取数据呢？我能想到的就是通过文件实例化`tf.data.TFRecordDataset`，此时我们得到的dataset，每一个element应该是一个`tf.Tensor`（注意，不是eager tensor），dtype是string。

> `tf.train.Example.FromString(bytes_string)`和`tf.io.parse_single_example(bytes_string, feature_desc)`的区别？
>
> 前者只能接受类型为`bytes`的参数，后者既可以接受python中的`bytes`，也可以接受类型为string的`tf.Tensor`。在解析`tf.data.TFRecordData`中的每个元素的时候，因为每一个element都是dtype为string的`tf.Tensor`，所以无法使用`FromString()`方法来解析。某些情况下我们可以用`FromString()`方法，但是需要先用`tensorflow.python.framework.ops.EagerTensor`的`numpy()`方法，将其转换成`bytes`。
>
> 如果我想在`TFRecordDataset`的map过程中（即读取了record文件之后，parse的过程）使用`FromString()`方法来解析，行不行呢？答案是不行的。因为在map中，转换函数（decode函数）接收的参数是一个example，但类型是`tensorflow.python.framework.ops.Tensor`，而不是`EagerTensor`，所以我们没有办法通过`numpy()`将example转换成bytes，因此也就无法使用`FromString()`函数了。
>
> 另外需要注意的是，`FromString()`得到的是Example对象，而`parse_single_example()`得到的是python中的dict对象。因此，在循环遍历解码好了的Dataset的时候，每一个example其实就是一个dict。
>
> `parse_single_example()`的第二个参数为feature_describtion，这个可以是完整的map，也可以只写我们想要得到的features。比如原来的每个example有10几个特征，但是我们可能只需要height, width这两个特征，这时feature_describtion只需要写这两个特征的map就行。

### `tf.data`API的性能

####  构造database

* `from_tensor()` & `from_tensor_slice()`

从内存中构造dataset。前者是将输入当做一个example，后者是将输入中的每一个element当做一个example。如果参数是dict，那么获得的Dataset的element也是dict。

* `from_generator()`

从python的generator中生成dataset

#### transformation

* `apply(transformation_fn)`

参数是转换函数，其接受的参数是`tf.data.Dataset`，返回的也是`tf.data.Dataset`。因此，在转换函数中，我们可以chain多个transformation。

* `as_numpy_iterator()`

将dataset转换成numpy的interator。可能在实际过程中很少使用，在debug过程中，或是tutorial中，方便我们了解dataset的具体内容。必须是处于eager execution的状态，不然会报错的。

* `filter(predicate)`

根据predicate（函数）进行过滤，predicate接受的参数是每一个element

* `flat_map(map_func)`

先对dataset每一个element执行map_func，然后将结果flatten（类似于`list.extend()`)

#### 数据准备

* `batch(batch_size, drop_remainder=False)`

对dataset取batch，之前的element是example，使用之后的element是batch。`drop_remainder`指的是如果对于element个数不是batch_size的整数倍的时候，多余的element是否抛弃。

> 这里有一个之前搞错的地方：我之前认为对一个dataset取batch的话，是得到多个dict，每个dict是之前的一个element。但其实得到的一个dict，这个dict中的每一个key的值变成了[batch_size, 原来的维度]。举个例子，batch之前，一个element是`{'first': <tf.Tensor: shape=(20,)}`,`batch(3)`之后，得到的dataset的一个element的是`{'first': <tf.Tensor: shape=(3,20)}`，而不是三个dict。

* `concatenate(dataset)`

将两个dataset合并

* `interleave(map_func, cycle_length=AUTOTUNE, block_length)`

我们在实例化一个dataset的时候，数据并没有全部都准备好放在内存中，只有我们在消费这个dataset的时候，数据才会进行生成。像`prefetch()`，`interleave()`之类的操作，就是帮助减少数据准备的等待时间。`interleave`的作用就是能同时处理多个elements。TODO：待续

* `prefetch(buffer_size)`

提前准备好`buffer_size`大小的Example，会多占用一点空间用来存储提前准备好的Example，但是能避免在处理example/batch的时候，需要等待example的production。官方文档建议一般的dataset都应该最后call一下prefetch这个method。

* `shuffle()`

具体是怎么操作的呢？