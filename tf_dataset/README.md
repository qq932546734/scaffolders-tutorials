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


# Example的序列化，其实就是dict的序列化，因此，多个序列化之后的bytes相加，仍然可以反序列化回去，对两个Example的结构没有要求（因为都是dict，就相当于dict相加一样，key完全一样都没问题）
new = tf.train.Example.FromString(first_bytes + second_bytes)
```
上面展示了如何将源数据转换成`tf.train.Example`，以及如何序列化example和如何反序列化。
这里我们引入`tf.data.TFRecordDataset`，一般以`.tfrec`结尾的文件，为tf record file文件。这类文件存储的就是序列化的`tf.train.Example`文件。

TFRecord文件由一系列的record组成，该文件只能串行读取。每一个record包含数据本身的bytes，数据的长度，和一个hash值用于检查数据的完整性的。

```python
# 如何从tf recordfile文件构造`tf.data.Dataset`
ds = tf.data.TFRecordDataset(filename=["first.tfrec", 'second.tfrec'])
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
# 构造映射关系，为什么需要这个呢？不是可以直接用Example.FromString()反序列化吗？这是因为Dataset是一个graph-execution的过程，即在搭建计算图的时候，数据还没有生成。因为数据还没有被消费，我们无法知道其结构，即key、datatype、shape之类的参数，因此，我们需要通过feature_description来提供这些参数。
feature_description = {
    'first_feature': tf.io.FixedLenFeature([], tf.int64, default_value=0)
}

def parse_example(example_protobuff):
    return tf.io.parse_single_example(example_protobuff, feature_description)

rec_dataset.map(parse_example)
```




### `tf.Tensor` & `tf.EagerTensor`

> 这两者的区别？