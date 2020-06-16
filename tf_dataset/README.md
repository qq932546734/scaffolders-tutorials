### `tf.Example`

Example是对一条训练数据的抽象。一条训练数据可以有多个feature，如大小，长度；每个feature都有值，可以是int，float，甚至是string。
> 如果一个feature的值是二维的数值，该如何用`tf.train.Int64List(value=value)`来表示呢？

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

```