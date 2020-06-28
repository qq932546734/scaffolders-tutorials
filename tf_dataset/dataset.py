import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np


def create_float_feat(values):
    """生成float类型的feature"""
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))

# 在parse example的时候使用
desc = { 
    'hight': tf.io.FixedLenFeature([10], tf.float32),
    'width': tf.io.FixedLenFeature([5], tf.float32)
}


def create_example():
    """随机生成一个example"""
    hight = np.random.random(10)
    width = np.random.random(5)
    feature = {}
    feature['hight'] = create_float_feat(hight)
    feature['width'] = create_float_feat(width)
    return tf.train.Example(features=tf.train.Features(feature=feature))


def gen(num=100):
    """生成`num`个example，并将其写入到fake.tfrec这个文件中去"""
    with tf.io.TFRecordWriter('fake.tfrec') as writer:
        for _ in range(100):
            ex = create_example()
            content = ex.SerializeToString()
            writer.write(content)


def decode(example_proto):
    # 这里不能用`tf.train.Example.FromString()`，一是因为
    # example_proto的类型是Tensor, 不是EagerTensor，无法转换
    # 成bytes；而是因为我们希望最后在循环Dataset的时候，获取到的
    # element是一个dict，而不是`tf.train.Example`对象。
    return tf.io.parse_single_example(example_proto, desc)


def load_dataset():
    a = tf.data.TFRecordDataset(filenames=['fake.tfrec'])
    a = a.map(decode)
    return a


def consume():
    ds = load_dataset()
    ds = ds.batch(2)
    for element in ds.take(2):
        print(element)
        print('####################')
        for key in element:
            print(key)


another_ds = tf.data.TFRecordDataset(filenames=['fake.tfrec'])
for ele in another_ds:
    b = ele.numpy()
    # 这样是没问题的，得到的Example对象
    example = tf.train.Example.FromString(b)
    # 这样也是没问题的
    example2 = tf.io.parse_single_example(b, desc)
    # 这样不转换成bytes也是可以的
    example3 = tf.io.parse_single_example(ele, desc)


# 一些example，每个example有两个特征，first和second
features = {}
features['first'] = [1,2,3,4]
features['second'] = [5,6,7,8]

dict_ds = tf.data.Dataset.from_tensor_slices(features)
# 此时得到的dataset每一个element就是一个dict