############## tf.function ######################
# TODO ? `tf.function`的作用是什么
import tensorflow as tf

@tf.function
def f(x, y):
    return x**2 + y

x = tf.constant([2, 3])
y = tf.constant([3, -2])

z = f(x, y)
print(z)


############# 如何保存ckpt和saved_model ################

import os 
import tensorflow as tf

# 加载训练数据
(train_i, train_l), (test_i, test_l) = tf.keras.datasets.mnist.load_data()

# 只取一部分，方便demo快速跑完
train_l = train_l[:1000]
test_l = test_l[:1000]
train_i = train_i[:1000].reshape(-1, 28*28) / 255.0
test_i = test_i[:1000].reshape(-1, 28*28) / 255.0

# 建立模型的函数
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])
    model.compile(optimizer='adam', 
                   loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])
    return model


model = create_model()
# 打印出模型的结构
model.summary()


ckpt_path = 'training_1/cp.ckpt'
ckpt_dir = os.path.dirname(ckpt_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path, save_weights_only=True, verbose=1)

#model.fit(train_i, train_l, epochs=10, validation_data=(test_i, test_l), callbacks=[cp_callback])

model = create_model()
model.load_weights(ckpt_path)

#loss, acc = model.evaluate(test_i, test_l, verbose=2)


ckpt_path = "training_2/cp-{epoch:04d}.ckpt"
ckpt_dir = os.path.dirname(ckpt_path)
# 每5个epoch
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=ckpt_path,
    verbose=1,
    save_weights_only=True,
    period=5)

model = create_model()
model.save_weights(ckpt_path.format(epoch=0))

#model.fit(train_i, train_l, epochs=50, callbacks=[cp_callback], validation_data=(test_i, test_l), verbose=0)

# 从文件夹中加载最新的ckpt，注意，这里是按照时间顺序来的，不是按照命名规则
latest = tf.train.latest_checkpoint(ckpt_dir)
print(latest)

model = create_model()
model.save('saved_model/my')

# 加载保存的`tf.keras.models.Model`模型，并打印；此时获取的model跟新建的model一样，
# 可以直接使用predict，evaluate之类的方法
loaded_model = tf.keras.models.load_model('saved_model/my')
loaded_model.summary()



###### 用keras #####################################
import os
import tempfile

import numpy as np
import tensorflow as tf

tmpdir = tempfile.mkdtemp()
print(tmpdir)

# 获取图片，然后加载图片
file = tf.keras.utils.get_file(
    'grace_hopper.jpg',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')

img = tf.keras.preprocessing.image.load_img(file, target_size=[224, 224])

x = tf.keras.preprocessing.image.img_to_array(img)
x = tf.keras.applications.mobilenet.preprocess_input(x[tf.newaxis, ...])

print(x.shape)

# 获取label与ID对应关系的map
labels_path = tf.keras.utils.get_file(
    'ImageNetLabels.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')

imagenet_labels = np.array(open(labels_path).read().splitlines())

print(imagenet_labels.shape)

# 获取一个预训练模型, 并用它来预测给定图片的分类
pretrained_model = tf.keras.applications.MobileNet()
#result_before_save = pretrained_model(x)
#decoded = imagenet_labels[np.argsort(result_before_save)[0, ::-1][:5]+1]


mobilenet_save_path = os.path.join('/home/cgh/tf_serving', 'mobilenet/1/')
# 将预训练模型保存到一个指定路径中去
#tf.saved_model.save(pretrained_model, mobilenet_save_path)

loaded = tf.saved_model.load(mobilenet_save_path)
print(list(loaded.signatures.keys()))

# 每一个saved_model都会有一个属性，signatures，这个属性是一个dict，记录了该saved_model的所有
# signatures，记录了用它来serving的时候，能接收的输入和输出？？？
infer = loaded.signatures['serving_default']
print(infer.structured_outputs)

# 那么我们该如何使用加载的模型呢？
# 这里我们需要注意的是，通过`tf.saved_model.load`加载的saved_model，返回的不是model本身
# （至少不是`tf.keras.models.Model`），想要用它来预测的话，
# 需要根据其signature
labeling = infer(tf.constant(x))
# 而这样得到是一个dict，
print(labeling)
### ???为什么infer存在一个attribute是inputs，其类型是list，不应该只有一个输入吗？？

