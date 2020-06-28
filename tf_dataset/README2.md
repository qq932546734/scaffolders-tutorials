### checkpoint和saved_model

checkpoint一般只存储参数，不会保存网络结果（计算图）。它一般有三个文件，`[name].ckpt.data-00000-of-00001`, `[name].ckpt.index`, `checkpoint`，index文件记录各个参数存储在哪个文件中；data文件存储着二进制的参数，如果是在多台机器上训练的话，会有多个data文件。

`saved_model`则是包括计算图和checkpoint（variables文件夹相当于ckpt）。

### saved_model

`tf.keras.models.Model.save_weights`该方法将模型参数保存为checkpoint文件；加载的时候，因为checkpoint不保存计算结构，因此我们需要先获取model对象，然后load_weights；`tf.keras.models.Model.save`该方法将模型保存为saved_model，加载的时候，不需要先实例化model，可以直接用`tf.keras.models.load_modal`加载模型。