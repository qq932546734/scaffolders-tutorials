### 连接s3
s3的连接参数有多种方式传参，官方推荐的方式是写入到指定路径下的配置文件。也可以直接通过在代码里面传参。
```python
import boto3

params = {
    "service_name": "s3",
    "aws_access_key_id": "",
    "aws_secret_access_key": ""
}

client = boto3.client(**params)

resource = boto3.resource(**params)
```
resource和client两个都是可以操作的变量。
如打印所有bucket的名称
```python
for bucket in resource.buckets.all():
    print(bucket.name)

# 或者在client对象上操作
for item in client.list_buckets()['Buckets']:
    print(item['Name'])
```


### list指定前缀的对象
在s3中没有文件夹的概念，每一个对象都以一个唯一的key与实际的data进行关联。虽然不能通过文件夹访问，但是我们可以根据key的prefix来进行访问对象。但`list_objects_v2`每次只能列出不超过1000个对象，所以，当存在较多文件的时候，我们需要引入paginator
```python
import boto3

response = client.list_objects_v2(Bucket="Bucketname", Prefix="2020/05/08")
contents = response["Content"]
for item in contents:
    print(item)

paginator = client.get_paginator('list_objects')
operation_parameters = {'Bucket': 'my-bucket',
                        'Prefix': 'my-prefix'}
page_iterator = paginator.paginate(**operation_parameters)
for page in page_iterator:
    print(page['Contents'])
```