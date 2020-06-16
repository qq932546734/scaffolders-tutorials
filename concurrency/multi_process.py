from multiprocessing import Pool


def fn(first_var):

    return first_var

vars = [1, 2, 3, 4]

# 设置进程池
p = Pool(10)
# 第一个参数是要执行的函数，第二个是参数的集合
p.map(fn, vars)

# NOTE：在执行以上之前，先运行一次单独的函数，检查是否会出错。因为如果函数报错的话，
# 在多进程执行的时候，是看不到报错的，进程会进入假死状态。