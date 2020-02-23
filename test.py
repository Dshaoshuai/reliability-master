from edge_cloud_env.module import *
import copy
import numpy as np
import tensorflow as tf
from params import *

def fn_0():
    num_cnt=9
    # a=fn_1(num_cnt,2)
    entry_md = module(0, None, None, None, None, True, None, 0)
    md_1=module(1, None, None, None, None, True, None, 0)
    md_2=module(2, None, None, None, None, True, None, 0)
    lis=[]
    lis.extend([entry_md,md_2,md_1])
    lis_1=lis
    lis_1[0].idx=3
    print(entry_md.idx)
    # lis_1=sorted(lis,key=lambda s:s.idx)
    # print([md.idx for md in lis])
    # print(entry_md.idx)
    # lis=[]
    # lis.append(entry_md)
    # fn_2(lis)
    # print(lis[0].idx,entry_md.idx)

def fn_2(lis):
    lis[0].idx+=1


def fn_1(num_cnt,sig):
    if sig>0:
        num_cnt+=1
    else:
        num_cnt-=1
    return [1]

def fn_3(sign):
    if sign>2:
        return 1,2
    else:
        return None,None
    # return 1,2

def sort_custom(md_0,md_1):
    if md_0>md_1:
        return 1
    elif md_0<md_1:
        return -1
    else:
        return 0

def fn_4():
    tf.set_random_seed(1)

    x_data=np.random.rand(100).astype(np.float32)
    y_data=x_data*0.1+0.3

    weights=tf.Variable(tf.random_uniform([1],-1,1))
    biases=tf.Variable(tf.zeros([1]))

    y=weights*x_data+biases
    loss=tf.reduce_mean(tf.square(y-y_data))

    optimizer=tf.train.GradientDescentOptimizer(0.5)
    train=optimizer.minimize(loss)

    init=tf.global_variables_initializer()
    sess=tf.Session()
    sess.run(init)
    for step in range(100):
        sess.run(train)
        if step%20==0:
            print(step,sess.run(weights),sess.run(biases))

def fn_5():
    tot=[]
    a=[1,2,3]
    b=[4,5,6]
    c=[1,2,4,5]
    tot.extend([a,b])
    print(tot)
    print(c in tot)

def fn_6():
    args.arrival_mode='poisson_arrival'
    print(args.arrival_mode)

def fn_7():
    md_1 = module(1, None, None, None, None, True, None, 0)
    print(type(md_1)==module)
    # a=np.zeros((2,3),dtype=module)
    # a[0,1]=md_1
    # print(a.tolist())
    # a=[md_1,2]
    # b=copy.deepcopy(a)
    # b.clear()
    # print(a[0],b)

if __name__=="__main__":
    # fn_0()
    # entry_md = module(0, None, None, None, None, True, None, 0)
    # lis=[]
    # lis.append(entry_md)
    # entry_md.layer=1
    # md_1=module(1,None, None, None, None, True, None, 0)
    # lis.append(md_1)
    # print(entry_md.layer,lis[0].layer,md_1 in lis)
    # fn_0()
    # entry_md = module(0, None, None, None, None, True, None, 0)
    # md_1=module(1, None, None, None, None, True, None, 0)
    # lis=[entry_md,md_1]
    # lis_1=copy.deepcopy(lis)
    # # md_2.idx=2
    # print(lis==lis_1)
    # a=np.zeros((2,2),dtype=np.int)
    # a[0][1]=1
    # print(a)
    # a=[]
    # for i in a:
    #     print(i)
    # a,b=fn_3(2)
    # print(a,b)
    # a=np.ones((2,3),dtype=np.int)
    # a[0,1]=0
    # a[1,1]=2
    # print(a)
    # for i in range(a.shape[0]):
    #     print(a[i,1])
    # fn_4()
    # fn_0()
    # fn_6()
    fn_7()