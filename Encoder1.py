import tensorflow as tf
from tf_op import *

class Encoder1(object):
    def __init__(self,mds_encoder,processor_encoder,act_fn,scope,sess):
        self.mds_embedding=mds_encoder.outputs
        self.processor_embedding=processor_encoder.outputs
        self.act_fn=act_fn
        self.scope=scope
        self.sess=sess

        self.merge_embedding=tf.concat([self.mds_embedding,self.processor_embedding],axis=0)


        #初始化是tensor，经过中间计算之后的结果仍然是tensor
        self.weight,self.bias=self.init(tf.shape(self.mds_embedding)[0],tf.shape(self.merge_embedding)[0])

        self.outputs=self.forward()


    def init(self,input_dim,output_dim):
        weight=glorot([input_dim,output_dim],scope=self.scope)
        bias=zeros([output_dim],scope=self.scope)
        return weight,bias

    def forward(self):
        x=self.merge_embedding
        x=tf.matmul(self.weight,x)
        x=self.act_fn(x)
        return x


