# encoder for reduncancy & deployment

import tensorflow as tf
from tf_op import *

class Encoder2(object):
    def __init__(self,embedding1,embedding2,out_dim,act_fn,scope,sess):
        self.embedding1=embedding1
        self.embedding2=embedding2
        self.out_dim=out_dim
        self.act_fn=act_fn
        self.scope=scope
        self.sess=sess
        self.merge_embedding=tf.concat([self.embedding1,self.embedding2],axis=1)

        self.weight,self.bias=self.init(self.out_dim,tf.shape(self.embedding1)[0])

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