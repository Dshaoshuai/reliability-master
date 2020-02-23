# modules & processors encoder

import tensorflow as tf
from tf_op import *

class Encoder(object):
    def __init__(self,inputs,input_dim,hid_dims,output_dim,
                 act_fn,scope):
        self.inputs=inputs
        self.input_dim=input_dim
        self.hid_dims=hid_dims
        self.output_dim=output_dim
        self.act_fn=act_fn
        self.scope=scope

        self.prep_weights,self.prep_bias=self.init(self.input_dim,self.hid_dims,self.output_dim)
        self.outputs=self.forward()

    def init(self,input_dim,hid_dims,output_dim):
        weights=[]
        bias=[]
        curr_in_dim=input_dim
        for hid_dim in hid_dims:
            weights.append(glorot([curr_in_dim,hid_dim],scope=self.scope))
            bias.append(zeros([hid_dim],scope=self.scope))
            curr_in_dim=hid_dim
        weights.append(glorot([curr_in_dim,output_dim],scope=self.scope))
        bias.append(zeros([output_dim],scope=self.scope))

        return weights,bias

    def forward(self):
        x=self.inputs
        for l in range(len(self.prep_weights)):
            x=tf.matmul(x,self.prep_weights[l])
            x+=self.prep_bias[l]
            x=self.act_fn(x)
        return x

