import tensorflow as tf
from tf_op import *
from Encoder import *
from Encoder1 import *
import tensorflow.contrib.layers as tl
from Encoder2 import *
from params import *
import numpy as np


class ActorAgent(object):
    def __init__(self,sess,module_input_dim,processor_input_dim,hid_dims,
                 output_dim,eps=1e-6,act_fn=leaky_relu,
                 optimizer=tf.train.AdamOptimizer,scope='actor_agent'):
        self.sess=sess
        # self.jobs_input=jobs_input

        #frontier_mds dimension: number of frontier mds * number of mds
        # self.frontier_mds=tf.placeholder(tf.float32,[None,None])               #设置为形参

        #indg_redep_mat dimension: number of mds * number of mds * number of processors
        # self.indg_redep_mat=tf.placeholder(tf.float32,[None,None,None])

        #mds_jobs_vec dimension: 1* number of jobs
        # self.mds_jobs_vec=tf.placeholder(tf.float32,[None])

        #processor_edge_vec dimension: 1 * number of edge clouds
        # self.processor_edge_vec=tf.placeholder(tf.float32,[None,])

        # self.front_mds_mat=front_mds_mat
        self.module_input_dim=module_input_dim
        self.processor_input_dim=processor_input_dim
        self.hid_dims=hid_dims
        self.output_dim=output_dim
        # self.processor_levels=processor_levels
        self.eps=eps
        self.act_fn=act_fn
        self.optimizer=optimizer
        self.scope=scope

        #frontier_mds: number of available modules
        # self.frontier_mds=tf.placeholder()

        # module_inputs dimension: number of available modules * module features
        self.module_inputs=tf.placeholder(tf.float32,[None,self.module_input_dim])
        print(self.module_inputs)

        # print(self.module_inputs)

        #processor_inputs dimension: number of processors * processor features
        self.processor_inputs=tf.placeholder(tf.float32,[args.num_processors,self.processor_input_dim])

        #advantage term (from Critic) ([batch_size,1])
        self.adv=tf.placeholder(tf.float32)


        ### 经过一个adv encoder来将adv的维度变为[number of available modules,1]，其传入的实参为[number of total modules,1]

        #adv_changed dimension: number of available modules * 1
        # self.adv_cg=tf.placeholder(tf.float32,[None,1])

        # for i in range(tf.shape(self.module_inputs)[0]):
        #     front_md_idx=tf.argmax(self.frontier_mds[i])


        self.mds_encoder=Encoder(self.module_inputs,self.module_input_dim,self.hid_dims,
                                self.output_dim,self.act_fn,self.scope)
        self.processors_encoder=Encoder(self.processor_inputs,self.processor_input_dim,self.hid_dims,
                                self.output_dim,self.act_fn,self.scope)
        # self.embedding_encoder=Encoder1(self.mds_encoder,self.processors_encoder,self.act_fn,self.scope,self.sess)

        # self.merge_embedding_0 = tf.transpose(tf.concat([self.mds_encoder.outputs, self.processors_encoder.outputs], axis=0))

        self.merge_embedding=tf.matmul(self.mds_encoder.outputs,tf.transpose(self.processors_encoder.outputs))

        self.module_act_probs,self.redundancy_act,self.ret=self.actor_network(
            self.module_inputs,self.mds_encoder,self.processor_inputs,self.processors_encoder,self.merge_embedding,
            self.act_fn)

        self.module_entropy=-tf.reduce_sum(tf.multiply(
            self.module_act_probs,tf.log(self.module_act_probs)
        ))

        #actor loss due to advantage (negated)
        self.adv_loss=tf.multiply(
            tf.log(self.module_act_probs[tf.argmax(self.module_act_probs)]+self.eps),
            -self.adv
        )

        #normalize entropy
        self.module_entropy/=tf.log(tf.cast(tf.shape(self.module_act_probs)[0],tf.float32))


        #定义当redundancy & deployment的决策全为0时，给予一个大的惩罚
        # self.red_loss=tf.cond(tf.equal(self.redundancy_act[tf.argmax(self.redundancy_act)],0),lambda :tf.constant(10,dtype=tf.float32),lambda :tf.constant(0,dtype=tf.float32))

        #define combined loss
        # self.act_loss=self.adv_loss+self.module_entropy
        self.act_loss=tf.to_int32(tf.reduce_sum(self.ret))+tf.to_int32(tf.reduce_sum(self.redundancy_act))+tf.to_int32(self.adv_loss)

        self.params=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=self.scope)

        self.input_params,self.set_params_op=self.define_params_op()

        # print('+++',tf.shape(self.act_loss))

        #actor gradients

        self.act_gradients=tf.gradients(self.act_loss,self.params)

        #adaptive learning rate
        self.lr_rate=tf.placeholder(tf.float32,shape=[])

        #actor optimizer
        # self.act_opt=self.optimizer(self.lr_rate).minimize(self.act_loss)

        # apply gradient directly to update parameters
        # self.apply_grads=self.optimizer(self.lr_rate).apply_gradients(
        #     zip(self.act_gradients,self.params)
        # )

        #network parameter saver
        self.saver=tf.train.Saver(max_to_keep=args.num_saved_models)
        self.sess.run(tf.global_variables_initializer())

        if args.saved_model is not None:
            self.saver.restore(self.sess,args.saved_model)


    def define_params_op(self):
        #define operations for setting network parameters
        input_params=[]
        for param in self.params:
            input_params.append(tf.placeholder(tf.float32,shape=param.get_shape()))
        set_params_op=[]
        for idx,param in enumerate(input_params):
            set_params_op.append(self.params[idx].assign(param))
        return input_params,set_params_op





    def actor_network(self,module_inputs,mds_embedding,processor_inputs,processors_embedding,merge_embedding,act_fn):
        #不用reshape
        with tf.variable_scope(self.scope):
            md_hid_0=tl.fully_connected(merge_embedding,32,activation_fn=act_fn)
            md_hid_1=tl.fully_connected(md_hid_0,16,activation_fn=act_fn)
            md_hid_2=tl.fully_connected(md_hid_1,8,activation_fn=act_fn)
            md_outputs=tl.fully_connected(md_hid_2,1,activation_fn=act_fn)

            md_outputs=tf.squeeze(md_outputs,axis=1)  #去除值为1的维度
            # print('***',md_outputs.shape)
            # try:
            md_act_prob=tf.nn.softmax(md_outputs)      #module action
            # except Exception as e:
            #     print('except:',e)

            # md_act_idx=tf.argmax(md_act_prob)
            # module_idx=tf.argmax(self.frontier_mds[md_act_idx])+1
            # sum_mds=0
            # job_idx=-1
            # md_job_idx=-1
            # for num in self.mds_jobs_vec:
            #     sum_mds+=num
            #     if sum_mds>=module_idx:
            #         job_idx=self.mds_jobs_vec.index(num)
            #         md_job_idx=module_idx-sum_mds-num-1
            #         break

            # module_act=self.frontier_mds[md_act_idx]
            # job_act=module_act.job

            #redundancy & deployment
            # merge_embedding_1=tf.matmul(self.processors_encoder.outputs,tf.transpose(tf.concat([md_hid_2, self.processors_encoder.outputs], axis=0)))
            merge_embedding_1=tf.matmul(self.processors_encoder.outputs,tf.matmul(tf.transpose(self.mds_encoder.outputs),md_hid_2))
            # merge_embedding_1=self.processors_encoder.outputs
            # merge_embedding_1=Encoder2(md_hid_2,md_outputs,tf.shape(processor_inputs)[0],act_fn,self.scope,self.sess)
            md_hid_3=tl.fully_connected(merge_embedding_1,32,activation_fn=act_fn)
            md_hid_4=tl.fully_connected(md_hid_3,16,activation_fn=act_fn)
            md_hid_5=tl.fully_connected(md_hid_4,8,activation_fn=act_fn)
            md_outputs_1=tl.fully_connected(md_hid_5,1,activation_fn=act_fn)

            md_outputs_11=tf.squeeze(md_outputs_1)
            redundancy_act=tf.to_int32(tf.round(tf.sigmoid(md_outputs_11)))  #冗余 action

            redundancy=tf.to_int32(tf.reduce_sum(redundancy_act))
            # module_act.Set_redundancy(redundancy)
            # module_act.Set_lis_processors(redundancy_act)
            # job_act.Upt_mat_red()

            #retransmission
            # num_pre_red_mds=job_act.Get_num_pre_red_mds(module_act)
            # merge_embedding_2=Encoder2(md_hid_5,md_outputs_1,tf.shape(processor_inputs)[0],act_fn,self.scope)


            merge_embedding_2=tf.concat([md_hid_5,md_outputs_1],axis=1)
            # merge_embedding_2=self.processors_encoder.outputs
            md_hid_6=tl.fully_connected(merge_embedding_2,32,activation_fn=act_fn)
            md_hid_7=tl.fully_connected(md_hid_6,16,activation_fn=act_fn)
            md_hid_8=tl.fully_connected(md_hid_7,8,activation_fn=act_fn)
            md_outputs_2=tl.fully_connected(md_hid_8,args.num_processors,activation_fn=act_fn)

            ret = tf.to_int32(tf.round(tf.nn.relu(md_outputs_2)))  #重传 action


            # lis_pre_mds_red = []                                             # 该module的前驱模块到该模块的重传次数矩阵的列表
            # num_processors = tf.shape(processor_inputs)[0]
            # if md_job_idx==0:                                                # 如果该module为entry module
            #     pre_mds=np.zeros((num_processors,),dtype=np.int)
            #     pre_mds[0]=1
            #     lis_pre_mds_red.append(pre_mds)
            # else:
            #     for indg in self.indg_redep_mat[job_idx][:,md_job_idx]:
            #         if any(indg!=0):                                             #此处需保证待选模块的前驱模块都已经被调度完成，需保证为numpy数组
            #             lis_pre_mds_red.append(indg)
            # retrans_act=[]
            #
            # for indg in lis_pre_mds_red:
            #     retrans=np.zeros((num_processors,num_processors),dtype=int)
            #     i=0
            #     while i <num_processors:
            #         while indg[i]==0 and i<num_processors:
            #             i+=1
            #         if i==num_processors:
            #             break
            #         inte = self.GetMdsInterval(i) # 输入一个i值，输出其位于的job的module indexes所在的区间
            #         j=0
            #         while j<num_processors:
            #             while redundancy_act[j]==0 and j<num_processors:
            #                 j+=1
            #             if j==num_processors:
            #                 break
            #             if j<inte[0] or j>inte[1]:
            #                 retrans[i,j]=ret[i,j]
            #             j+=1
            #         i+=1
            #     retrans_act.append(retrans)
            # retrans_act=tf.convert_to_tensor(retrans_act)
            return md_act_prob,redundancy_act,ret

    def GetMdsInterval(self,idx,processor_edge_vec):
        sum_i=0
        inte=[]
        for num_p in processor_edge_vec:
            sum_i+=num_p
            if sum_i>=idx+1:
                inte.extend([sum_i-num_p,sum_i-1])
                break
        return inte



    def set_params(self,input_params):
        self.sess.run(self.set_params_op,feed_dict={
            i:d for i, d in zip(self.input_params,input_params)
        })

    def invoke_model(self,obs):
        modules_input,processors_input,indg_redep_mat,\
        mds_jobs_vec,front_mds_mat,processor_edge_vec,\
        lis_cur_jobs, previous_job,jobs_selected, lis_frontier_mds,\
        lis_mds_selected, exe_env=self.translate_state(obs)     #numpy type

        # print('front_mds_mat: ',front_mds_mat)

        # print(modules_input,'\n',processors_input)

        md_act_prob, redundancy_act, ret=\
            self.predict(modules_input,processors_input,indg_redep_mat,
                         mds_jobs_vec,front_mds_mat,processor_edge_vec)

        # indg_redep_mat=indg_redep_mat.tolist()
        mds_jobs_vec=mds_jobs_vec.tolist()
        front_mds_mat=front_mds_mat.tolist()
        md_act_idx = int(np.argmax(md_act_prob))
        # if 1 not in front_mds_mat[md_act_idx]:
        #     print(front_mds_mat[md_act_idx],'\n',md_act_idx,'\n',len(front_mds_mat),'\n',np.array(front_mds_mat))
        module_idx = front_mds_mat[md_act_idx].index(1) + 1
        sum_mds = 0
        job_idx = -1
        md_job_idx = -1
        # print(mds_jobs_vec,'\n',module_idx)
        for i in range(len(mds_jobs_vec)):
            sum_mds += mds_jobs_vec[i]
            if sum_mds >= module_idx:
                job_idx = i
                md_job_idx = module_idx - (sum_mds - mds_jobs_vec[i]) - 1
                break

        # print(job_idx,md_job_idx)

        lis_pre_mds_red = []  # 该module的前驱模块到该模块的重传次数矩阵的列表
        num_processors = np.shape(processors_input)[0]
        if md_job_idx == 0:  # 如果该module为entry module
            pre_mds = np.zeros((num_processors,), dtype=np.int)
            pre_mds[0] = 1
            lis_pre_mds_red.append(pre_mds)
        else:
            for indg in indg_redep_mat[job_idx][:, md_job_idx]:
                if any(indg != 0):      # 此处需保证待选模块的前驱模块都已经被调度完成，需保证为numpy数组
                    lis_pre_mds_red.append(indg)
        retrans_act = []
        # print('lis_pre_mds_red: ',lis_pre_mds_red)
        # print('processor_edge_vec: ',processor_edge_vec)

        # print(num_processors)

        for indg in lis_pre_mds_red:
            # print('ret: ',ret)
            # print('indg: ',indg)
            # print('redundancy_act: ',redundancy_act)
            retrans = np.zeros((num_processors, num_processors), dtype=np.int32)
            i = 0
            while i < num_processors:
                while i < num_processors and indg[i] == 0:
                    i += 1
                if i == num_processors:
                    break
                inte = self.GetMdsInterval(i,processor_edge_vec)  # 输入一个i值，输出其位于的job的module indexes所在的区间
                j = 0
                while j < num_processors:
                    while j < num_processors and redundancy_act[j] == 0:
                        j += 1
                    if j == num_processors:
                        break
                    if j < inte[0] or j > inte[1]:
                        retrans[i, j] = ret[i, j]
                    j += 1
                i += 1
            # print('retrans: ',retrans)
            retrans_act.append(retrans)

        if len(retrans_act)==0:
            retrans_act.append(np.zeros((num_processors, num_processors), dtype=np.int32))

        retrans_act=np.array(retrans_act)

        mds_jobs_vec = np.array(mds_jobs_vec)
        front_mds_mat = np.array(front_mds_mat)

        return md_act_prob,redundancy_act,retrans_act,\
               modules_input,processors_input,indg_redep_mat,\
               mds_jobs_vec,front_mds_mat,processor_edge_vec,ret


    def predict(self,modules_input,processors_input,indg_redep_mat,mds_jobs_vec,front_mds_mat,processor_edge_vec):

        # print('predict modules_input: ',modules_input)

        return self.sess.run([self.module_act_probs,self.redundancy_act,self.ret],
                             feed_dict={i: d for i,d in zip(
                                 [self.module_inputs]+[self.processor_inputs],
                                 [modules_input]+[processors_input]
                             )})


    def get_gradients(self,modules_input,processors_input,module_act_prob,redundancy_act,
            retrans_act,adv):

        return self.sess.run([self.act_gradients, [self.act_loss, self.adv_loss,self.module_entropy]],
                             feed_dict={i: d for i, d in zip(
                                 [self.module_inputs] + [self.processor_inputs]+
                                 [self.module_act_probs]+
                                 [self.redundancy_act]+[self.ret]+[self.adv],
                                 [modules_input] + [processors_input]+
                                 [module_act_prob]+[redundancy_act]+[retrans_act]+[adv]
                             )})

        # return self.sess.run([self.act_gradients],
        #                      feed_dict={i: d for i, d in zip(
        #                          [self.module_inputs] + [self.processor_inputs] +
        #                          [self.module_act_probs] +
        #                          [self.redundancy_act] + [self.ret] + [self.adv],
        #                          [modules_input] + [processors_input] +
        #                          [module_act_prob] + [redundancy_act] + [retrans_act] + [adv]
        #                      )})







    def get_params(self):
        return self.sess.run(self.params)

    def apply_gradients(self,gradients,lr_rate):

        self.apply_grads = self.optimizer(self.lr_rate).apply_gradients(
            zip(self.act_gradients, self.params)
        )

        self.sess.run(self.apply_grads,feed_dict={
            i:d for i,d in zip(
                [self.act_gradients]+[self.lr_rate],
                [gradients]+[lr_rate]
            )
        })

    def save_model(self, file_path):
        self.saver.save(self.sess, file_path)





    def translate_state(self,obs):
        #Translate the observation to matrix form
        lis_cur_jobs, previous_job, jobs_selected, lis_frontier_mds, lis_mds_selected, exe_env=obs
        indg_redep_mat=[]
        mds_jobs_vec = []
        for job in lis_cur_jobs:
            indg_redep_mat.append(job.indg_redep_mat)
            mds_jobs_vec.append(job.num_mds)
        front_mds_mat=np.zeros((len(lis_frontier_mds),sum(mds_jobs_vec)),dtype=np.int32)
        # print(lis_cur_jobs)
        for md in lis_frontier_mds:
            job=md.job
            # for job in lis_cur_jobs:
            #     if job.JudMd(md):
            # if job not in lis_cur_jobs:
            #     print(job.lis_mds.index(md),len(job.lis_mds))
            if job not in lis_cur_jobs:
                # print('not in lis_cur_jobs: ',job)
                # print('lis_cur_jobs: ',lis_cur_jobs)
                print('lis_frontier_mds: ',job.lis_mds.index(lis_frontier_mds[0]))
                print('job.lis_mods: ',[job.lis_mds.index(md) for md in job.lis_mds])
                print('job.ori_adj_mat: \n',job.ori_adj_mat)
            job_idx=lis_cur_jobs.index(job)
            md_idx=job.GetIndxMd(md)
            front_mds_mat[lis_frontier_mds.index(md)][sum(mds_jobs_vec[:job_idx])+md_idx]=1
        processor_edge_vec=[]                                               #1 * number of edge clouds (element: number of processors of each edge cloud)
        lis_processors = []
        lis_edge_clouds=exe_env.lis_edge_clouds
        for edge_cloud in exe_env.lis_edge_clouds:
            processor_edge_vec.append(edge_cloud.num_processors)
            lis_processors.extend(edge_cloud.lis_processors)
        modules_input=np.zeros((len(lis_frontier_mds),args.module_input_dim),dtype=int)
        # res_mds_jobs=[]
        for md in lis_frontier_mds:
            md_idx=lis_frontier_mds.index(md)
            job=md.job
            modules_input[md_idx,0]=lis_cur_jobs.index(job)
            modules_input[md_idx,1]=job.release_time
            modules_input[md_idx,2]=job.num_mds
            modules_input[md_idx, 3] = job.total_md_op_amount
            modules_input[md_idx,4]=job.res_num_mds
            modules_input[md_idx,5]=job.res_md_op_amound
            modules_input[md_idx,6]=job.ori_total_edges_data_volume
            modules_input[md_idx,7]=job.res_edge_data_volume
            idc=0
            if job.is_stateful:
                idc=1
                modules_input[md_idx,8]=idc
                modules_input[md_idx,9]=job.total_stat_amount
                modules_input[md_idx,10]=job.res_stat_amount
            else:
                modules_input[md_idx,8]=idc
                modules_input[md_idx,9]=0
                modules_input[md_idx,10]=0
            modules_input[md_idx,11]=job.lis_mds.index(md)
            modules_input[md_idx,12]=md.earliest_start_time
            modules_input[md_idx,13]=md.op_amount
            modules_input[md_idx,14]=len(md.lis_pre_mods)
            modules_input[md_idx,15]=len(md.lis_sub_mods)
            idc_md=0
            if md.is_stateful:
                idc_md=1
                modules_input[md_idx,16]=idc_md
                modules_input[md_idx,17]=md.state.amount
            else:
                modules_input[md_idx,16]=idc_md
                modules_input[md_idx,17]=0

        processors_input=np.zeros((sum(processor_edge_vec),args.processor_input_dim),dtype=int)

        assert sum(processor_edge_vec)==len(lis_processors)
        for processor in lis_processors:
            processor_idx=lis_processors.index(processor)
            edge_cloud=processor.edge_cloud
            processors_input[processor_idx,0]=lis_edge_clouds.index(edge_cloud)
            processors_input[processor_idx,1]=edge_cloud.num_processors
            # processors_input[processor_idx,2]=len(edge_cloud.lis_idle_processors)          #删除
            # processors_input[processor_idx,3]=len(edge_cloud.lis_cur_wai_modules)          #删除
            # total_op_amount_wait=0
            # for md in edge_cloud.lis_cur_wai_modules:
            #     total_op_amount_wait+=md.op_amount
            # processors_input[processor_idx,4]=total_op_amount_wait                           #删除
            num_channels=0
            # num_edges = 0
            # total_data_volume=0
            num_edges_wait=0
            total_data_volume_wait=0
            edge_cloud_idx=exe_env.lis_edge_clouds.index(edge_cloud)
            for net in exe_env.net_env_adj_mat[edge_cloud_idx]:
                if net:
                    num_channels+=net.num_net_channels
                    # num_edges+=len(net.lis_finished_edges)
                    # total_data_volume+=net.total_data_volume
                    num_edges_wait+=len(net.lis_wait_edges)
                    total_data_volume_wait+=net.total_data_volume_wait
            processors_input[processor_idx,2]=num_channels
            # processors_input[processor_idx,6]=num_edges                       #删除，参照MDP的原理，只考虑当前的等待信息
            # processors_input[processor_idx,7]=total_data_volume               #删除
            processors_input[processor_idx,3]=num_edges_wait
            processors_input[processor_idx,4]=total_data_volume_wait
            processors_input[processor_idx,5]=len(edge_cloud.lis_stat_mds)
            processors_input[processor_idx,6]=edge_cloud.total_stat

            processors_input[processor_idx,7]=edge_cloud.lis_processors.index(processor)
            processors_input[processor_idx,8]=processor.cap_process
            # idc_pro=0
            # if processor.is_idle:
            #     idc_pro=1
            # processors_input[processor_idx,14]=1                                 #删除
            processors_input[processor_idx,9]=len(processor.lis_wait_modules)
            processors_input[processor_idx,10]=processor.total_op_amount_wait
            # processors_input[processor_idx,17]=len(processor.lis_edges)          #删除
            # processors_input[processor_idx,18]=processor.total_data_volume       #删除
            # processors_input[processor_idx,19]=len(processor.lis_edges_wait)     #删除
            # processors_input[processor_idx,20]=processor.total_data_volume_wait  #删除
            processors_input[processor_idx,11]=len(processor.lis_stat_mds)
            processors_input[processor_idx,12]=processor.total_stat

        return modules_input,processors_input,np.array(indg_redep_mat),np.array(mds_jobs_vec),front_mds_mat,\
               np.array(processor_edge_vec),lis_cur_jobs, previous_job,jobs_selected, lis_frontier_mds, lis_mds_selected, exe_env


























