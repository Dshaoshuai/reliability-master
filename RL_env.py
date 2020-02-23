import numpy as np
from edge_cloud_env.WallTime import *
from edge_cloud_env.TimeLine import *
from RewardCalculator import *
from exe_environment_generator import *
from params import *
from job_generator import *
from action_map import *
from edge_cloud_env.red_module import *
from edge_cloud_env.red_edge import *
import math

class RL_env(object):
    def __init__(self):
        self.np_random=np.random.RandomState()

        #全局计时器
        self.wall_time=WallTime()

        #在线到达的作业jobs列表（按照释放时间先后排序）
        self.timeline=[]

        #for computing reward at each step
        # self.reward_calculator=RewardCalculator()

        #迄今选择的modules列表
        self.lis_mds_selected=[]


        # self.lis_finished_jobs=[]

        #当前可供选择的jobs
        self.lis_cur_jobs=[]

        #迄今选择的jobs列表
        self.jobs_selected=[]

        #当前可供选择的modules，相当于调度器的等待队列
        self.lis_frontier_mds=[]

        #执行环境对象
        self.exe_env=generate_exe_environment(args.num_edge_clouds,args.lis_dist_ec_grade,args.dic_num_ec_grade,args.dic_class_edge_clouds,args.cap_processor
                                              ,args.dic_num_ec_class_total,args.dic_num_ec_class_grade,args.bw_in_channel,args.min_num_channels,args.max_num_channels,
                                              args.avg_processor_err_prob,args.avg_bw_err_prob)


    def observe(self):
        return self.lis_cur_jobs,self.previous_job,self.jobs_selected,self.lis_frontier_mds,self.lis_mds_selected,self.exe_env
    # observations包含：当前可供选择的jobs对象列表，上一次调度的module所在的job对象，当前可供选择的modules对象列表，已经被调度执行的modules对象列表，当前的执行环境现状

    # def get_lis_frontier_mds(self):
    #     pass

    def Get_init_frontier_mds(self):
        init_frontier_mds=[]
        for job in self.lis_cur_jobs:
            init_frontier_mds.append(job.lis_mds[0])
        return init_frontier_mds



    def step(self,module_act,redundancy_act,retrans_act):
        # print(module_act,'\n',self.lis_mds_selected)
        assert module_act not in self.lis_mds_selected
        # self.lis_mds_selected.append(module_act)
        job_act=module_act.job
        lis_job_mds=job_act.lis_mds
        lis_ecs=self.exe_env.lis_edge_clouds
        lis_processors=self.exe_env.lis_processors
        lis_pre_mds=module_act.lis_pre_mods
        # print(redundancy_act)                           ###
        poses_1=self.Get_poses(redundancy_act)
        red_idx=0
        # print('&&&&', len(module_act.lis_redund))
        if module_act.lis_redund:
            module_act.lis_redund.clear()
        for pos in poses_1:
            red_md=red_module(red_idx,module_act,job_act,None,module_act.is_stateful,module_act.state,module_act.op_amount)
            red_md.processor=lis_processors[pos]
            red_md.edge_cloud=lis_processors[pos].edge_cloud
            module_act.lis_redund.append(red_md)
        # print('&&&',len(module_act.lis_redund))
        if lis_job_mds.index(module_act)==0:               #如果该module是entry module
            eg=module_act.entry_edge
            retrans=retrans_act[0]
            # print(retrans)                              ###
            ec_0_md=lis_ecs[0]
            for pos in poses_1:
                # print(pos)                              ###
                red_md=module_act.lis_redund[poses_1.index(pos)]
                t_1=job_act.release_time
                if not self.Jud_same_ec(pos,0):             #判断两个md是否在同一个ec，如果不在同一个ec，就会产生cross edge的传输
                    ec_idx = self.Get_ec_idx(pos)
                    net = self.exe_env.net_env_adj_mat[0][ec_idx]
                    # print('exe_env.net_env_adj_mat: ',self.exe_env.net_env_adj_mat)

                    red_eg=red_edge(pos,red_md,red_md,eg,eg.data_volume)
                    red_eg.Set_Release_time(t_1)
                    # print('ec_idx: ',ec_idx)

                    # print('net: ',net)
                    # print('ec_idx: ',ec_idx)
                    # print('len(net.lis_net_channels): ',len(net.lis_net_channels))
                    # print('len(net.lis_ear_channels): ',len(net.lis_ear_channels))

                    channel = net.lis_ear_channels[0]
                    if red_eg.release_time>=net.earliest_avail_time:      #如果释放时间大于net_env的最早可用时间，即不需要等待                               #任务的释放时间（关于保证FCFS顺利执行的问题在下面再讨论）
                        red_eg.Set_Start_time(red_eg.release_time)
                    else:
                        red_eg.Set_Start_time(net.earliest_avail_time)
                        red_eg.wait_time=net.earliest_avail_time-red_eg.release_time
                        #不考虑等待队列
                    red_eg.Set_net_env(net)
                    red_eg.Set_net_channel(channel)
                    red_eg.trans_time=red_eg.sour_edge.data_volume/channel.net_bandwidth
                    red_eg.num_retrans=retrans[0,pos]
                    red_eg.ori_propg_time=self.exe_env.dis_adj_mat[0][ec_idx]/args.propa_speed
                    red_eg.trans_suc = math.exp(-channel.err_prob_tran * (red_eg.ori_propg_time+red_eg.trans_time))
                    red_eg.exp_num_trans=self.Get_exp_num_trans(red_eg.num_retrans,red_eg.trans_suc)      #计算期望传输次数的函数
                    red_eg.exp_propg_time=(red_eg.exp_num_trans-1)*red_eg.ori_propg_time*2+red_eg.ori_propg_time
                    red_eg.Set_end_time()
                    red_eg.Set_reliability()

                    channel.lis_finished_edges.append(red_eg)
                    net.lis_finished_edges.append(red_eg)


                    # 更新net env的edges等待队列
                    net.Updat_lis_wait_edges(red_eg)


                    channel.Set_earliest_avail_time(red_eg.end_time)
                    # print('red_eg.end_time: ',red_eg.end_time)
                    # print('channel.earliest_avail_time: ',channel.earliest_avail_time)
                    net.Set_earliest_avail_time()
                    # print('lis net earliest_avail_time: ',[cha.earliest_avail_time for cha in net.lis_net_channels])


                    # print([ch.earliest_avail_time for ch in net.lis_net_channels])    ###
                    # print(len(net.lis_net_channels), net.earliest_avail_time, len(net.lis_ear_channels))    ###

                    net.Set_lis_ear_channels()


                    # print(len(net.lis_net_channels),net.earliest_avail_time,len(net.lis_ear_channels))       ###

                    t_1=red_eg.end_time
                    # print('num_net_channels: ',net.num_net_channels)
                    # print('red_eg start time: ',red_eg.start_time,' end time: ',red_eg.end_time)


                red_md.release_time=t_1
                proces=red_md.processor
                if red_md.release_time>=proces.earlist_aval_time:
                    red_md.start_time=red_md.release_time
                else:
                    red_md.start_time=proces.earlist_aval_time
                    red_md.wait_time=proces.earlist_aval_time-red_md.release_time
                red_md.exec_time=red_md.source_mod.op_amount/proces.cap_process
                red_md.Set_end_time()
                red_md.Set_exe_suc(proces.err_prob_op)
                # print('entry_md, start time: ',red_md.start_time,' end_time: ',red_md.end_time)

                # 更新processor的等待队列
                proces.Updat_lis_wait_modules(red_md)

                proces.lis_sum_modules.append(red_md)
                proces.Set_earlist_aval_time(red_md.end_time)
                proces.edge_cloud.lis_red_modules.append(red_md)

                #如果该red_md为stateful
                if red_md.is_stateful:
                    proces.lis_stat_mds.append(red_md)
                    proces.total_stat+=red_md.state.amount
                    proces.edge_cloud.lis_stat_mds.append(red_md)
                    proces.edge_cloud.total_stat+=red_md.state.amount

                if not self.Jud_same_ec(pos, 0):                               #在计算md的reliability时，只有当该md是entry md时只需要计算该md和前驱数据流（如果有的话），否则还需要乘上前驱md的可靠性
                    red_md.reliability=red_md.exe_suc*red_eg.reliability
                else:
                    red_md.reliability=red_md.exe_suc


        elif lis_job_mds.index(module_act)==len(lis_job_mds)-1:                #如果该md为exit module
            for red_md in module_act.lis_redund:
                release_contrib=[]                   #记录每个前驱md的完成时间，每个前驱cross edge的结束时间，最后取最大值作为该re_md的释放时间
                pos_1=poses_1[module_act.lis_redund.index(red_md)]
                ec_idx_1=self.Get_ec_idx(pos_1)             #返回所在的edge cloud的编号
                relia_tem_1=1
                for pre_md in module_act.lis_pre_mods:
                    retrans = retrans_act[module_act.lis_pre_mods.index(pre_md)]
                    sour_eg=job_act.ori_adj_mat_edges[lis_job_mds.index(pre_md),lis_job_mds.index(module_act)]
                    # print('sour_eg: ',sour_eg)
                    # print('job_act.ori_adj_mat_edges: ',job_act.ori_adj_mat_edges)

                    lis_pre_re_mds=pre_md.lis_redund
                    poses_2=self.Get_poses(pre_md.lis_processors)
                    relia_pre=1
                    for pos_2 in poses_2:
                        ec_idx_2=self.Get_ec_idx(pos_2)
                        pre_re_md=lis_pre_re_mds[poses_2.index(pos_2)]
                        t_2=pre_re_md.end_time
                        relia_tem_2=pre_re_md.reliability
                        if ec_idx_2!=ec_idx_1:
                            red_eg=red_edge(pos_2,lis_pre_re_mds[poses_2.index(pos_2)],red_md,sour_eg,sour_eg.data_volume)
                            net=self.exe_env.net_env_adj_mat[ec_idx_2][ec_idx_1]
                            red_eg.Set_Release_time(t_2)
                            channel = net.lis_ear_channels[0]
                            if red_eg.release_time >= net.earliest_avail_time:  # 如果释放时间大于net_env的最早可用时间，即不需要等待                               #任务的释放时间（关于保证FCFS顺利执行的问题在下面再讨论）
                                red_eg.Set_Start_time(red_eg.release_time)
                            else:
                                red_eg.Set_Start_time(net.earliest_avail_time)
                                red_eg.wait_time = net.earliest_avail_time - red_eg.release_time
                            red_eg.Set_net_env(net)
                            red_eg.Set_net_channel(channel)
                            red_eg.trans_time = red_eg.data_volume / channel.net_bandwidth
                            red_eg.num_retrans = retrans[pos_2, pos_1]
                            red_eg.ori_propg_time = self.exe_env.dis_adj_mat[ec_idx_2][ec_idx_1] / args.propa_speed
                            red_eg.trans_suc = math.exp(-channel.err_prob_tran * (red_eg.ori_propg_time+red_eg.trans_time))
                            red_eg.exp_num_trans = self.Get_exp_num_trans(red_eg.num_retrans, red_eg.trans_suc)
                            red_eg.exp_propg_time = (red_eg.exp_num_trans - 1) * red_eg.ori_propg_time * 2 + red_eg.ori_propg_time
                            red_eg.Set_end_time()
                            red_eg.Set_reliability()         #将cross edge的可靠性修改为包含前驱red_md
                            red_eg.Set_reliability_pre(pre_re_md)
                            channel.lis_finished_edges.append(red_eg)
                            net.lis_finished_edges.append(red_eg)

                            # 更新net env的edges等待队列
                            net.Updat_lis_wait_edges(red_eg)

                            channel.Set_earliest_avail_time(red_eg.end_time)
                            net.Set_earliest_avail_time()
                            net.Set_lis_ear_channels()
                            t_2 = red_eg.end_time
                            relia_tem_2=red_eg.reliability_pre
                        release_contrib.append(t_2)
                        relia_pre*=1-relia_tem_2
                    relia_tem_1*=1-relia_pre                  #该pre_md中red_md至少有一个执行成功的概率
                # relia_tem_1*=relia_pre
                # red_md.reliability=relia_tem_1
                red_md.release_time=max(release_contrib)
                proces = red_md.processor
                if red_md.release_time >= proces.earlist_aval_time:
                    red_md.start_time = red_md.release_time
                else:
                    red_md.start_time = proces.earlist_aval_time
                    red_md.wait_time = proces.earlist_aval_time - red_md.release_time
                red_md.exec_time = red_md.source_mod.op_amount / proces.cap_process
                red_md.Set_end_time()
                red_md.Set_exe_suc(proces.err_prob_op)

                #更新该processor的等待队列
                proces.Updat_lis_wait_modules(red_md)

                proces.lis_sum_modules.append(red_md)
                proces.Set_earlist_aval_time(red_md.end_time)
                proces.edge_cloud.lis_red_modules.append(red_md)
                red_md.reliability=relia_tem_1*red_md.exe_suc

                # 如果该red_md为stateful
                if red_md.is_stateful:
                    proces.lis_stat_mds.append(red_md)
                    proces.total_stat += red_md.state.amount
                    proces.edge_cloud.lis_stat_mds.append(red_md)
                    proces.edge_cloud.total_stat += red_md.state.amount

            exit_eg=module_act.exit_edge
            # print(len(retrans_act),len(module_act.lis_pre_mods),redundancy_act,[pred.lis_processors for pred in module_act.lis_pre_mods],[pr.sign_md_finished for pr in module_act.lis_pre_mods])
            retrans = retrans_act[0]
            job_end_contrib = []
            relia_tem=1
            for pos in poses_1:
                lis_red_mds=module_act.lis_redund
                pre_re_md=lis_red_mds[poses_1.index(pos)]
                t_md_1=pre_re_md.end_time
                relia_pre_1=pre_re_md.reliability
                if not self.Jud_same_ec(pos,0):
                    red_eg=red_edge(pos,pre_re_md,pre_re_md,exit_eg,exit_eg.data_volume)
                    red_eg.Set_Release_time(pre_re_md.end_time)
                    ec_idx = self.Get_ec_idx(pos)
                    net = self.exe_env.net_env_adj_mat[0][ec_idx]
                    channel = net.lis_ear_channels[0]
                    if red_eg.release_time >= net.earliest_avail_time:  # 如果释放时间大于net_env的最早可用时间，即不需要等待                               #任务的释放时间（关于保证FCFS顺利执行的问题在下面再讨论）
                        red_eg.Set_Start_time(red_eg.release_time)
                    else:
                        red_eg.Set_Start_time(net.earliest_avail_time)
                        red_eg.wait_time = net.earliest_avail_time - red_eg.release_time
                    red_eg.Set_net_env(net)
                    red_eg.Set_net_channel(channel)
                    red_eg.trans_time = red_eg.data_volume / channel.net_bandwidth
                    red_eg.num_retrans = retrans[0, pos]
                    red_eg.ori_propg_time = self.exe_env.dis_adj_mat[0][ec_idx] / args.propa_speed
                    red_eg.trans_suc = math.exp(-channel.err_prob_tran * (red_eg.ori_propg_time+red_eg.trans_time))
                    red_eg.exp_num_trans = self.Get_exp_num_trans(red_eg.num_retrans, red_eg.trans_suc)
                    red_eg.exp_propg_time = (red_eg.exp_num_trans - 1) * red_eg.ori_propg_time * 2 + red_eg.ori_propg_time
                    red_eg.Set_end_time()
                    red_eg.Set_reliability()  # 将cross edge的可靠性修改为包含前驱red_md
                    red_eg.Set_reliability_pre(pre_re_md)
                    channel.lis_finished_edges.append(red_eg)
                    net.lis_finished_edges.append(red_eg)

                    # 更新net env的edges等待队列
                    net.Updat_lis_wait_edges(red_eg)

                    channel.Set_earliest_avail_time(red_eg.end_time)
                    net.Set_earliest_avail_time()
                    net.Set_lis_ear_channels()
                    relia_pre_1=red_eg.reliability_pre
                    t_md_1=red_eg.end_time
                relia_tem*=1-relia_pre_1
                job_end_contrib.append(t_md_1)
            job_act.reliability_real=1-relia_tem
            job_act.end_time=max(job_end_contrib)
            job_act.Set_Make_Span()
            #设置next_obs，reward

        else:                                    #该md为中间md
            for red_md in module_act.lis_redund:
                release_contrib=[]                   #记录每个前驱md的完成时间，每个前驱cross edge的结束时间，最后取最大值作为该re_md的释放时间
                # print(poses_1,module_act.lis_redund.index(red_md),len(module_act.lis_redund))
                pos_1=poses_1[module_act.lis_redund.index(red_md)]
                ec_idx_1=self.Get_ec_idx(pos_1)             #返回所在的edge cloud的编号
                # print('poses_1: ',poses_1)

                relia_tem_1=1
                # print('*** ',len(module_act.lis_pre_mods),lis_job_mds.index(module_act),'\n',module_act.job.ori_adj_mat)
                for pre_md in module_act.lis_pre_mods:
                    # print('len(module_act.lis_pre_mods): ',len(module_act.lis_pre_mods))
                    # print(retrans_act)
                    # print('@@@',len(module_act.lis_pre_mods),module_act.idx,pre_md.idx)
                    retrans = retrans_act[module_act.lis_pre_mods.index(pre_md)]
                    sour_eg=job_act.ori_adj_mat_edges[lis_job_mds.index(pre_md),lis_job_mds.index(module_act)]
                    lis_pre_re_mds=pre_md.lis_redund
                    poses_2=self.Get_poses(pre_md.lis_processors)
                    # print('### ',poses_2)                         ###
                    relia_pre=1
                    for pos_2 in poses_2:
                        ec_idx_2=self.Get_ec_idx(pos_2)
                        pre_re_md=lis_pre_re_mds[poses_2.index(pos_2)]
                        t_2=pre_re_md.end_time
                        relia_tem_2=pre_re_md.reliability
                        # print('pos_1: ',pos_1,'pos_2: ',pos_2)
                        # print('ec_idx_2: ',ec_idx_2,' ec_idx_1: ',ec_idx_1)
                        if ec_idx_2!=ec_idx_1:
                            red_eg=red_edge(pos_2,lis_pre_re_mds[poses_2.index(pos_2)],red_md,sour_eg,sour_eg.data_volume)
                            # print(ec_idx_1,ec_idx_2)
                            net=self.exe_env.net_env_adj_mat[ec_idx_2][ec_idx_1]
                            # print(net)
                            red_eg.Set_Release_time(t_2)
                            # print('len(net.lis_ear_channels): ',len(net.lis_ear_channels))
                            # print('len(net.num_net_channels): ',net.num_net_channels)
                            # print('')
                            # print('pos_2: ',pos_2)
                            # print('ec_idx_2: ',ec_idx_2,', ec_idx_1: ',ec_idx_1)
                            # print('net: ',net)
                            # print('net.lis_net_channels: ',net.lis_net_channels)
                            # print('net.lis_ear_channels: ',net.lis_ear_channels)
                            channel = net.lis_ear_channels[0]
                            if red_eg.release_time >= net.earliest_avail_time:  # 如果释放时间大于net_env的最早可用时间，即不需要等待                               #任务的释放时间（关于保证FCFS顺利执行的问题在下面再讨论）
                                red_eg.Set_Start_time(red_eg.release_time)
                            else:
                                red_eg.Set_Start_time(net.earliest_avail_time)
                                red_eg.wait_time = net.earliest_avail_time - red_eg.release_time
                            red_eg.Set_net_env(net)
                            red_eg.Set_net_channel(channel)
                            red_eg.trans_time = red_eg.data_volume / channel.net_bandwidth
                            red_eg.num_retrans = retrans[pos_2, pos_1]
                            red_eg.ori_propg_time = self.exe_env.dis_adj_mat[ec_idx_2][ec_idx_1] / args.propa_speed
                            red_eg.trans_suc = math.exp(-channel.err_prob_tran * (red_eg.ori_propg_time+red_eg.trans_time))
                            red_eg.exp_num_trans = self.Get_exp_num_trans(red_eg.num_retrans, red_eg.trans_suc)
                            red_eg.exp_propg_time = (red_eg.exp_num_trans - 1) * red_eg.ori_propg_time * 2 + red_eg.ori_propg_time
                            red_eg.Set_end_time()
                            red_eg.Set_reliability()         #将cross edge的可靠性修改为包含前驱red_md
                            red_eg.Set_reliability_pre(pre_re_md)
                            channel.lis_finished_edges.append(red_eg)
                            net.lis_finished_edges.append(red_eg)

                            #更新net env的edges等待队列
                            net.Updat_lis_wait_edges(red_eg)
                            channel.Set_earliest_avail_time(red_eg.end_time)
                            net.Set_earliest_avail_time()
                            # print('net.earliest_avail_time: ',net.earliest_avail_time)
                            net.Set_lis_ear_channels()
                            t_2 = red_eg.end_time
                            relia_tem_2=red_eg.reliability_pre
                        release_contrib.append(t_2)
                        relia_pre*=1-relia_tem_2
                    relia_tem_1*=1-relia_pre                  #该pre_md中red_md至少有一个执行成功的概率
                # relia_tem_1*=relia_pre
                # red_md.reliability=relia_tem_1
                # print('mid_release_contrib:',release_contrib)
                red_md.release_time=max(release_contrib)
                proces = red_md.processor
                if red_md.release_time >= proces.earlist_aval_time:
                    red_md.start_time = red_md.release_time
                else:
                    red_md.start_time = proces.earlist_aval_time
                    red_md.wait_time = proces.earlist_aval_time - red_md.release_time
                red_md.exec_time = red_md.source_mod.op_amount / proces.cap_process
                red_md.Set_end_time()
                red_md.Set_exe_suc(proces.err_prob_op)
                # print('mid_red_md end time: ',red_md.end_time)

                # 更新该processor的等待队列
                proces.Updat_lis_wait_modules(red_md)

                proces.lis_sum_modules.append(red_md)
                proces.Set_earlist_aval_time(red_md.end_time)
                proces.edge_cloud.lis_red_modules.append(red_md)
                red_md.reliability=relia_tem_1*red_md.exe_suc

                # 如果该red_md为stateful
                if red_md.is_stateful:
                    proces.lis_stat_mds.append(red_md)
                    proces.total_stat += red_md.state.amount
                    proces.edge_cloud.lis_stat_mds.append(red_md)
                    proces.edge_cloud.total_stat += red_md.state.amount



        # print('***')
        #1. 更新obs相关（粗粒度，还需对照state逐一更新）
        module_act.lis_processors=redundancy_act
        module_act.retrans_act=retrans_act
        module_act.sign_md_finished=True
        module_act.earliest_start_time=min([red.release_time for red in module_act.lis_redund])
        module_act.end_time=max([red.end_time for red in module_act.lis_redund])
        # print([red.end_time for red in module_act.lis_redund])

        # print([mod.idx for mod in self.lis_frontier_mds])

        self.lis_frontier_mds.remove(module_act)

        print('selected module_act: ', job_act.lis_mds.index(module_act))
        print('index lis_sub_md: ',[job_act.lis_mds.index(mo) for mo in module_act.lis_sub_mods])
        for md in module_act.lis_sub_mods:
            sign=True
            for pre_md in md.lis_pre_mods:
                if pre_md.sign_md_finished==False:
                    sign=False
                    break
            if sign==True:
                # md.Set_earliest_start_time()
                # print(md.earliest_start_time,' +')
                # if md in self.lis_frontier_mds:
                #     # print(md.job.lis_mds.index(md),len(md.job.lis_mds),'\n',[md.job.lis_mds.index(mod) for mod in md.lis_pre_mods],'\n',md.job.ori_adj_mat[md.job.lis_mds.index(md)],'\n',md.job.ori_adj_mat[:,md.job.lis_mds.index(md)],'\n',self.lis_frontier_mds.index(md),len(self.lis_frontier_mds))
                #     print(md.job.lis_mds.index(module_act),md.job.lis_mds.index(md),'\n',md.job.ori_adj_mat)
                # print('idx lis_frontier_mds: ',[job_act.lis_mds.index(modu) for modu in self.lis_frontier_mds])
                print('added md: ',job_act.lis_mds.index(md))
                print('index lis_pre_md: ',[job_act.lis_mds.index(modul) for modul in md.lis_pre_mods])
                assert md not in self.lis_frontier_mds
                self.lis_frontier_mds.append(md)

        print('lis_frontier_mds: ',[job_act.lis_mds.index(modu) for modu in self.lis_frontier_mds])
        job_act.Upt_indg_redep_mat(module_act)
        job_act.res_num_mds-=1
        job_act.res_md_op_amound-=module_act.op_amount
        #统计该md在其job中的前驱数据流量的和
        job_act.res_edge_data_volume-=self.Get_pre_edges_volume(module_act)
        if module_act.is_stateful:
            job_act.res_stat_amount-=module_act.state.amount
        if lis_job_mds.index(module_act)==len(lis_job_mds)-1:        #若为exit module
            self.jobs_selected.append(job_act)
            job_act.Set_lis_red()
            job_act.Set_lis_retrans()
            job_act.sign_job_finished=True
            self.lis_cur_jobs.remove(job_act)
        self.previous_job=job_act

        done=False
        et=module_act.end_time
        if lis_job_mds.index(module_act)==len(lis_job_mds)-1:
            et=job_act.end_time
        if args.arrival_mode=='poisson_arrival':
            while len(self.timeline)>0 and self.timeline[0].release_time<=et:
                jb=self.timeline.pop(0)
                self.lis_cur_jobs.append(jb)
                self.lis_frontier_mds.append(jb.lis_mds[0])
            if self.lis_cur_jobs==[] and self.timeline!=[]:
                jb=self.timeline.pop(0)
                self.lis_cur_jobs.append(jb)
                self.lis_frontier_mds.append(jb.lis_mds[0])
            elif self.lis_cur_jobs==[] and self.timeline==[]:
                done=True
        if self.lis_cur_jobs==[] and self.lis_frontier_mds==[]:
            done=True




        ### 增加job_selected信息，更新processor输入相关，增加每个step结束后判断是否需要添加新的job（流式到达【假设其服从泊松分布】）

        next_obs=self.lis_cur_jobs,self.previous_job,self.jobs_selected,self.lis_frontier_mds,self.lis_mds_selected,self.exe_env




        #2. 给出reward
        reward_md_et=0           #该md的执行持续时间，结束时间-所有前驱red_md的最早完成时间
        reward_md_relinc=0          #该md的执行可靠性的增加量，相对于只部署一个能得到最大的
        reward_jb_mk=0
        reward_jb_rel=0

        # earliest_start_time=1000
        # for pre_md in module_act.lis_pre_mods:
        #     for pre_re_md in pre_md.lis_redund:
        #         if pre_re_md.end_time<earliest_start_time:
        #             earliest_start_time=pre_re_md.end_time
        reward_md_et=-((module_act.end_time-module_act.earliest_start_time)/args.end_time_scale)

        reward_md_relinc=self.CalRelInc(module_act)
        if lis_job_mds.index(module_act) == len(lis_job_mds) - 1:  # 若为exit module
            reward_jb_mk=-(job_act.make_span/args.make_span_scale)
            reward_jb_rel=job_act.reliability_real-job_act.reliability_req
        reward=reward_md_et+reward_md_relinc+reward_jb_mk+reward_jb_rel

        #1. 参考decima来规范化reward的4个组成；2. 逐一更新observation
        return next_obs,reward,done


    def Get_pre_edges_volume(self,md):
        edges_adj_mat=md.job.ori_adj_mat_edges
        md_idx=md.job.lis_mds.index(md)
        pre_edges_volume=0
        for eg in edges_adj_mat[:,md_idx]:
            if type(eg)==edge:
                pre_edges_volume+=eg.data_volume
        return pre_edges_volume




    def CalRelInc(self,md):
        rel_1=1
        rel_2 = -1
        for re_md in md.lis_redund:
            rel_1*=1-re_md.reliability
            if re_md.reliability>rel_2:
                rel_2=re_md.reliability
        rel_1=1-rel_1
        return rel_1-rel_2




    def Get_ec_idx(self,pos):
        processor_edge_vec = []
        for ec in self.exe_env.lis_edge_clouds:
            processor_edge_vec.append(ec.num_processors)
        # print('proc')
        sum_i = 0
        ec_idx=-1
        for num_p in processor_edge_vec:
            sum_i += num_p
            if sum_i >= pos + 1:
                ec_idx=processor_edge_vec.index(num_p)
                break
        return ec_idx









                    # else:
                    #
                    #得到该red_md的前驱数据边为cross edge时的完成时间，要得到其释放时间，还需要考虑前驱数据流为非cross edge时的完成时间（即前驱md的完成时间），取最小值作为该前驱md对于该md的释放时间的贡献




    def Jud_same_ec(self,pos1,pos2):
    # def GetMdsInterval(self, idx):
        processor_edge_vec=[]
        for ec in self.exe_env.lis_edge_clouds:
            processor_edge_vec.append(ec.num_processors)
        sum_i = 0
        inte = []
        for num_p in processor_edge_vec:
            sum_i += num_p
            if sum_i >= pos1 + 1:
                inte.extend([sum_i - num_p, sum_i - 1])
                break
        # return inte
        if pos2>=inte[0] and pos2<=inte[1]:
            return True
        else:
            return False


    def Get_exp_num_trans(self,num_trans,trans_suc):
        exp_num_trans=0
        num_trans+=1
        for i in range(1,num_trans+1):
            exp_num_trans+=math.pow(1-trans_suc,i-1)*trans_suc*i
        return int(exp_num_trans)+1





    def Get_poses(self,red_act):
        poses_1=[]
        for i in range(len(red_act)):
            if red_act[i]==1:
                poses_1.append(i)
        return poses_1




        


    def reset(self):
        # pass
        self.wall_time.reset()
        self.timeline.clear()
        # self.reward_calculator.reset()
        self.lis_mds_selected.clear()
        self.jobs_selected.clear()
        self.exe_env.reset()
        self.lis_cur_jobs.clear()
        self.lis_jobs=generate_lis_jobs(self.np_random,self.timeline,self.wall_time)
        # print(len(self.lis_jobs))
        self.num_md_map=compute_num_md_map(self.lis_jobs)
        # add initial set of jobs in the system
        if args.arrival_mode=='batch_arrival':
            for job in self.lis_jobs:
                self.lis_cur_jobs.append(job)
        elif args.arrival_mode=='poisson_arrival':
            while len(self.timeline)>0 and self.timeline[0].release_time==0:
                self.lis_cur_jobs.append(self.timeline.pop(0))
        self.previous_job=None                        #上一个被调度module所在的job对象
        self.lis_frontier_mds=self.Get_init_frontier_mds()
        print('job.ori_adj_mat: \n',self.lis_cur_jobs[0].ori_adj_mat)

        # print(len(self.lis_frontier_mds),'\n','___',len(self.lis_cur_jobs))



    def seed(self,seed):
        self.np_random.seed(seed)




