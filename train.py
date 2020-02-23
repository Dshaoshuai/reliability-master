import tensorflow as tf
import numpy as np
from params import *
from utils import *
from RL_env import *
from actor_agent import *
from tf_logger import *
from average_reward import *
import multiprocessing as mp
from compute_gradients import *
import time


def invoke_model(actor_agent,obs,exp):
    #parse observation
    lis_cur_jobs,previous_job,jobs_selected,lis_frontier_mds,lis_mds_selected,exe_env=obs

    md_act_prob, redundancy_act, retrans_act, \
    modules_input, processors_input, indg_redep_mat, \
    mds_jobs_vec, front_mds_mat, processor_edge_vec,ret=actor_agent.invoke_model(obs)

    md_idx=np.argmax(md_act_prob)
    module_act=lis_frontier_mds[md_idx]

    # job = module_act.job
    # job_idx = lis_cur_jobs.index(job)
    # md_idx = job.GetIndxMd(module_act)
    # md_job_idx=sum(mds_jobs_vec[:job_idx])+md_idx

    # if any(redundancy_act)==0 or any(redundancy_act)==1:
    #     return module_act,redundancy_act,retrans_act


    #store experience
    exp['modules_inputs'].append(modules_input)
    exp['processors_inputs'].append(processors_input)
    exp['module_act_prob'].append(md_act_prob)
    exp['redundancy_act'].append(redundancy_act)
    exp['retrans_act'].append(retrans_act)
    exp['indg_redep_mat'].append(indg_redep_mat)
    exp['mds_jobs_vec'].append(mds_jobs_vec)
    exp['front_mds_mat'].append(front_mds_mat)
    exp['processor_edge_vec'].append(processor_edge_vec)
    exp['ret'].append(ret)
    # exp['md_job_idx'].append(md_job_idx)

    return module_act,redundancy_act,retrans_act






def train_agent(agent_id,param_queue,reward_queue,adv_queue,gradient_queue):
    # pass
    tf.set_random_seed(agent_id)

    #set up environment
    env=RL_env()

    #gpu configuration
    config=tf.ConfigProto(device_count={'GPU':args.worker_num_gpu},gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=args.master_gpu_fraction))

    sess=tf.Session(config=config)
    actor_agent=ActorAgent(sess,args.module_input_dim,args.processor_input_dim,
                           args.hid_dims,args.output_dim)

    # collect experiences
    # while True:
    (actor_params,seed)=param_queue.get()

    #synchronize model
    actor_agent.set_params(actor_params)

    env.seed(seed)
    env.reset()

    #set up storage for experience
    exp={'modules_inputs':[],'processors_inputs':[],
         'module_act_prob':[],'redundancy_act':[],
         'retrans_act':[],'indg_redep_mat':[],
         'mds_jobs_vec':[],'front_mds_mat':[],'processor_edge_vec':[],
         'reward':[],'wall_time':[],'md_interval':[],'job_reliability':[],
         'job_diff_rel':[],'sign_job_rel':[],'md_job_idx':[],'ret':[]
        # ,'sorted_reward':[],'sorted_md_interval':[]
         }  #'md_job_idx'记录的是每个action选出的module在总md列表中的idx，可据此和front_mds_mat还原出adv

    # try:
    obs=env.observe()
    done=False

    exp['wall_time'].append(env.wall_time.curr_time)

    while not done:
        module_act,redundancy_act,retrans_act=invoke_model(actor_agent,obs,exp)

        # if any(redundancy_act)==0:
        #     continue
        if sum(redundancy_act)==0:
            redundancy_act[0]=1

        obs,reward,done=env.step(module_act,redundancy_act,retrans_act)
        print('done: ',done)

        exp['reward'].append(reward)
        exp['wall_time'].append(module_act.end_time)
        exp['md_interval'].append(module_act.end_time-module_act.earliest_start_time)
        # print(module_act.end_time,' &&& ',module_act.earliest_start_time)
        # print(module_act.lis_processors,' *** \n',module_act.retrans_act)
    # print('(((',exp['md_interval'])

    assert len(exp['modules_inputs'])==len(exp['reward'])
    exp['job_reliability']=[j.reliability_real for j in env.jobs_selected]
    exp['job_diff_rel']=[j.reliability_real-j.reliability_req for j in env.jobs_selected]
    a=[j.reliability_real>=j.reliability_req for j in env.jobs_selected]
    exp['sign_job_rel']=list(np.array(a).astype(int))
    #将exp['reward']中的reward按照调度的module在总mds中的顺序重新排序
    # sorted_md_idx=sorted(exp['md_job_idx'])
    # # sorted_bat_reward=[]
    # # sorted_md_interval=[]
    # for md_idx in sorted_md_idx:
    #     exp['sorted_reward'].append(exp['reward'][exp['md_job_idx'].index(md_idx)])
    #     exp['sorted_md_interval'].append(exp['md_interval'][exp['md_job_idx'].index(md_idx)])
    # assert len(exp['reward'])==len(exp['sorted_reward'])
    reward_queue.put(
        [exp['reward'],exp['wall_time'],exp['md_interval'],
         np.max([j.end_time for j in env.jobs_selected]),
        len(env.jobs_selected),
         [j.make_span for j in env.jobs_selected],
         np.mean([j.make_span for j in env.jobs_selected]),
         exp['job_reliability'],exp['job_diff_rel'],exp['sign_job_rel']
    ])           #分别记录每个action的reward，每个action的结束时间（第一个为开始时间0），每个action的时间间隔，所有作业的最大完成时间（即结束时间），完成job的数量，
                 #job的make span列表，jobs的平均make_span，jobs达到的reliability列表，jobs达到的reliability和需求的可靠性之间的差值，
                 #标志其是否达到可靠性要求(1表示达到要求，0表示没有达到)，记录每个action的md所在总idx
    # print('reward_queue: ',reward_queue.get())
    print('exp[reward]: ',exp['reward'])
    print('exp[wall_time]: ',exp['wall_time'])
    print('exp[md_interval]: ',exp['md_interval'])
    print('max job.end_time: ',np.max([j.end_time for j in env.jobs_selected]))
    print('len(jobs_selected): ',len(env.jobs_selected))
    print('job.make_span: ',[j.make_span for j in env.jobs_selected])
    print('exp[job_reliability]: ',exp['job_reliability'])
    print('exp[job_diff_rel]: ',exp['job_diff_rel'])
    print('exp[sign_job_rel]: ',exp['sign_job_rel'])
    batch_adv=adv_queue.get()
    # if batch_adv is None:
    #     continue

    actor_gradient,loss=compute_actor_gradients(actor_agent,exp,batch_adv)

    gradient_queue.put([actor_gradient,loss])

                


def main():
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    #create result and model folder
    create_folder_if_not_exits(args.result_folder)
    create_folder_if_not_exits(args.model_folder)

    # initialize communication queues
    params_queues = [mp.Queue(1) for _ in range(args.num_agents)]
    reward_queues = [mp.Queue(1) for _ in range(args.num_agents)]
    adv_queues = [mp.Queue(1) for _ in range(args.num_agents)]
    gradient_queues = [mp.Queue(1) for _ in range(args.num_agents)]

    # set up training agents
    agents = []
    for i in range(args.num_agents):
        agents.append(mp.Process(target=train_agent, args=(
            i, params_queues[i], reward_queues[i],
            adv_queues[i], gradient_queues[i])))

    # start training agents
    for i in range(args.num_agents):
        agents[i].start()

    # gpu configuration
    config = tf.ConfigProto(
        device_count={'GPU': args.master_num_gpu},
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=args.master_gpu_fraction))

    sess = tf.Session(config=config)

    # set up actor agent
    actor_agent=ActorAgent(sess,args.module_input_dim,args.processor_input_dim,
                           args.hid_dims,args.output_dim)

    #1: only one training agent
    # agent_id=0
    # train_agent(agent_id,params_queues,reward_queues,gradient_queue)

    # tensorboard logging

    tf_logger = TFLogger(sess, [
        'all_act_loss', 'all_adv_loss', 'all_module_entropy', 'episode_length',
        'average_reward_per_second', 'sum_reward',
        'num_jobs', 'average_job_make_span','job_rel','job_diff_rel','job_ratio_rel'
    ])

    # store average reward for computing differential rewards
    avg_reward_calculator = AveragePerStepReward(
        args.average_reward_storage_size)

    # initialize entropy parameters
    # entropy_weight = args.entropy_weight_init

    # initialize episode reset probability
    reset_prob = args.reset_prob

    # ---- start training process ----
    for ep in range(1, args.num_ep):
        print('training epoch', ep)

        # synchronize the model parameters for each training agent
        actor_params=actor_agent.get_params()

        for i in range(args.num_agents):
            params_queues[i].put([actor_params,args.seed+ep])

        #storage for advantage computation
        # all_rewards,all_diff_times,all_times,\
        # all_batch_end_time,all_num_finished_jobs,all_avg_job_duration\
        # =[],[],[],[],[],[]

        all_rewards,all_md_intervals,all_finish_time,all_num_finished_jobs,all_job_make_span,\
        all_avg_job_make_span,all_job_rel,all_job_diff_rel,all_sign_job_rel,all_times=\
            [],[],[],[],[],[],[],[],[],[]

        t1=time.time()
        for i in range(args.num_agents):
            result=reward_queues[i].get()
            print('result[0]: ',result[0])

            if result is None:
                any_agent_panic=True
                continue
            else:
                batch_reward,batch_time,batch_md_interval,batch_finish_time,\
                num_finished_jobs,batch_job_make_span,avg_job_make_span,batch_job_rel,\
                batch_job_diff_rel,batch_sign_job_rel=result
                print('len(batch_reward): ',len(batch_reward))

            # diff_time=np.array(batch_time[1:])-np.array(batch_time[:-1])

            all_rewards.append(batch_reward)
            all_md_intervals.append(batch_md_interval)
            all_finish_time.append(batch_finish_time)
            all_num_finished_jobs.append(num_finished_jobs)
            all_job_make_span.append(batch_job_make_span)
            all_avg_job_make_span.append(avg_job_make_span)
            all_job_rel.append(batch_job_rel)
            all_job_diff_rel.append(batch_job_diff_rel)
            all_sign_job_rel.append(batch_sign_job_rel)
            all_times.append(batch_time[1:])
            # all_sorted_rewards

            # all_md_job_idx.append(md_job_idx)

            # all_rewards.append(batch_reward)
            # all_diff_times.append(diff_time)
            # all_batch_end_time.append(batch_end_time)
            # all_num_finished_jobs.append(num_finished_jobs)
            # all_avg_job_duration.append(avg_job_duration)

            avg_reward_calculator.add_list_filter_zero(batch_reward,batch_md_interval)

        t2=time.time()

        print('got reward from workers ',t2-t1,' seconds.')

        # compute differential reward
        all_cum_reward=[]
        avg_per_step_reward = avg_reward_calculator.get_avg_per_step_reward()
        print('avg_per_step_reward: ',avg_per_step_reward)
        for i in range(args.num_agents):
            if args.diff_reward_enabled:
                # differential reward mode on
                rewards = np.array([r - avg_per_step_reward * t for \
                                    (r, t) in zip(all_rewards[i], all_md_intervals[i])])
            else:
                # regular reward
                rewards = np.array([r for \
                                    (r, t) in zip(all_rewards[i], all_md_intervals[i])])
            cum_reward = discount(rewards, args.gamma)

            all_cum_reward.append(cum_reward)

        # compute baseline
        # baselines = get_piecewise_linear_fit_baseline(all_cum_reward, all_times)

        # give worker back the advantage
        for i in range(args.num_agents):
            batch_adv = all_cum_reward[i] #- baselines[i]
            # batch_adv = np.reshape(batch_adv, [len(batch_adv), 1])
            print('len(batch_adv): ',batch_adv)
            adv_queues[i].put(batch_adv)

        t3 = time.time()
        print('advantage ready', t3 - t2, 'seconds')

        actor_gradients=[]
        all_act_loss=[]
        all_adv_loss=[]
        all_module_entropy=[]
        all_red_loss=[]

        for i in range(args.num_agents):
            (actor_gradient, loss) = gradient_queues[i].get()
            actor_gradients.append(actor_gradient)
            all_act_loss.append(loss[0])
            all_adv_loss.append(loss[1])
            all_module_entropy.append(loss[2])
            all_red_loss.append(loss[3])

        t4 = time.time()
        print('worker send back gradients', t4 - t3, 'seconds')

        actor_agent.apply_gradients(
            aggregate_gradients(actor_gradients), args.lr)

        t5 = time.time()
        print('apply gradient', t5 - t4, 'seconds')

        tf_logger.log(ep,[
            np.mean(all_act_loss),
            np.mean(all_adv_loss),
            np.mean(all_module_entropy),
            np.mean(all_red_loss),
            np.mean([len(b) for b in baselines]),
            avg_per_step_reward * args.reward_scale,
            np.mean([cr[0] for cr in all_cum_reward]),
            np.mean(all_num_finished_jobs),
            np.mean(all_avg_job_make_span),
            np.mean(all_job_rel),
            np.mean(all_job_diff_rel),
            np.mean([float(sum(b)/len(b)) for b in all_sign_job_rel])
        ])
        print('ep: ',ep)

        if ep % args.model_save_interval == 1:
            actor_agent.save_model(args.model_folder + \
                'model_ep_' + str(ep))
    sess.close()

if __name__=='__main__':
    main()