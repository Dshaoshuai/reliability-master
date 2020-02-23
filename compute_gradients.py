from params import *
import numpy as np
from tf_op import *

def compute_actor_gradients(actor_agent,exp,batch_adv):
    # batch size
    all_gradients=[]
    all_loss=[[],[],[],[]]
    len_1=len(exp['modules_inputs'])
    # print('len(exp[modules_inputs]): ',len(exp['modules_inputs']))
    # print()
    args.batch_size=int(len_1/2)
    ba_start=np.random.randint(len_1-args.batch_size+1)
    for b in range(ba_start,ba_start+args.batch_size):
        modules_input=exp['modules_inputs'][b]
        processors_input=exp['processors_inputs'][b]
        module_act_prob=exp['module_act_prob'][b]
        redundancy_act=exp['redundancy_act'][b]
        ret=exp['ret'][b]
        indg_redep_mat=exp['indg_redep_mat'][b]
        mds_jobs_vec=exp['mds_jobs_vec'][b]
        front_mds_mat=exp['front_mds_mat'][b]
        processor_edge_vec=exp['processor_edge_vec'][b]
        adv=batch_adv[b]
        # print('b: ',b)
        # print('exp[modules_inputs]: ',exp['modules_inputs'])
        # print('adv: ',adv)
        # print('modules_input: ',modules_input)
        print('shape of modules_input: ',modules_input.shape,modules_input)
        print('shape of processors_input: ',processors_input.shape,processors_input)
        print('shape of module_act_prob: ',module_act_prob.shape,module_act_prob)
        print('shape of redundancy_act: ',redundancy_act.shape,redundancy_act)
        print('shape of retrans_act: ',ret.shape,ret)
        print(adv)


        act_gradients,loss=actor_agent.get_gradients(
            modules_input,processors_input,module_act_prob,redundancy_act,
            ret,adv
        )
        # act_gradients, loss = actor_agent.get_gradients(
        #     modules_input, processors_input)
        print('act_gradients: ',act_gradients)
        # print('done')
        all_gradients.append(act_gradients)
        all_loss[0].append(loss[0])
        all_loss[1].append(loss[1])
        all_loss[2].append(loss[2])
        # all_loss[3].append(loss[3])
    all_loss[0]=np.sum(all_loss[0])   # all act_loss
    all_loss[1]=np.sum(all_loss[1])   # all adv_loss
    all_loss[2]=np.sum(all_loss[2])   # all module_entropy
    # all_loss[3]=np.sum(all_loss[3])   # all red_loss

    gradients=aggregate_gradients(all_gradients)

    return gradients,all_loss

