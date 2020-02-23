from edge_cloud_env.module import *
from edge_cloud_env.job_dag import *
from edge_cloud_env.state import *
import numpy as np
from edge_cloud_env.edge import *
from params import *

def generate_job(job_idx,reliability_req,release_time,adj_mat,is_state,lis_is_stat_mds,perc_stat_mds,num_stat_mds,total_stat_volume,avg_stat_volume,num_mds,total_mds_op_volume,
                 avg_md_op_volume,num_edges,total_edges_data_volume,avg_edge_data_volume,avg_md_indegrees):
    # pass
    job=JobDAG(job_idx,num_mds+2,is_state,None,avg_md_op_volume,adj_mat,perc_stat_mds,avg_stat_volume,lis_is_stat_mds,reliability_req,release_time
               ,num_edges,total_edges_data_volume,avg_edge_data_volume,avg_md_indegrees,num_stat_mds,total_stat_volume,total_mds_op_volume)                                                                             #注：这里的num_mds为包含entry module和exit module的数量
    md_idx=0
    mds_set=[]
    # print(adj_mat)
    num_mds=len(lis_is_stat_mds)           #当生成job时，module总数包含entry module和exit module，即args.num_mods+2
    for stat_sign in lis_is_stat_mds:
        # pass
        lis_pre_mds=[]
        for i in range(num_mds):
            if adj_mat[i,md_idx]==1 and i<len(mds_set):
                # print(i)
                lis_pre_mds.append(mds_set[i])
        if stat_sign:
            stat=state(None,None,job,np.random.normal(avg_stat_volume,0.5))
        else:
            stat=None
        md=module(md_idx,job,np.random.normal(avg_md_op_volume,0.5),lis_pre_mds,None,stat_sign,stat,None)
        if stat_sign:
            stat.Set_sour_ori_md(md)
        mds_set.append(md)
        md_idx+=1

    job.ori_adj_mat_edges=np.zeros((num_mds,num_mds),dtype=edge)
    for i in range(num_mds):
        # print('i: ',i)
        if i == num_mds - 1:   #exit module
            ed_idx = mds_set[i].idx
            mds_set[i].Set_Exit_Edge(edge(ed_idx, job, mds_set[i], mds_set[i], None, None, None,np.random.normal(avg_edge_data_volume, 0.5)))
        if mds_set[i].lis_pre_mods:    #mid modules
            for pre_md in mds_set[i].lis_pre_mods:
                ed_idx=pre_md.idx
                edg=edge(ed_idx,job,pre_md,mds_set[i],None,None,None,np.random.normal(avg_edge_data_volume,0.5))
                job.ori_adj_mat_edges[mds_set.index(pre_md),i]=edg
        else:   #此module为entry module
            # print('i-th md: ',i)
            # print('entry_md lis_pre_mods: ',mds_set[i].lis_pre_mods)
            ed_idx=mds_set[i].idx
            mds_set[i].Set_Entry_Edge(edge(ed_idx,job,mds_set[i],mds_set[i],None,None,None,np.random.normal(avg_edge_data_volume,0.5)))
            mds_set[i].entry_edge.Set_Release_time(job.release_time)
    # print('job.ori_adj_mat_edges: ',job.ori_adj_mat_edges)



    for i in range(num_mds):
        for j in range(num_mds):
            if adj_mat[i][j]==1:
                mds_set[i].lis_sub_mods.append(mds_set[j])

    job.Set_lis_mds(mds_set)
    return job

def generate_lis_jobs(np_random,timeline,wall_time):
    # pass
    lis_jobs=[]
    release_time=0
    for job_x in range(args.num_init_jobs+args.num_stream_jobs):
        job_size=args.lis_num_mds[np_random.randint(len(args.lis_num_mds))]
        job_idx=np_random.randint(args.num_tem_jobs)
        job_path=args.job_folder+job_size+'/'+job_size+'_'+str(job_idx)+'.npy'
        config_file=np.load(job_path,allow_pickle=True).item()
        job=generate_job(job_idx,args.reliability_req,None,config_file['adj_mat'],config_file['is_state'],config_file['lis_is_stat_mds'],config_file['perc_stat_mds'],
                         config_file['num_stat_mds'],config_file['total_stat_volume'],config_file['avg_stat_volume'],config_file['num_mds'],config_file['total_mds_op_volume'],
                         config_file['avg_md_op_volume'],config_file['num_edges'],config_file['total_edges_data_volume'],config_file['avg_edge_data_volume'],config_file['avg_md_indegrees'])
        if job_x>=args.num_init_jobs:
            release_time += int(np_random.exponential(args.stream_interval))
            job.Set_release_time(release_time)
            timeline.append(job)
        else:
            # print(job_idx,args.num_init_jobs)
            job.Set_release_time(release_time)
            lis_jobs.append(job)
    # timeline=sorted(timeline,key=lambda s:s.rel)
    # print(args.num_init_jobs,len(lis_jobs))
    return lis_jobs
    # for job_idx in range(args.num_init_jobs,args.num_init_jobs+args.num_stream_jobs):
    #     release_time+=int(np_random.exponential(args.stream_interval))
    #     job_size=args.lis_num_mds[np_random.randint(len(args.lis_num_mds))]
    #     job_idx=str(np_random.randint(args.num_tem_jobs))
    #     job_path=args.job_folder+job_size+'/'+job_size+'_'+job_idx+'.npy'
    #     config_file=np.load(job_path,allow_pickle=True).item()
    #     job=







