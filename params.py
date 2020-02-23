import argparse

parser=argparse.ArgumentParser(description='jobs_env_RL')

# -- Basic --
parser.add_argument('--job_folder',type=str,default='./job_dags/',help='job folder path \
                                                     (default: C:/Users/25714/PycharmProjects/reliability/job_dags/)')
parser.add_argument('--seed',type=int,default=42,help='random seed (default: 42)')
parser.add_argument('--result_folder',type=str,default='./results',help='result folder path (default: ./results)')
parser.add_argument('--model_folder',type=str,default='./models',help='model folder path (default: ./models)')

# -- jobs generation --
parser.add_argument('--num_init_jobs',type=int,default=1,help='Number of initial jobs in system (default: 10)')
parser.add_argument('--tpch_num',type=int,default=20,help='number of TPCH queries (default: 20)')

parser.add_argument('--lis_num_mds',type=str,default=['10','30'],nargs='+',help='number list of modules for job generation (default: [10,30])')
parser.add_argument('--num_tem_jobs',type=int,default=10,help='number of template jobs for generation (default: 10)')

parser.add_argument('--num_stream_jobs',type=int,default=0,help='number of streaming jobs (default: 50)')
# parser.add_argument('--stream_interval',type=int,default=15,help='inter job arrival time in seconds obeying exponential distribution (default : 15s)')

parser.add_argument('--max_num_stream_jobs',type=int,default=500,help='maximum number of streaming jobs (default: 500)')
parser.add_argument('--stream_interval',type=int,default=25,help='inter job arrival time in seconds (default: 25s)')
parser.add_argument('--avg_md_op_volume',type=int,default=5,help='average operation volume of each module (default: 5MB)')
parser.add_argument('--avg_edge_data_volume',type=int,default=2,help='average data volume of each edge (default: 2MB)')
parser.add_argument('--perc_stat_mds',type=float,default=0.5,help='the percentage of stateful modules in stateful jobs')
parser.add_argument('--avg_stat_volume',type=int,default=2,help='the average state volume of each state (default: 2MB)')
parser.add_argument('--num_pre_edge_md',type=int,default=3,help='the number of edges for each module (default: 3)')
parser.add_argument('--num_mods',type=int,default=10,help='the number of modules (default: 30)')   #不包含entry module和exit module
parser.add_argument('--avg_indegrees',type=int,default=3,help='the number of indegree of each module (default: 3)')
parser.add_argument('--reliability_req',type=float,default=0.999,help='the reliability requirement for each job (default: 0.999)')


# -- exe environment --
parser.add_argument('--num_edge_clouds',type=int,default=2,help='Number of total edge clouds (default: 10)')
parser.add_argument('--num_processors',type=int,default=7,help='number of total processors')
parser.add_argument('--lis_dist_ec_grade',type=int,default=[0,50],nargs='+',help='the\
                       distance list between mobile device and each edge clouds from near to far')
parser.add_argument('--dic_num_ec_grade',type=int,default={0:1,50:1},help='the number dic of ecs in each distance grade')
parser.add_argument('--dic_class_edge_clouds',type=int,default={'A':6,'C':1},help='the dic of each ec type and its number of processors')
parser.add_argument('--dic_num_ec_class_total',type=int,default={'A':1,'C':1},help='the dic of each type of ec and its total number')
parser.add_argument('--dic_num_ec_class_grade',type=int,default={0:{'A':0,'C':1},50:{'A':1,'C':0}},help=\
                    'the dic of different ec type number in each distance')
parser.add_argument('--cap_processor',type=int,default=5,help='the processing capacity of each processor (default: 5MBps)')
parser.add_argument('--bw_in_channel',type=int,default=1,help='the bandwidth of each net-channel (default: 1MBps)')
parser.add_argument('--min_num_channels',type=int,default=1,help='minimum number of net-channels (default: 1)')
parser.add_argument('--max_num_channels',type=int,default=10,help='maximum number of net-channels (default: 10)')
parser.add_argument('--avg_processor_err_prob',type=float,default=0.98,help='(average) error probability of each processor')
parser.add_argument('--avg_bw_err_prob',type=float,default=0.98,help='(average) error probability of each net-channel')
parser.add_argument('--propa_speed',type=float,default=3.0*pow(10,8),help='the propagation speed (m/s)')
parser.add_argument('--arrival_mode',type=str,default='batch_arrival',help='the arrival mode of jobs (default: batch_arrival)(alternative: poisson_arrival)')


# -- Learning --
parser.add_argument('--worker_num_gpu',type=int,default=0,help='number of GPU cores used in worker (default: 0)')
parser.add_argument('--master_gpu_fraction',type=float,default=0.5,help='fraction of memory master uses in GPU (default: 0.5)')
parser.add_argument('--module_input_dim',type=int,default=18,help='module input dimensions of graph embedding (default:18)')
parser.add_argument('--processor_input_dim',type=int,default=13,help='processor input dimensions of graph embedding (default:27)')
parser.add_argument('--hid_dims',type=int,default=[16,8],nargs='+',help='hidden dimensions throughout graph embedding (default: [16,8])')
parser.add_argument('--output_dim',type=int,default=8,help='output dimensions throughout graph embedding (default: 8)')
# parser.add_argument('--exec_cap',type=int,default=)
parser.add_argument('--max_penalty',type=float,default=10,help='the maximum penalty for redundancy & deployment (default: 10)')
parser.add_argument('--num_saved_models',type=int,default=100,
                    help='Number of models to keep (default:100)')
parser.add_argument('--saved_model',type=str,default=None,
                    help='Path to the saved tf model (default: None)')
parser.add_argument('--average_reward_storage_size', type=int, default=1000,
                    help='Storage size for computing average reward (default: 100000)')
parser.add_argument('--entropy_weight_init', type=float, default=1,
                    help='Initial exploration entropy weight (default: 1)')
parser.add_argument('--reset_prob', type=float, default=0,
                    help='Probability for episode to reset (after x seconds) (default: 0)')
parser.add_argument('--num_ep', type=int, default=10000000,
                    help='Number of training epochs (default: 10000000)')
parser.add_argument('--num_agents', type=int, default=1,
                    help='Number of parallel agents (default: 1)')
parser.add_argument('--end_time_scale',type=float,default=100.0,help='scale the end_time of md to some normal values (default: 100.0)')
parser.add_argument('--make_span_scale',type=float,default=500.0,help='scale the make span of the job to some normal values (default: 500.0)')
parser.add_argument('--batch_size',type=int,default=50,help='batch size for training (default: 50)')
parser.add_argument('--diff_reward_enabled', type=int, default=0,
                    help='Enable differential reward (default: 0)')
parser.add_argument('--gamma', type=float, default=1,
                    help='discount factor (default: 1)')

parser.add_argument('--model_save_interval', type=int, default=100,
                    help='Interval for saving Tensorflow model (default: 1000)')
parser.add_argument('--master_num_gpu', type=int, default=0,
                    help='Number of GPU cores used in master (default: 0)')



args = parser.parse_args()