def compute_num_md_map(lis_jobs):
    num_md_map={}
    num=0
    for job in lis_jobs:
        for md in job.lis_mds:
            num_md_map[num]=md
            num+=1
    return num_md_map