import argparse

def create_parser():
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Script to test KT')
    parser.add_argument('--name', default='demo',help='Name for the experiment')
    parser.add_argument('--nodes', default='', help='slurm nodes for the experiment')
    parser.add_argument('--slurm_partition', default='',
                        help='slurm partitions for the experiment')
    # Basic Parameters
    parser.add_argument('--model_type', type=str,
                        default='B', help='type')
    parser.add_argument('--dataset', type=str,
                        default='assist2009_pid', help='type')
    parser.add_argument('--dropout', type=float, default=0.25, help='type')
    parser.add_argument('--lr', type=float, default=0.1, help='type')
    parser.add_argument('--batch_size', type=int, default=384, help='type')
    parser.add_argument('--epochs', type=int, default=100, help='type')
    parser.add_argument('--fold', type=int, default=1, help='type')
    parser.add_argument('--model', type=str, default='lstm', help='type')
    parser.add_argument('--setup', type=str, default='kt', help='type')
    parser.add_argument('--head', type=int, default=8, help='type')
    parser.add_argument('--neptune', action='store_true')
    parser.add_argument('--seed', type=int, default=221, help='type')
    parser.add_argument('--file_name', type=str,
                        default='', help='type')
    parser.add_argument('--hash', type=str,
                        default='', help='type')
    parser.add_argument('--sample_num', type=int,
                        default=1, help='type')
    parser.add_argument('--isBayes', action='store_true')
    parser.add_argument('--root_path', type=str, default='/content/sample/')
    parser.add_argument('--skill_item', type=int,
                        default=1, help='type')
    parser.add_argument('--is_rasch', type=int,
                        default=0, help='type')
    parser.add_argument('--l2', type=float,
                        default=1e-5, help='l2 penalty for difficulty')
    parser.add_argument('--step_size', type=int,
                        default=15, help='type')
    parser.add_argument('--conv_nlayer', type=int,
                        default=1, help='type')
    parser.add_argument('--kernel_size', type=int,
                        default=2, help='kernel size for 1dconv')
    
                

    params = parser.parse_args()
    if params.dataset=="Eddi":
        params.data_name = "Eddi"
        params.n_question = 27800+1
        params.n_subject = 1200+1
        # params.n_question = 110+1
        # params.n_subject = 16891+1
        params.hidden_dim=100
        params.q_dim=50
        params.s_dim=50
        params.memory_size=50
        params.input_path = "./data/Eddi/"
        params.seqlen = 200
        # params.seqlen = 1000
    if params.dataset=="junyi":
        params.data_name = "junyi"
        params.n_question = 800+1
        params.n_subject = 800+1
        # params.n_question = 110+1
        # params.n_subject = 16891+1
        params.hidden_dim=100
        params.q_dim=50
        params.s_dim=50
        params.memory_size=50
        params.input_path = "./data/junyi/"
        # params.seqlen = 200
        params.seqlen = 1000
    if params.dataset=="assist2009_pid":
        params.data_name = "assist2009_pid"
        params.n_question = 16891+1
        params.n_subject = 110+1
        # params.n_question = 110+1
        # params.n_subject = 16891+1
        # params.hidden_dim=100
        # params.q_dim=50
        # params.s_dim=50
        # params.memory_size=50
        params.hidden_dim=100
        params.q_dim=50
        params.s_dim=50
        params.memory_size=100
        params.input_path = "./data/assist2009_pid/"
        params.seqlen = 200
    if params.dataset=="assist2009_all":
        "json dataset"
        params.data_name = "assist2009_all"
        params.n_question = 207348+1
        params.n_subject = 400
        # params.n_question = 110+1
        # params.n_subject = 16891+1
        params.hidden_dim=100
        params.q_dim=50
        params.s_dim=50
        params.memory_size=50
        params.input_path = "./data/assist2009_all/"
        params.seqlen = 200
    if params.dataset=="assist2009_updated":
        params.data_name = "assist2009_updated"
        # params.n_question = 16891+1
        params.n_question = 110+1
        params.n_subject = 110+1
        params.hidden_dim=100
        params.q_dim=50
        params.s_dim=50
        params.memory_size=50
        params.input_path = "./data/assist2009_updated/"
        params.seqlen = 200
    elif params.dataset=="assist2017_pid":
        params.data_name = "assist2017_pid"
        params.n_question = 3162+1
        # params.n_question = 102+1
        params.n_subject = 102+1
        params.hidden_dim=100
        params.q_dim=50
        params.s_dim=50
        params.memory_size=50
        # params.memory_size=100
        params.input_path = "./data/assist2017_pid/"
        params.seqlen = 200
        params.step_size = 50
    elif params.dataset=="assist2017":
        params.data_name = "assist2017"
        params.n_question = 3162+1
        params.n_subject = 102+1
        params.hidden_dim=100
        params.q_dim=50
        params.s_dim=50
        params.memory_size=50
        params.input_path = "./data/assist2017/"
        params.seqlen = 200
    elif params.dataset=="statics":
        params.data_name = "statics"
        params.n_question = 3162+1
        params.n_subject = 3162+1
        params.hidden_dim=50
        params.q_dim=50
        params.s_dim=50
        params.input_path = "./data/statics/"
        params.seqlen = 200
        params.memory_size=50
        params.step_size = 30
    elif params.dataset=="assist2015":
        params.data_name = "assist2015"
        params.n_question = 100+1
        params.n_subject = 100+1
        params.hidden_dim=50
        params.q_dim=50
        params.s_dim=50
        params.memory_size=50
        params.seqlen = 200
        params.input_path = './data/assist2015/'
    elif params.dataset=="simu_item300_learner2000_sigma0.1":
        params.data_name = "simu_item300_learner2000_sigma0.1"
        params.n_question = 300+1
        params.n_subject = 1+1
        params.hidden_dim=100
        params.q_dim=50
        params.s_dim=50
        params.memory_size=50
        params.input_path = "./data/simu_item300_learner2000_sigma0.1/"
        params.seqlen = 300
    elif params.dataset=="simu_item300_learner2000_sigma0.3":
        params.data_name = "simu_item300_learner2000_sigma0.3"
        params.n_question = 300+1
        params.n_subject = 1+1
        params.hidden_dim=100
        params.q_dim=50
        params.s_dim=50
        params.memory_size=50
        params.input_path = "./data/simu_item300_learner2000_sigma0.3/"
        params.seqlen = 300
    elif params.dataset=="simu_item300_learner2000_sigma0.5":
        params.data_name = "simu_item300_learner2000_sigma0.5"
        params.n_question = 300+1
        params.n_subject = 1+1
        params.hidden_dim=100
        params.q_dim=50
        params.s_dim=50
        params.memory_size=50
        params.input_path = "./data/simu_item300_learner2000_sigma0.5/"
        params.seqlen = 300
    elif params.dataset=="simu_item300_learner2000_sigma1.0":
        params.data_name = "simu_item300_learner2000_sigma1.0"
        params.n_question = 300+1
        params.n_subject = 1+1
        params.hidden_dim=100
        params.q_dim=50
        params.s_dim=50
        params.memory_size=50
        params.input_path = "./data/simu_item300_learner2000_sigma1.0/"
        params.seqlen = 300
    elif params.dataset=="simu_item200_learner2000_sigma0.1":
        params.data_name = "simu_item200_learner2000_sigma0.1"
        params.n_question = 200+1
        params.n_subject = 1+1
        params.hidden_dim=100
        params.q_dim=50
        params.s_dim=50
        params.memory_size=50
        params.input_path = "./data/simu_item200_learner2000_sigma0.1/"
        params.seqlen = 200
    elif params.dataset=="simu_item200_learner2000_sigma0.3":
        params.data_name = "simu_item200_learner2000_sigma0.3"
        params.n_question = 200+1
        params.n_subject = 1+1
        params.hidden_dim=100
        params.q_dim=50
        params.s_dim=50
        params.memory_size=50
        params.input_path = "./data/simu_item200_learner2000_sigma0.3/"
        params.seqlen = 200
    elif params.dataset=="simu_item200_learner2000_sigma0.5":
        params.data_name = "simu_item200_learner2000_sigma0.5"
        params.n_question = 200+1
        params.n_subject = 1+1
        params.hidden_dim=100
        params.q_dim=50
        params.s_dim=50
        params.memory_size=50
        params.input_path = "./data/simu_item200_learner2000_sigma0.5/"
        params.seqlen = 200
    elif params.dataset=="simu_item200_learner2000_sigma1.0":
        params.data_name = "simu_item200_learner2000_sigma1.0"
        params.n_question = 200+1
        params.n_subject = 1+1
        params.hidden_dim=100
        params.q_dim=50
        params.s_dim=50
        params.memory_size=50
        params.input_path = "./data/simu_item200_learner2000_sigma1.0/"
        params.seqlen = 200
    elif params.dataset=="simu_item100_learner2000_sigma0.1":
        params.data_name = "simu_item100_learner2000_sigma0.1"
        params.n_question = 100+1
        params.n_subject = 1+1
        params.hidden_dim=100
        params.q_dim=50
        params.s_dim=50
        params.memory_size=50
        params.input_path = "./data/simu_item100_learner2000_sigma0.1/"
        params.seqlen = 100
    elif params.dataset=="simu_item100_learner2000_sigma0.3":
        params.data_name = "simu_item100_learner2000_sigma0.3"
        params.n_question = 100+1
        params.n_subject = 1+1
        params.hidden_dim=100
        params.q_dim=50
        params.s_dim=50
        params.memory_size=50
        params.input_path = "./data/simu_item100_learner2000_sigma0.3/"
        params.seqlen = 100
    elif params.dataset=="simu_item100_learner2000_sigma0.5":
        params.data_name = "simu_item100_learner2000_sigma0.5"
        params.n_question = 100+1
        params.n_subject = 1+1
        params.hidden_dim=100
        params.q_dim=50
        params.s_dim=50
        params.memory_size=50
        params.input_path = "./data/simu_item100_learner2000_sigma0.5/"
        params.seqlen = 100
    elif params.dataset=="simu_item100_learner2000_sigma1.0":
        params.data_name = "simu_item100_learner2000_sigma1.0"
        params.n_question = 100+1
        params.n_subject = 1+1
        params.hidden_dim=100
        params.q_dim=50
        params.s_dim=50
        params.memory_size=50
        params.input_path = "./data/simu_item100_learner2000_sigma1.0/"
        params.seqlen = 100
    elif params.dataset=="simu_item50_learner2000_sigma0.1":
        params.data_name = "simu_item50_learner2000_sigma0.1"
        params.n_question = 50+1
        params.n_subject = 1+1
        params.hidden_dim=100
        params.q_dim=50
        params.s_dim=50
        params.memory_size=50
        params.input_path = "./data/simu_item50_learner2000_sigma0.1/"
        params.seqlen = 50
    elif params.dataset=="simu_item50_learner2000_sigma0.3":
        params.data_name = "simu_item50_learner2000_sigma0.3"
        params.n_question = 50+1
        params.n_subject = 1+1
        params.hidden_dim=100
        params.q_dim=50
        params.s_dim=50
        params.memory_size=50
        params.input_path = "./data/simu_item50_learner2000_sigma0.3/"
        params.seqlen = 50
    elif params.dataset=="simu_item50_learner2000_sigma0.5":
        params.data_name = "simu_item50_learner2000_sigma0.5"
        params.n_question = 50+1
        params.n_subject = 1+1
        params.hidden_dim=100
        params.q_dim=50
        params.s_dim=50
        params.memory_size=50
        params.input_path = "./data/simu_item50_learner2000_sigma0.5/"
        params.seqlen = 50
    elif params.dataset=="simu_item50_learner2000_sigma1.0":
        params.data_name = "simu_item50_learner2000_sigma1.0"
        params.n_question = 50+1
        params.n_subject = 1+1
        params.hidden_dim=100
        params.q_dim=50
        params.s_dim=50
        params.memory_size=50
        params.input_path = "./data/simu_item50_learner2000_sigma1.0/"
        params.seqlen = 50
    elif params.dataset=='eedi':
        params.wait = 20
        params.epoch = 400
    else:
        params.wait = 10
        params.epoch = 200

    params.n_quiz=params.n_group = 0
    #
    if 'ednet' in params.dataset:
        params.n_question = 13169
        params.n_subject = 189
        params.n_user = 784310
        params.max_len = 7
        
    if params.dataset == 'coda':
        params.n_question = 27613
        params.n_subject = 389
        params.n_user = 118971
        #n_quiz = 17305
        #n_group = 11844
        params.max_len = 6
    if params.dataset == 'eedi':
        params.n_question = 948
        params.n_subject = 389
        params.n_user = 6148
        params.max_len = 2

    params.out_dir = params.root_path+params.data_name+str(params.fold)+"_si"+str(params.skill_item)+"/"
    params.save_name = params.data_name+str(params.fold)+"_si"+str(params.skill_item)
    return params
