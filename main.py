"""
@author: AlbertTan

Part of the code in <utils.py> is from SimGNN@benedekrozemberczki
"""

import os
import argparse
import random
from datetime import datetime
import torch

from trainer import GEDTrainer
from utils import create_dir_if_not_exists, tab_printer, log_args


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--wandb_activate",
                        default=False)

    parser.add_argument("--seed",
                        default=2022)

    parser.add_argument('--norm_dist_method',
                        default='neg^exp')

    parser.add_argument("--explain_study",
                        default=False)
    parser.add_argument("--load_model",
                        default=False)
    parser.add_argument('--case_study',
                        default=False)
    parser.add_argument("--loaded_model_signature",
                        default="")

    parser.add_argument('--dataset', type=str, help='indicate the specific data set',
                        default='IMDBMulti')
    parser.add_argument('--gpu_index', type=str, help="gpu index to use",
                        default='0')

    # ----- hyper parameters -------

    parser.add_argument('--sim_mat_learning_ablation',
                        help='replace SimCNN with SimMatPooling',
                        default=False)

    parser.add_argument('--msa_bias',
                        default=True)

    # GCN layers
    parser.add_argument("--embedding_size",
                        default=32)
    parser.add_argument("--graph_transformer_active",
                        default=True)
    parser.add_argument("--encoder_ffn_size",
                        default=128)

    parser.add_argument("--GT_res",
                        default=True)
    parser.add_argument("--share_qk",
                        default=True)
    parser.add_argument("--use_dist",
                        default=True)
    parser.add_argument('--dist_decay',
                        default=0)
    parser.add_argument("--dist_start_decay", type=float,
                        default=0.5)

    parser.add_argument('--encoder_mask',
                        default=False)
    parser.add_argument('--interaction_mask',
                        default=False)
    parser.add_argument('--align_mask',
                        default=False)
    parser.add_argument('--cnn_mask',
                        default=False)

    # GraphTransformer params
    parser.add_argument("--n_heads", type=int,
                        default=8)
    parser.add_argument('--channel_align',
                        default=True)
    parser.add_argument("--n_channel_transformer_heads", type=int,
                        default=4)
    parser.add_argument("--channel_ffn_size",
                        default=128)

    # conv params
    parser.add_argument("--conv_channels_0",
                        default=32)
    parser.add_argument("--conv_channels_1",
                        default=64)
    parser.add_argument("--conv_channels_2",
                        default=1)
    parser.add_argument("--conv_channels_3",
                        default=256)

    parser.add_argument("--conv_l_relu_slope",
                        default=0.33)
    parser.add_argument("--conv_dropout",
                        default=0.1)

    parser.add_argument("--pooling_res",
                        default=20)

    # training parameters
    parser.add_argument('--iterations', type=int, help='number of training epochs',
                        default=10000)
    parser.add_argument('--iter_val_start', type=int,
                        default=9000)
    parser.add_argument('--patience',
                        default=100)
    parser.add_argument('--iter_val_every', type=int,
                        default=1)

    parser.add_argument("--batch_size", type=int, help="Number of graph pairs per batch.",
                        default=128)
    parser.add_argument("--lr", type=float, help="Learning rate.",
                        default=5e-4)
    parser.add_argument("--lr_reduce_factor",
                        default=0.5)
    parser.add_argument("--lr_schedule_patience",
                        default=800)
    parser.add_argument("--min_lr",
                        default=1e-6)
    parser.add_argument("--dropout", type=float, help="Dropout probability.",
                        default=0.1)
    parser.add_argument("--weight_decay", type=float,
                        default=0)

    parser.add_argument("--temp",
                        default={'cur_iter': 0})

    # experiment settings
    parser.add_argument('--log_path', type=str, help='path for log file',
                        default='./GSTLogs')
    parser.add_argument('--repeat_run', type=int, help='indicated the index of repeat run',
                        default=0)
    parser.add_argument('--data_dir', type=str, help='root directory for the data',
                        default='./datasets/')

    parser.add_argument('--GNN',
                        default='GCN')

    parsed_args = parser.parse_args()

    if parsed_args.dataset == 'LINUX':
        parsed_args.embedding_size = 32
        parsed_args.n_channel_transformer_heads = 4
    elif parsed_args.dataset == 'AIDS700nef':
        parsed_args.embedding_size = 128
        parsed_args.n_channel_transformer_heads = 4
    elif parsed_args.dataset == 'IMDBMulti':
        parsed_args.embedding_size = 32
        parsed_args.n_channel_transformer_heads = 8

    if parsed_args.load_model:
        model_sig = parsed_args.loaded_model_signature
        if model_sig.find('LINUX') != -1:
            parsed_args.dataset = 'LINUX'
            parsed_args.embedding_size = 32
            parsed_args.n_channel_transformer_heads = 4
        elif model_sig.find('AIDS700nef') != -1:
            parsed_args.dataset = 'AIDS700nef'
            parsed_args.embedding_size = 128
            parsed_args.n_channel_transformer_heads = 4
        else:
            parsed_args.dataset = 'IMDBMulti'
            parsed_args.embedding_size = 32
            parsed_args.n_channel_transformer_heads = 8

    return parsed_args


if __name__ == '__main__':
    parsed_args = get_args()

    if parsed_args.wandb_activate:
        import wandb
        wandb.init(project=parsed_args.dataset, entity="<YourEntityName>", config=parsed_args)
        parsed_args = wandb.config

    torch.manual_seed(parsed_args.seed)
    if torch.cuda:
        torch.cuda.manual_seed(parsed_args.seed)

    parsed_args.device_count = torch.cuda.device_count()
    if parsed_args.device_count == 1:
        parsed_args.gpu_index = '0'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parsed_args.device = device

    create_dir_if_not_exists(parsed_args.log_path)
    log_root_dir = parsed_args.log_path
    signature = parsed_args.dataset + '_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") \
                + '-' + str(random.randint(100000, 1000000))
    current_run_dir = os.path.join(log_root_dir, signature)
    create_dir_if_not_exists(current_run_dir)
    parsed_args.model_save_path = os.path.join(current_run_dir, 'best_model.pt')
    parsed_args.log_file_path = os.path.join(current_run_dir, 'log.txt')
    parsed_args.dist_mat_path = os.path.join(parsed_args.data_dir, parsed_args.dataset,
                                             parsed_args.dataset + '_distance.npy')
    ged_main_dir = parsed_args.data_dir

    tab_printer(parsed_args)
    log_args(parsed_args.log_file_path, parsed_args)
    trainer = GEDTrainer(args=parsed_args)

    if parsed_args.load_model:
        parsed_args.model_save_path = os.path.join(log_root_dir, parsed_args.loaded_model_signature, "best_model.pt")
    else:
        trainer.train()

    if parsed_args.case_study:
        trainer.case_study()
    elif parsed_args.explain_study:
        trainer.explain_study()
    else:
        trainer.test()
