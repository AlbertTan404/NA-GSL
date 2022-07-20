"""
Part of the code in <trainer.py> is from H2MN@cszhangzhen
"""

import torch
import numpy as np
import torch_geometric.utils
import wandb
import torch.nn.functional as F
from torch_geometric.data import Batch
from scipy.stats import spearmanr, kendalltau

from datetime import datetime

from data import GraphSimDataset
from model.NAGSL import NAGSLNet
from utils import write_log_file, metrics_spearmanr_rho, metrics_kendall_tau, \
     calculate_ranking_correlation, prec_at_ks


class GEDTrainer(object):

    def __init__(self, args):
        super(GEDTrainer, self).__init__()

        self.max_iterations = args.iterations
        self.iter_val_start = args.iter_val_start
        self.iter_val_every = args.iter_val_every
        self.batch_size = int(args.batch_size)
        self.lr = args.lr
        self.args = args

        self.dataset = GraphSimDataset(args)
        args.n_max_nodes = self.dataset.n_max_nodes

        # self.model = GraphSimTransformer(self.args).to(self.args.device)
        self.model = NAGSLNet(self.args).to(self.args.device)
        args.model_params = sum(p.numel() for p in self.model.parameters())
        print("model params:{}".format(self.args.model_params))

        if args.wandb_activate:
            wandb.watch(self.model)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                                    factor=args.lr_reduce_factor,
                                                                    patience=args.lr_schedule_patience,
                                                                    min_lr=args.min_lr,
                                                                    verbose=True)
        write_log_file(self.args.log_file_path, str(self.model))

    def train(self):
        print('\nModel training.\n')
        time = datetime.now()
        patience_cnt = 0
        min_loss = 1e10

        for iteration in range(self.args.iterations):
            self.args.temp['cur_iter'] = iteration
            self.model.train()
            batches = self.dataset.create_batches(self.dataset.training_set)
            main_index = 0
            loss_sum = 0
            for index, batch_pair in enumerate(batches):
                self.optimizer.zero_grad()
                data = self.dataset.transform(batch_pair)
                prediction = self.model(data)
                loss = F.mse_loss(prediction, data['target'], reduction='sum')
                loss.backward()
                self.optimizer.step()
                self.scheduler.step(loss)
                main_index = main_index + batch_pair[0].num_graphs
                loss_sum = loss_sum + loss.item()
            loss = loss_sum / main_index

            if iteration % max(1, self.max_iterations / 20) == 0:
                time_spent = datetime.now() - time
                time = datetime.now()
                pred = prediction.cpu().detach().numpy()
                trues = data['target'].cpu().detach().numpy()
                train_rho = metrics_spearmanr_rho(pred, trues)
                train_tau = metrics_kendall_tau(pred, trues)
                write_log_file(self.args.log_file_path,
                               "Iteration = {}\n\tbatch loss = {} (e-3)\n\trho = {} \n\ttau = {} @ {}" \
                               .format(iteration, loss * 1000 , train_rho, train_tau, time_spent))

                if self.args.wandb_activate:
                    wandb.log({
                        "loss": loss * 1000,
                        "rho": train_rho,
                        "tau": train_tau
                    })

            if iteration + 1 >= self.args.iter_val_start:
                if iteration % self.args.iter_val_every != 0:
                    continue
                # else validate
                torch.cuda.empty_cache()
                val_loss = self.validate()
                torch.cuda.empty_cache()
                if self.args.wandb_activate:
                    wandb.log({
                        "val_loss": val_loss * 1000,
                    })
                if val_loss < min_loss:
                    write_log_file(self.args.log_file_path,
                                   '\tvalidation mse decreased ( {} ---> {} (e-3) ), and save the model ... '.format(
                                       min_loss * 1000, val_loss * 1000))
                    min_loss = val_loss
                    torch.save(self.model.state_dict(), self.args.model_save_path)
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                    write_log_file(self.args.log_file_path,
                                   '\tvalidation mse: {}'.format(val_loss * 1000))

                if patience_cnt == self.args.patience:
                    break

    def validate(self):
        self.args.temp['cur_iter'] = self.args.iterations
        self.model.eval()
        batches = self.dataset.create_batches(self.dataset.val_set)
        main_index = 0
        loss_sum = 0
        with torch.no_grad():
            for index, batch_pair in enumerate(batches):
                data = self.dataset.transform(batch_pair)
                prediction = self.model(data)
                loss = F.mse_loss(prediction, data['target'], reduction='sum')
                main_index = main_index + batch_pair[0].num_graphs
                loss_sum = loss_sum + loss.item()
            loss = loss_sum / main_index

        return loss

    def all_validate(self):
        self.args.temp['cur_iter'] = self.args.iterations
        self.model.eval()
        scores = np.zeros((len(self.dataset.val_set), len(self.dataset.training_set)))
        ground_truth = np.zeros((len(self.dataset.val_set), len(self.dataset.training_set)))
        prediction_mat = np.zeros((len(self.dataset.val_set), len(self.dataset.training_set)))

        with torch.no_grad():
            for i, g in enumerate(self.dataset.val_set):
                source_batch = Batch.from_data_list([g] * len(self.dataset.training_set))
                target_batch = Batch.from_data_list(self.dataset.training_set)

                data = self.dataset.transform((source_batch, target_batch))
                target = data['target']
                ground_truth[i] = target.cpu().numpy()
                prediction = self.model(data)
                prediction_mat[i] = prediction.detach().cpu().numpy()

                scores[i] = F.mse_loss(prediction, target, reduction='none').detach().cpu().numpy()

        return np.mean(scores)

    def test(self):
        # self.args.temp['cur_iter'] = self.args.iterations
        print('\nModel evaluation.')
        if torch.cuda.device_count() == 1:
            self.model.load_state_dict(torch.load(self.args.model_save_path, map_location='cuda:0'))
        else:
            self.model.load_state_dict(torch.load(self.args.model_save_path))

        self.model.eval()
        scores = np.zeros((len(self.dataset.testing_graphs), len(self.dataset.training_graphs)))
        ground_truth = np.zeros((len(self.dataset.testing_graphs), len(self.dataset.training_graphs)))
        prediction_mat = np.zeros((len(self.dataset.testing_graphs), len(self.dataset.training_graphs)))

        rho_list = []
        tau_list = []
        pre_at_10_list = []
        pre_at_20_list = []

        with torch.no_grad():
            for i, g in enumerate(self.dataset.testing_graphs):
                source_batch = Batch.from_data_list([g] * len(self.dataset.training_graphs))
                target_batch = Batch.from_data_list(self.dataset.training_graphs)

                data = self.dataset.transform((source_batch, target_batch))
                target = data['target']
                ground_truth[i] = target.cpu().numpy()
                prediction = self.model(data)
                prediction_mat[i] = prediction.detach().cpu().numpy()

                scores[i] = F.mse_loss(prediction, target, reduction='none').detach().cpu().numpy()

                rho_list.append(calculate_ranking_correlation(spearmanr, prediction_mat[i], ground_truth[i]))
                tau_list.append(calculate_ranking_correlation(kendalltau, prediction_mat[i], ground_truth[i]))
                pre_at_10_list.append(prec_at_ks(ground_truth[i], prediction_mat[i], 10))
                pre_at_20_list.append(prec_at_ks(ground_truth[i], prediction_mat[i], 20))

        rho = np.mean(rho_list)
        tau = np.mean(tau_list)
        pre_at_10 = np.mean(pre_at_10_list)
        pre_at_20 = np.mean(pre_at_20_list)
        test_mse = np.mean(scores)

        test_results = {
            'test_mse': test_mse * 1e3,
            'test_rho': rho,
            'test_tau': tau,
            'test_p10': pre_at_10,
            'test_p20': pre_at_20
        }

        if self.args.wandb_activate:
            wandb.log(test_results)

        write_log_file(self.args.log_file_path, 'Test results:')
        for k, v in test_results.items():
            write_log_file(self.args.log_file_path, '\t {} = {}'.format(k, v))

        write_log_file(self.args.log_file_path, 'Model params:' + str(self.args.model_params))

    def explain_study(self):

        if torch.cuda.device_count() == 1:
            self.model.load_state_dict(torch.load(self.args.model_save_path, map_location='cuda:0'))
        else:
            self.model.load_state_dict(torch.load(self.args.model_save_path))

        self.model.eval()

        with torch.no_grad():

            for i, g in enumerate(self.dataset.testing_graphs):
                if i == 0:
                    continue
                source_batch = Batch.from_data_list([g] * len(self.dataset.training_graphs))
                target_batch = Batch.from_data_list(self.dataset.training_graphs)

                data = self.dataset.transform((source_batch, target_batch))
                target = data['target'].detach().cpu().numpy()
                prediction = self.model(data).detach().cpu().numpy()

                break

    def case_study(self):
        import networkx as nx
        import matplotlib.pyplot as plt

        print('\nConducting case study.')

        if torch.cuda.device_count() == 1:
            self.model.load_state_dict(torch.load(self.args.model_save_path, map_location='cuda:0'))
        else:
            self.model.load_state_dict(torch.load(self.args.model_save_path))
        self.model.eval()

        with torch.no_grad():
            def get_color(g):

                # only AIDS700nef is labeled (different kinds of color required)
                if self.args.dataset != 'AIDS700nef':
                    return '#3399ff'

                # beautiful colors
                colors = ['#33ff33', '#ff3333', '#3399ff', '#eded00', '#00cccc', '#ff66ff', ] + ['#999999'] * 23

                ret = []
                for line in g.x:
                    for k, v in enumerate(line):
                        if v == 1:
                            # for each node, append corresponding color
                            ret.append(colors[k])
                            break
                return ret

            for i, g in enumerate(self.dataset.testing_graphs):

                # draw the query graph
                nx_g = torch_geometric.utils.to_networkx(g, to_undirected=True)
                nx.draw_networkx(nx_g, node_size=500, with_labels=False, node_color=get_color(g))
                plt.show()

                source_batch = Batch.from_data_list([g] * len(self.dataset.training_graphs))
                target_batch = Batch.from_data_list(self.dataset.training_graphs)

                data = self.dataset.transform((source_batch, target_batch))
                target = data['target'].detach().cpu().numpy()
                prediction = self.model(data).detach().cpu().numpy()

                i_tar = list(enumerate(target))
                i_tar.sort(key=lambda x: x[1], reverse=True)

                i_pre = list(enumerate(prediction))
                i_pre.sort(key=lambda x: x[1], reverse=True)

                # ---------------------- ground truth ----------------------- #
                # draw top-3 graphs
                for index in range(3):
                    nx_g = torch_geometric.utils.to_networkx(self.dataset.training_graphs[i_tar[index][0]], to_undirected=True)
                    nx.draw_networkx(nx_g, node_size=500, with_labels=False, node_color=get_color(self.dataset.training_graphs[i_tar[index][0]]))
                    plt.show()

                # draw the N/2 graph
                mid = len(i_tar) // 2
                print(mid)
                nx_g = torch_geometric.utils.to_networkx(self.dataset.training_graphs[i_tar[mid][0]], to_undirected=True)
                nx.draw_networkx(nx_g, node_size=500, with_labels=False, node_color=get_color(self.dataset.training_graphs[i_tar[mid][0]]))
                plt.show()

                # draw the last graph
                nx_g = torch_geometric.utils.to_networkx(self.dataset.training_graphs[i_tar[-1][0]], to_undirected=True)
                nx.draw_networkx(nx_g, node_size=500, with_labels=False, node_color=get_color(self.dataset.training_graphs[i_tar[-1][0]]))
                plt.show()
                # ---------------------- ground truth ----------------------- #

                # ---------------------- prediction ----------------------- #
                for index in range(3):
                    nx_g = torch_geometric.utils.to_networkx(self.dataset.training_graphs[i_pre[index][0]], to_undirected=True)
                    nx.draw_networkx(nx_g, node_size=500, with_labels=False, node_color=get_color(g))
                    plt.show()

                mid = len(i_pre) // 2
                print(mid)
                nx_g = torch_geometric.utils.to_networkx(self.dataset.training_graphs[i_pre[mid][0]], to_undirected=True)
                nx.draw_networkx(nx_g, node_size=500, with_labels=False, node_color=get_color(self.dataset.training_graphs[i_pre[mid][0]]))
                plt.show()

                nx_g = torch_geometric.utils.to_networkx(self.dataset.training_graphs[i_pre[-1][0]], to_undirected=True)
                nx.draw_networkx(nx_g, node_size=500, with_labels=False, node_color=get_color(self.dataset.training_graphs[i_pre[-1][0]]))
                plt.show()
                # ---------------------- prediction ----------------------- #
