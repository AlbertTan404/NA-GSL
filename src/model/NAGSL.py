import torch

from model.EmbeddingLearning import GCNTransformerEncoder
from model.EmbeddingInteraction import CrossTransformer
from model.SimMatLearning import SimMatLearning


class NAGSLNet(torch.nn.Module):
    def __init__(self, args):
        super(NAGSLNet, self).__init__()
        self.args = args

        if self.args.share_qk:
            q = torch.nn.Linear(args.embedding_size, args.embedding_size * args.n_heads, bias=args.msa_bias)
            k = torch.nn.Linear(args.embedding_size, args.embedding_size * args.n_heads, bias=args.msa_bias)
        else:
            q = k = None

        self.embedding_learning = GCNTransformerEncoder(args, q, k).to(args.device)

        self.embedding_interaction = CrossTransformer(args, q, k).to(args.device)

        self.sim_mat_learning = SimMatLearning(args).to(args.device)

    def forward(self, data):

        x_0 = data['g0']['x']
        adj_0 = data['g0']['adj']
        mask_0 = data['g0']['mask']
        dist_0 = data['g0']['dist']
        x_1 = data['g1']['x']
        adj_1 = data['g1']['adj']
        mask_1 = data['g1']['mask']
        dist_1 = data['g1']['dist']

        embeddings_0 = self.embedding_learning(x_0, adj_0, mask_0, dist_0)
        embeddings_1 = self.embedding_learning(x_1, adj_1, mask_1, dist_1)

        if self.args.encoder_mask or self.args.interaction_mask or self.args.align_mask or self.args.cnn_mask:
            mask_ij = torch.einsum('ij,ik->ijk', mask_0, mask_1)
        else:
            mask_ij = None

        sim_mat = self.embedding_interaction(embeddings_0, mask_0, embeddings_1, mask_1, mask_ij)
        score = self.sim_mat_learning(sim_mat, mask_ij)

        return score
