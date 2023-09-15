import torch
from torch import nn
from util.layers import SeqBN


class AmortizedNeuralGP(nn.Module):
    def __init__(self, feature_length, n_out, emb_size=2048, num_heads=1,
                 num_layers=12, hidden_dim=2048, dropout=0., input_ln=True):
        super().__init__()

        self.feature_length = feature_length
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.n_out = n_out
        self.input_ln = input_ln

        self.bucket_means = None
        self.num_tokens = None
        self.feature_selection = None
        self.loss_object = None
        self.min_training_samples = None

        self.norm = SeqBN(emb_size) if self.input_ln else nn.Identity()
        self.input_encoder = nn.Linear(feature_length, emb_size)
        self.y_encoder = nn.Linear(1, emb_size)

        transformer_layers = nn.TransformerEncoderLayer(emb_size, num_heads, dim_feedforward=hidden_dim,
                                                        activation='gelu', dropout=dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layers, num_layers)

        self.decoder = nn.Sequential(nn.Linear(emb_size, 2 * emb_size),
                                     nn.GELU(),
                                     nn.Linear(2 * emb_size, n_out))

        for layer in self.transformer_encoder.layers:
            nn.init.zeros_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)
            nn.init.zeros_(layer.self_attn.out_proj.weight)
            nn.init.zeros_(layer.self_attn.out_proj.bias)

    def set_loss_object(self, c):
        self.loss_object = c

    def set_bucket_means(self, b):
        self.bucket_means = b

    def set_num_tokens(self, n):
        self.num_tokens = n

    def set_feature_selection(self, m):
        self.feature_selection = m

    def set_min_training_samples(self, m):
        self.min_training_samples = m

    @staticmethod
    def generate_D_q_matrix(sz, query_size):
        train_size = sz-query_size
        mask = torch.zeros(sz, sz) == 0
        mask[:, train_size:].zero_()
        mask |= torch.eye(sz) == 1
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        return mask

    def forward(self, x, y, current_eval_pos):
        # Generate the attn mask from the data shape
        src_mask = self.generate_D_q_matrix(x.shape[0], x.shape[0] - current_eval_pos).to(x.device)

        x = self.input_encoder(x)
        y = self.y_encoder(y.unsqueeze(-1))

        # add the labels to the data to condition the training samples
        train_x = x[:current_eval_pos] + y[:current_eval_pos]

        # Add on the eval samples
        src = torch.cat((train_x, x[current_eval_pos:]), 0)

        # Pass it through the model
        src = self.norm(src)
        out_embs = self.transformer_encoder(src, src_mask)
        y_hat = self.decoder(out_embs)

        return y_hat
