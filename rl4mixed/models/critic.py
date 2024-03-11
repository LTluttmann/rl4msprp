import torch.nn as nn
from rl4mixed.models.encoder import MHSAEncoder
from rl4mixed.settings import ModelParams


class Critic(nn.Module):

    def __init__(self, model_params: ModelParams):
        
        super().__init__()

        self.graph_embedder = MHSAEncoder(model_params)

        self.value_head = nn.Sequential(
            nn.Linear(model_params.embedding_dim, model_params.embedding_dim),
            nn.ReLU(),
            nn.Linear(model_params.embedding_dim, 1)
        )

    def forward(self, input):
        h, = self.graph_embedder(input)
        graph_embedding = h.mean(1)
        return self.value_head(graph_embedding)
