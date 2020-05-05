import torch
import torch.nn as nn
import torch.nn.functional as F
from geomloss import SamplesLoss


class TransformSinkhornRegModel(nn.Module):
    def __init__(self, config):
        super(TransformSinkhornRegModel, self).__init__()
        self.config = config
        self.M = nn.Parameter(torch.randn(self.config['D_in'], self.config['D_out']))
        self.sinkhorn = SamplesLoss(loss='sinkhorn', p=self.config['p'], blur=self.config['blur'], scaling=self.config['scaling'])

    def forward(self, d, s):
        dist = self.sinkhorn(torch.mm(d, self.M), torch.mm(s, self.M))
        return torch.exp(-dist)


class TransformSinkhornPRModel(nn.Module):
    def __init__(self, config):
        super(TransformSinkhornPRModel, self).__init__()
        self.config = config
        self.M = nn.Parameter(torch.randn(self.config['D_in'], self.config['D_out']))
        self.sinkhorn = SamplesLoss(loss='sinkhorn', p=self.config['p'], blur=self.config['blur'], scaling=self.config['scaling'])
        self.sigm = nn.Sigmoid()

    def forward(self, d, s1, s2):
        dist1 = self.sinkhorn(torch.mm(d, self.M), torch.mm(s1, self.M))
        dist2 = self.sinkhorn(torch.mm(d, self.M), torch.mm(s2, self.M))
        return self.sigm(self.config['scaling_factor'] * (dist2 - dist1))


class NeuralNetSinkhornPRModel(nn.Module):
    def __init__(self, config):
        super(NeuralNetSinkhornPRModel, self).__init__()
        self.config = config
        self.sinkhorn = SamplesLoss(loss='sinkhorn', p=self.config['p'], blur=self.config['blur'], scaling=self.config['scaling'])
        self.layer = nn.Linear(self.config['D_in'], self.config['D_out'])
        self.sigm = nn.Sigmoid()
    
    def predict(self, x):
        return F.relu(self.layer(x))

    def forward(self, d, s1, s2):
        dist1 = self.sinkhorn(self.predict(d), self.predict(s1))
        dist2 = self.sinkhorn(self.predict(d), self.predict(s2))
        return self.sigm(self.config['scaling_factor'] * (dist2 - dist1))


class NeuralNetScoringPRModel(nn.Module):
    def __init__(self, config):
        super(NeuralNetScoringPRModel, self).__init__()
        self.config = config
        self.layer1 = nn.Linear(self.config['D_in'], self.config['H'])
        self.layer2 = nn.Linear(self.config['H'], 1)
        self.sigm = nn.Sigmoid()
    
    def predict(self, x):
        return self.layer2(F.relu(self.layer1(x)))
    
    def score(self, d, s):
        n = s.shape[0]
        d = d.mean(axis=0).repeat(n, 1)
        x = torch.cat((d, s), axis=1)
        return torch.sum(self.predict(x))
    
    def forward(self, d, s1, s2):
        score1 = self.score(d, s1)
        score2 = self.score(d, s2)
        return self.sigm(self.config['scaling_factor'] * (score1 - score2))


class NeuralNetScoringPREmbModel(NeuralNetScoringPRModel):
    def __init__(self, num_emb, config):
        super(NeuralNetScoringPREmbModel, self).__init__(config)
        self.config = config
        self.emb = nn.Embedding(num_emb, self.config['emb_dim'])
    
    def embed(self, d):
        from_table = lambda w: self.emb(torch.tensor([w])).mean(axis=0)
        return torch.cat(list(map(from_table, d)), axis=0)
    
    def forward(self, d, s1, s2):
        d = self.embed(d)
        s1 = self.embed(s1)
        s2 = self.embed(s2)
        score1 = self.score(d, s1)
        score2 = self.score(d, s2)
        return self.sigm(self.config['scaling_factor'] * (score1 - score2))
