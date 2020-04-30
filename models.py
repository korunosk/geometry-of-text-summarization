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
    
    def predict(self, d, si):
        a0 = torch.cat((d, si), axis=0)
        z1 = self.layer1(a0)
        a1 = F.relu(z1)
        return self.layer2(a1)
    
    def score(self, d, s):
        return sum([self.predict(d, si) for si in s]).squeeze()
    
    def forward(self, d, s1, s2):
        d = d.mean(axis=0)
        return self.sigm(self.config['scaling_factor'] * (self.score(d, s1) - self.score(d, s2)))


class NeuralNetScoringPREmbModel(nn.Module):
    def __init__(self, num_emb, config):
        super(NeuralNetScoringPREmbModel, self).__init__()
        self.config = config
        self.emb = nn.Embedding(num_emb, self.config['emb_dim'])
        self.layer1 = nn.Linear(self.config['D_in'], self.config['H'])
        self.layer2 = nn.Linear(self.config['H'], 1)
        self.sigm = nn.Sigmoid()
    
    def score(self, d, s):
        return sum([self.predict(d, si) for si in s]).squeeze()
    
    def embed(self, d):
        from_table = lambda w: self.emb(torch.tensor([w])).mean(axis=0)
        return torch.cat(list(map(from_table, d)), axis=0)
        
    def predict(self, d, si):
        a0 = torch.cat((d, si), axis=0)
        z1 = self.layer1(a0)
        a1 = F.relu(z1)
        return self.layer2(a1)
    
    def forward(self, d, s1, s2):
        d  = self.embed(d).mean(axis=0)
        s1 = self.embed(s1)
        s2 = self.embed(s2)
        return self.sigm(self.config['scaling_factor'] * (self.score(d, s1) - self.score(d, s2)))
