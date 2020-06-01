import torch
import torch.nn as nn


# Loss
# if you want further info, please refer to original paper
class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = self.correct_mask(batch_size)

        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    @staticmethod
    def correct_mask(batch_size):
        mask = torch.ones(batch_size * 2, batch_size * 2).bool()
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):

        p1 = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_f(p1.unsqueeze(1), p1.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        pos = torch.cat((sim_i_j, sim_j_i), dim=0).view(self.batch_size * 2, 1)
        neg = sim[self.mask].reshape(self.batch_size * 2, -1)

        logits = torch.cat((pos, neg), dim=1)
        labels = torch.zeros(self.batch_size * 2, device=z_i.device).long()
        loss = self.criterion(logits, labels)
        loss /= 2 * self.batch_size
        return loss
