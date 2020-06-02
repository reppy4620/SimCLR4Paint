import torch
import torch.nn as nn


# Loss module
class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = self.get_mask(batch_size)

        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    @staticmethod
    def get_mask(batch_size):
        m = torch.ones(batch_size, batch_size).fill_diagonal_(0).bool()
        m2 = torch.cat([m, m], dim=0)
        mask = torch.cat([m2, m2], dim=1)
        return mask

    def forward(self, z_i, z_j):
        z = torch.cat((z_i, z_j), dim=0)  # 2B x C
        # calculate similarity
        # As z_i and z_j are normalized in ResNet module, just calculate dot product.
        sim = (z * z.t()) / self.temperature  # 2B x 2B

        # both values are equal because dot product isn't depended on calculation order.
        sim_i_j = torch.diag(sim, self.batch_size)  # B
        sim_j_i = torch.diag(sim, -self.batch_size)  # B

        pos = torch.cat((sim_i_j, sim_j_i), dim=0).view(self.batch_size * 2, 1)  # 2B x 1
        neg = sim[self.mask].reshape(self.batch_size * 2, -1)  # 2B x (2B - 1)

        logits = torch.cat((pos, neg), dim=1)  # 2B x 2B
        # use torch.zeros because pos samples are in logits[:, 0]
        labels = torch.zeros(self.batch_size * 2, device=z_i.device).long()  # 2B
        loss = self.criterion(logits, labels)
        loss /= 2 * self.batch_size
        return loss
