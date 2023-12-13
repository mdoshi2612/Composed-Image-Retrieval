import torch

class LossFunction(torch.nn.Module):
  def __init__(self, margin, device):
    super(LossFunction, self).__init__()
    self.margin = margin
    self.device = device
    self.cosineLoss = torch.nn.CosineEmbeddingLoss(self.margin)

  def forward(self, embedding_1, embedding_2):
    batch_size = embedding_1.size()[0]
    return torch.mean(self.cosineLoss(embedding_1, embedding_2, torch.ones(batch_size).to(self.device)) + torch.mean(embedding_1 * embedding_2))