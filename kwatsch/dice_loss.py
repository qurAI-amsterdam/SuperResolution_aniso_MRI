import torch


def soft_dice_score(prob_c, one_hot):
    """

    Computing the soft-dice-loss for a SPECIFIC class according to:

    DICE_c = \frac{\sum_{i=1}^{N} (R_c(i) * A_c(i) ) }{ \sum_{i=1}^{N} (R_c(i) +   \sum_{i=1}^{N} A_c(i)  }

    Input: (1) probs: 4-dim tensor [batch_size, num_of_classes, width, height]
               contains the probabilities for each class
           (2) true_binary_labels, 4-dim tensor with the same dimensionalities as probs, but contains binary
           labels for a specific class

           Remember that classes 0-3 belongs to ES images (phase) and 4-7 to ED images

    """
    eps = 1.0e-6
    nominator = 2 * torch.sum(one_hot * prob_c, dim=(2, 3))
    denominator = torch.sum(one_hot, dim=(2, 3)) + torch.sum(prob_c, dim=(2, 3)) + eps
    return - torch.mean(nominator/denominator)


class DiceLoss(torch.nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def forward(self, input, target):
        # if self.n_classes > 2:
        one_hot = torch.zeros(input.shape).to(input.device).scatter_(1, target.unsqueeze(1), 1)
        return soft_dice_score(input, one_hot)