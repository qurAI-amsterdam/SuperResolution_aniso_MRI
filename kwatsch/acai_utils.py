import torch
from torchvision.utils import make_grid
import numpy as np
import torch.nn.utils as torchutils


def build_batches(x, n):
    x = np.asarray(x)
    m = (x.shape[0] // n) * n
    return x[:m].reshape(-1, n, *x.shape[1:])


def clip_grad_norm(optimizer, max_norm, norm_type=2):
    """Clip the norm of the gradients for all parameters under `optimizer`.

    Args:
        optimizer (torch.optim.Optimizer):
        max_norm (float): The maximum allowable norm of gradients.
        norm_type (int): The type of norm to use in computing gradient norms.
    """
    for group in optimizer.param_groups:
        torchutils.clip_grad_norm_(group['params'], max_norm, norm_type)


# helper functions for visualizing the status
def generate_recon_grid(img_ref, img_recons, max_items=16):
    if img_ref.is_cuda:
        img_ref = img_ref.detach().cpu()
    if img_recons.is_cuda:
        img_recons = img_recons.detach().cpu()
    # assuming that batch size of x is divisible by 4
    if img_recons.size(0) < max_items:
        max_items = img_recons.size(0)
    diff = img_ref[:max_items] - img_recons[:max_items]
    recons_grid = torch.cat([img_ref[:max_items], img_recons[:max_items], diff], dim=0)
    recons_grid = make_grid(recons_grid.detach().cpu(), max_items, padding=2, normalize=False, pad_value=0.5).numpy()

    return recons_grid


def interpolate_2(trainer, x, num_interpol=9, show_critic=False, side=None, clear_cache=False, eval=True):
    if side is None:
        side = x.size(0) // 2

    if isinstance(trainer.model, torch.nn.DataParallel):
        z = trainer.model.module.encode(x)
    else:
        z = trainer.encode(x, eval=eval)
    z = z.data.cpu().numpy()
    # print("interpolate_2 !!!")
    a, b = z[:side], z[-side:]
    z_interp = [a * (1 - t) + b * t for t in np.linspace(0., 1., num_interpol+2)[1:-1]]
    # print("interpolate_2 ", a.shape, b.shape)
    # z_interp = torch.stack(z_interp)
    # print("interpolate_2 ", z_interp.shape)
    z_interp = np.vstack(z_interp)
    x_interp = trainer.decode(torch.FloatTensor(z_interp).to(x.device), eval=eval, use_sr_model=True)
    # x_interp = trainer.decode(torch.FloatTensor(z_interp))
    if show_critic:
        x_all = torch.cat([x[:side], x_interp, x[-side:]], dim=0)
        alpha_pred = trainer.discriminator(x_all)
    else:
        alpha_pred = None
    x_interp = x_interp.cpu().data.numpy()

    x_fixed = x.data.cpu().numpy()
    all = []
    all.extend(x_fixed[:side])
    all.extend(x_interp)
    all.extend(x_fixed[-side:])
    all = torch.from_numpy(np.asarray(all))
    img_grid = make_grid(all, side, padding=2, normalize=False, pad_value=0.5).numpy().squeeze().transpose(1, 2, 0)
    if clear_cache:
        torch.cuda.empty_cache()

    if show_critic:
        return img_grid, alpha_pred.detach().cpu().numpy().squeeze()
    else:
        return img_grid


def create_interpol_grid(trainer, x, num_interpol=9, slice_step=1):
    if x.dim() == 3:
        # assuming x has [z, y, x] and is missing channel dimension. Insert default one
        x = x[:, None]
    if x.dtype != 'cuda':
        x = x.to('cuda')
    z = trainer.model.encode(x)
    z = z.data.cpu().numpy()
    a, b = z[slice_step:], z[:-slice_step]
    z_interp = [a * (1 - t) + b * t for t in np.linspace(0, 1, num_interpol + 2)[1:-1]]
    z_interp = np.vstack(z_interp)
    x_interp = trainer.model.decode(torch.FloatTensor(z_interp).to(x.device))
    x_interp = x_interp.cpu().data.numpy()
    x_fixed = x.data.cpu().numpy()
    all = []
    all.extend(x_fixed[slice_step:])
    all.extend(x_interp)
    all.extend(x_fixed[:-slice_step])
    all = torch.from_numpy(np.asarray(all))

    img_grid = make_grid(all, a.shape[0], padding=2, normalize=False, pad_value=0.5).numpy().squeeze().transpose(1, 2, 0)
    return img_grid


