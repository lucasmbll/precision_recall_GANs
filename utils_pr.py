import torch
import os

# utils_pr.py (patchs clés)
def D_train_auxiliary(x_real, G, D, D_optimizer, device, r1_gamma=5.0, noise_std=0.05):
    D.zero_grad(set_to_none=True)

    # Instance noise for real data
    xr = x_real + noise_std * torch.randn_like(x_real)
    xr = xr.detach().requires_grad_(True) 

    T_real = D(xr)
    loss_real = T_real.mean()

    z = torch.randn(x_real.size(0), 100, device=device)
    x_fake = G(z).detach()
    xf = x_fake + noise_std * torch.randn_like(x_fake)

    T_fake = D(xf)

    # CLIP of logits for numerical stability
    T_fake_clip = torch.clamp(T_fake, -10.0, 10.0)
    g_star_fake = torch.exp(T_fake_clip - 1.0)
    loss_fake = g_star_fake.mean()

    # R1 penalty
    grad_real = torch.autograd.grad(
        outputs=T_real.sum(), inputs=xr, create_graph=True, only_inputs=True
    )[0]
    r1 = (grad_real.pow(2).sum(dim=1)).mean()

    D_loss = -(loss_real - loss_fake) + 0.5 * r1_gamma * r1
    D_loss.backward()
    D_optimizer.step()
    return D_loss.item()


def G_train_primal(G, D, G_optimizer, device, batch_size, noise_std=0.05, lambda_=1.5):
    """
    PR-divergence generator step (hinge form with gradients in the right region):
      if lambda_ >= 1:  L = E[ relu(1 - lambda_ * r) ]     # push r up (precision)
      else:             L = E[ relu(lambda_ * r - 1) ]     # push r down (recall)
    where r = exp(T - 1), with T = D(x_fake) (logits, no sigmoid).
    """
    G.zero_grad(set_to_none=True)

    z = torch.randn(batch_size, 100, device=device)
    x_fake = G(z)

    # small instance noise (optional but helps)
    xf = x_fake + noise_std * torch.randn_like(x_fake)

    T_fake = D(xf)
    log_r = (T_fake - 1.0).clamp(-10.0, 10.0)  # numeric stability
    r = torch.exp(log_r)

    if lambda_ >= 1.0:
        G_loss = torch.relu(1.0 - lambda_ * r).mean()
    else:
        G_loss = torch.relu(lambda_ * r - 1.0).mean()

    G_loss.backward()
    G_optimizer.step()
    return G_loss.item()

def save_models(G, D, folder):
    torch.save(G.state_dict(), os.path.join(folder,'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder,'D.pth'))


def load_model(G, folder, device):
    ckpt_path = os.path.join(folder,'G.pth')
    ckpt = torch.load(ckpt_path, map_location=device)
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G