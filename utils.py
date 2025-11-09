import torch
import os

def g_star_KL(t):
    return torch.exp(t - 1)

def g_star_pearson(t):
    return 0.25 * t**2 + t

def delta_g_star_KL(t):
    return torch.exp(t - 1)

def delta_g_star_pearson(t):
    return 0.5 * t + 1

def f_lambda(r, lambda_):
    r = torch.as_tensor(r)
    lam_t = torch.as_tensor(lambda_, device=r.device, dtype=r.dtype)
    term1 = torch.maximum(lam_t * r, torch.ones_like(r))
    term2 = torch.maximum(lam_t, torch.ones_like(lam_t))
    # make term2 broadcast to r safely
    term2 = term2.expand_as(term1)

    return term1 - term2

# utils_pr.py (patchs clés)
def D_train_auxiliary(x_real, G, D, D_optimizer, device, r1_gamma=None, noise_std=None, use_pearson=False):
    D.zero_grad(set_to_none=True)

    # Instance noise for real data, this makes training more stable (cf. Amortised MAP Inference for Image Super-resolution, Sønderby et al. 2016)
    if noise_std is not None:
        xr = x_real + noise_std * torch.randn_like(x_real)
    else:
        xr = x_real
    xr = xr.detach().requires_grad_(True) 

    T_real = D(xr)
    loss_real = T_real.mean()

    # Sample fake data
    z = torch.randn(x_real.size(0), 100, device=device)
    x_fake = G(z).detach()

    # Instance noise for fake data
    if noise_std is not None:
        xf = x_fake + noise_std * torch.randn_like(x_fake)
    else:
        xf = x_fake

    T_fake = D(xf)

    if use_pearson:
        T_fake_clip = torch.clamp(T_fake, -2.0, 20.0)
        g_star_fake = g_star_pearson(T_fake_clip)
    else:
        T_fake_clip = torch.clamp(T_fake, -10.0, 10.0)
        g_star_fake = g_star_KL(T_fake_clip)
    
    loss_fake = g_star_fake.mean()

    reg = 0.0
    if r1_gamma is not None: # R1 penalty
        grad_real = torch.autograd.grad(
            outputs=T_real.sum(), inputs=xr, create_graph=True, only_inputs=True
        )[0]
        r1 = (grad_real.pow(2).sum(dim=1)).mean()
        reg = 0.5 * r1_gamma * r1

    D_loss = -(loss_real - loss_fake) + reg
    D_loss.backward()
    D_optimizer.step()
    return D_loss.item()


def G_train_primal(G, D, G_optimizer, device, batch_size, noise_std=None, lambda_=1.0, use_pearson=False):
    G.zero_grad(set_to_none=True)

    # Sample fake data
    z = torch.randn(batch_size, 100, device=device)
    x_fake = G(z)

    # Instance noise for fake data
    if noise_std is not None:
        xf = x_fake + noise_std * torch.randn_like(x_fake)
    else:
        xf = x_fake

    T_fake = D(xf)
    
    if use_pearson:
        r = delta_g_star_pearson(torch.clamp(T_fake, -2.0, 20.0))
    else:
        r = delta_g_star_KL(torch.clamp(T_fake, -10.0, 10.0))

    #G_loss = f_lambda(r, lambda_).mean()
    G_loss = torch.relu(1.0 - lambda_ * r).mean() # Surrogate loss for training, to avoid flat regions of the original f_lambda

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

