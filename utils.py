import torch
import os



def D_train(x, G, D, D_optimizer, criterion, device):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real = x.to(device)
    y_real = torch.ones(x.shape[0], 1, device=device)

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on fake
    z = torch.randn(x.shape[0], 100, device=device)
    x_fake = G(z)
    y_fake = torch.zeros(x.shape[0], 1, device=device)

    D_output = D(x_fake)
    
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
        
    return  D_loss.data.item()

def D_train_soft_labels(x, G, D, D_optimizer, criterion, device):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real (with label smoothing)
    x_real = x.to(device)
    y_real = torch.ones(x.shape[0], 1, device=device) * 0.9  # Smooth real labels

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on fake (with label smoothing)
    z = torch.randn(x.shape[0], 100, device=device)
    x_fake = G(z)
    y_fake = torch.ones(x.shape[0], 1, device=device) * 0.1  # Smooth fake labels (instead of 0)

    D_output = D(x_fake)
    
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
        
    return  D_loss.data.item()


def D_train_soft_labels_noise_inputs(x, G, D, D_optimizer, criterion, device, noise_std=0.1):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real (with noise)
    x_real = x.to(device)
    x_real = x_real + torch.randn_like(x_real) * noise_std  # Add Gaussian noise
    y_real = torch.ones(x.shape[0], 1, device=device) * 0.9

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on fake (with noise)
    z = torch.randn(x.shape[0], 100, device=device)
    x_fake = G(z)
    x_fake = x_fake + torch.randn_like(x_fake) * noise_std  # Add Gaussian noise
    y_fake = torch.ones(x.shape[0], 1, device=device) * 0.1

    D_output = D(x_fake)
    
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
        
    return  D_loss.data.item()

def G_train(x, G, D, G_optimizer, criterion, device):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 100, device=device)
    y = torch.ones(x.shape[0], 1, device=device)
                 
    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()



def save_models(G, D, folder):
    torch.save(G.state_dict(), os.path.join(folder,'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder,'D.pth'))


def load_model(G, folder, device):
    ckpt_path = os.path.join(folder,'G.pth')
    ckpt = torch.load(ckpt_path, map_location=device)
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G


# ----- χ² auxiliary divergence -----
def g_star_chi2(t):            # g*(t) = t^2/4 + t
    return 0.25 * t * t + t

def grad_g_star_chi2(t):       # ∇g*(t) = t/2 + 1  (likelihood-ratio estimate r ≈ p/p̂)
    return 0.5 * t + 1.0

# ----- PR f-divergence -----
def f_pr(u, lam):
    # f_λ(u) = max(λu, 1) - max(λ,1)
    max_lam_1 = lam if lam > 1.0 else 1.0
    return torch.maximum(lam * u, torch.ones_like(u)) - max_lam_1

@torch.no_grad()
def _set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

def D_train_PR(x, G, T, D_optimizer):
    """
    Maximize E_real[T(x)] - E_fake[g*(T(x_fake))].
    We minimize the negative of that with the optimizer.
    """
    T.train()
    D_optimizer.zero_grad()

    B = x.size(0)
    # real
    t_real = T(x)              # (B,1)
    # fake
    z = torch.randn(B, 100, device=x.device)
    x_fake = G(z).detach()     # detach G when updating T
    t_fake = T(x_fake)

    loss_dual = -(t_real.mean() - g_star_chi2(t_fake).mean())  # minimize negative
    loss_dual.backward()
    D_optimizer.step()
    return float(loss_dual.item())

def G_train_PR(x, G, T, G_optimizer, lam: float):
    """
    Minimize E_fake[f_λ(r(x_fake))], with r = ∇g*(T(x_fake)).
    Freeze T's params, but allow gradient through T(x_fake) to flow to G.
    """
    T.eval()
    _set_requires_grad(T, False)     # freeze T's weights

    G_optimizer.zero_grad()
    B = x.size(0)
    z = torch.randn(B, 100, device=x.device)
    x_fake = G(z)                    # (B,784)

    t_fake = T(x_fake)               # gradient flows x_fake -> G, but T params frozen
    r = grad_g_star_chi2(t_fake)     # (B,1), positive ratio estimate
    loss_pr = f_pr(r, lam).mean()    # scalar

    loss_pr.backward()
    G_optimizer.step()

    _set_requires_grad(T, True)      # unfreeze for next D step
    return float(loss_pr.item())
