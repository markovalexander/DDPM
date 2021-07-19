import torch


def accumulate(model1, model2, decay=0.9999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def samples_fn(model, diffusion, shape):
    samples = diffusion.p_sample_loop(model=model,
                                      shape=shape,
                                      noise_fn=torch.randn)
    return {
        'samples': (samples + 1) / 2
    }


def progressive_samples_fn(model, diffusion, shape, device, include_x0_pred_freq=50):
    samples, progressive_samples = diffusion.p_sample_loop_progressive(
        model=model,
        shape=shape,
        noise_fn=torch.randn,
        device=device,
        include_x0_pred_freq=include_x0_pred_freq
    )
    return {'samples': (samples + 1) / 2, 'progressive_samples': (progressive_samples + 1) / 2}


class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, obj(b) if isinstance(b, dict) else b)


def bpd_fn(model, diffusion, x):
    total_bpd_b, terms_bpd_bt, prior_bpd_b, mse_bt = diffusion.calc_bpd_loop(model=model, x_0=x, clip_denoised=True)

    return {
        'total_bpd': total_bpd_b,
        'terms_bpd': terms_bpd_bt,
        'prior_bpd': prior_bpd_b,
        'mse': mse_bt
    }


def validate(val_loader, model, diffusion):
    model.eval()
    bpd = []
    mse = []
    with torch.no_grad():
        for i, (x, y) in enumerate(iter(val_loader)):
            x = x
            metrics = bpd_fn(model, diffusion, x)

            bpd.append(metrics['total_bpd'].view(-1, 1))
            mse.append(metrics['mse'].view(-1, 1))

        bpd = torch.cat(bpd, dim=0).mean()
        mse = torch.cat(mse, dim=0).mean()

    return bpd, mse


def get_variance(small, large, var_coef):
    var_coef = (var_coef + 1) / 2  # [-1, 1] to [0, 1]
    log_var = var_coef * large + (1 - var_coef) * small
    var = torch.exp(log_var)
    return var, log_var
