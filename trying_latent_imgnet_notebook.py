if __name__ == '__main__':

    t = 0
    # ### s'il nous manque la librairie?
    # pip install omegaconf>=2.0.0 pytorch-lightning>=1.0.8 torch-fidelity einops


    # import sys
    # sys.path.append(".")
    # sys.path.append('./taming_transformers')
    # from src.taming_transformers.taming.models import vqgan


    # ### aller le faire à la main...
    #@title Download
    # %cd latent-diffusion/
    #
    # !mkdir -p models/ldm/cin256-v2/
    # !wget -O models/ldm/cin256-v2/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/cin/model.ckpt
    # ### aller le faire à la main...


    #@title loading utils
    import torch
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config


    def load_model_from_config(config, ckpt):
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt)#, map_location="cpu")
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        model.cuda()
        model.eval()
        return model


    def get_model():
        config = OmegaConf.load("configs/latent-diffusion/cin256-v2.yaml")
        model = load_model_from_config(config, "models/ldm/cin256-v2/model.ckpt")
        return model


    from ldm.models.diffusion.ddim import DDIMSampler

    model = get_model()
    sampler = DDIMSampler(model)


    # And go. Quality, sampling speed and diversity are best controlled via the scale, ddim_steps and ddim_eta variables. As a rule of thumb, higher values of
    # scale produce better samples at the cost of a reduced output diversity. Furthermore, increasing ddim_steps generally also gives higher quality samples,
    # but returns are diminishing for values > 250. Fast sampling (i e. low values of ddim_steps) while retaining good quality can be achieved by using
    # ddim_eta = 0.0.


    import numpy as np
    from PIL import Image
    from einops import rearrange
    from torchvision.utils import make_grid

    classes = [25, 187, 448, 992]  # define classes to be sampled here
    n_samples_per_class = 6

    ddim_steps = 20
    ddim_eta = 0.0
    scale = 3.0  # 0 for unconditional guidance

    all_samples = list()

    with torch.no_grad():
        with model.ema_scope():
            uc = model.get_learned_conditioning({model.cond_stage_key: torch.tensor(n_samples_per_class * [1000]).to(model.device)})

            for class_label in classes:
                print(f"rendering {n_samples_per_class} examples of class '{class_label}' in {ddim_steps} steps and using s={scale:.2f}.")
                xc = torch.tensor(n_samples_per_class * [class_label])
                c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})

                samples_ddim, _ = sampler.sample(S=ddim_steps, conditioning=c, batch_size=n_samples_per_class, shape=[3, 64, 64], verbose=False,
                                                 unconditional_guidance_scale=scale, unconditional_conditioning=uc, eta=ddim_eta)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                all_samples.append(x_samples_ddim)

    # display as grid
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=n_samples_per_class)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    im = Image.fromarray(grid.astype(np.uint8))
    im.show()
