import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from robomimic.models.base_nets import ResNet18Conv, SpatialSoftmax
from robomimic.algo.diffusion_policy import replace_bn_with_gn, ConditionalUnet1D
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

class FiLMConditioning(nn.Module):
    def __init__(self, feature_dim, conditioning_dim):
        super().__init__()
        self.linear_a = nn.Linear(conditioning_dim, feature_dim)
        self.linear_b = nn.Linear(conditioning_dim, feature_dim)

    def forward(self, x, cond):
        a = self.linear_a(cond)
        b = self.linear_b(cond)
        return a * x + b

class DiffusionPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()

        self.camera_names = args_override["camera_names"]
        self.observation_horizon = args_override["observation_horizon"]
        self.action_horizon = args_override["action_horizon"]
        self.prediction_horizon = args_override["prediction_horizon"]
        self.num_inference_timesteps = args_override["num_inference_timesteps"]
        self.lr = args_override["lr"]
        self.weight_decay = 0

        self.num_kp = 32
        self.feature_dimension = 64
        self.ac_dim = args_override["action_dim"]
        self.obs_dim = self.feature_dimension * len(self.camera_names) + 14  # camera features and proprio

        in_shape = [512, 15, 20]

        backbones = []
        pools = []
        linears = []
        for _ in self.camera_names:
            backbone = ResNet18Conv(
                **{
                    "input_channel": 3,
                    "pretrained": False,
                    "input_coord_conv": False,
                }
            )
            backbones.append(backbone)
            pools.append(
                SpatialSoftmax(
                    **{
                        "input_shape": in_shape,
                        "num_kp": self.num_kp,
                        "temperature": 1.0,
                        "learnable_temperature": False,
                        "noise_std": 0.0,
                    }
                )
            )
            linears.append(
                torch.nn.Linear(int(np.prod([self.num_kp, 2])), self.feature_dimension)
            )
        self.backbones = nn.ModuleList(backbones)
        self.pools = nn.ModuleList(pools)
        self.linears = nn.ModuleList(linears)

        self.backbones = replace_bn_with_gn(self.backbones)

        # Initialize Multihead Attention Layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.feature_dimension, num_heads=8, batch_first=True)

        # Initialize FiLM conditioning layers for each convolution layer in the noise-prediction network
        self.film_conditioning = nn.ModuleList([
            FiLMConditioning(self.feature_dimension, self.obs_dim * self.observation_horizon)
            for _ in range(self.num_inference_timesteps)
        ])

        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=self.obs_dim * self.observation_horizon,
        )

        nets = nn.ModuleDict(
            {
                "policy": nn.ModuleDict(
                    {
                        "backbones": self.backbones,
                        "pools": self.pools,
                        "linears": self.linears,
                        "noise_pred_net": self.noise_pred_net,
                    }
                )
            }
        )

        nets = nets.float()
        if args_override["multi_gpu"] and not args_override["is_eval"]:
            assert torch.cuda.device_count() > 1
            print(f"Using {torch.cuda.device_count()} GPUs")
            nets = torch.nn.DataParallel(nets)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nets.to(device)
        self.nets = nets

        # setup noise scheduler
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=50,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="epsilon",
        )

        # count parameters
        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("number of trainable parameters: %.2fM" % (n_parameters / 1e6,))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.nets.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    def apply_film_conditioning(self, x, cond, t):
        for film_layer in self.film_conditioning:
            x = film_layer(x, cond)
        return x

    def __call__(self, qpos, image, actions=None, is_pad=None):
        B = qpos.shape[0]
        if actions is not None:  # training time
            nets = self.nets
            all_features = []
            for cam_id in range(len(self.camera_names)):
                cam_image = image[:, cam_id]
                cam_features = nets["policy"]["backbones"][cam_id](cam_image)
                pool_features = nets["policy"]["pools"][cam_id](cam_features)
                pool_features = torch.flatten(pool_features, start_dim=1)
                out_features = nets["policy"]["linears"][cam_id](pool_features)
                all_features.append(out_features)

            # Convert list to tensor for multihead attention
            all_features = torch.stack(all_features, dim=1)  # Shape: [B, num_cameras, feature_dim]

            # Apply multihead attention
            attn_output, _ = self.multihead_attn(all_features, all_features, all_features)
            attn_output = attn_output.mean(dim=1)  # Average over the camera dimension

            obs_cond = torch.cat([attn_output, qpos], dim=1)

            # Apply FiLM conditioning
            film_output = self.apply_film_conditioning(actions, obs_cond, self.num_inference_timesteps)

            # sample noise to add to actions
            noise = torch.randn(film_output.shape, device=obs_cond.device)

            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (B,),
                device=obs_cond.device,
            ).long()

            # add noise to the clean actions according to the noise magnitude at each diffusion iteration
            noisy_actions = self.noise_scheduler.add_noise(film_output, noise, timesteps)

            # predict the noise residual
            noise_pred = nets["policy"]["noise_pred_net"](
                noisy_actions, timesteps, global_cond=obs_cond
            )

            # L2 loss
            all_l2 = F.mse_loss(noise_pred, noise, reduction="none")
            loss = (all_l2 * ~is_pad.unsqueeze(-1)).mean()

            loss_dict = {}
            loss_dict["l2_loss"] = loss
            loss_dict["loss"] = loss
            return loss_dict
        else:  # inference time
            To = self.observation_horizon
            Ta = self.action_horizon
            Tp = self.prediction_horizon
            action_dim = self.ac_dim

            nets = self.nets
            all_features = []
            for cam_id in range(len(self.camera_names)):
                cam_image = image[:, cam_id]
                cam_features = nets["policy"]["backbones"][cam_id](cam_image)
                pool_features = nets["policy"]["pools"][cam_id](cam_features)
                pool_features = torch.flatten(pool_features, start_dim=1)
                out_features = nets["policy"]["linears"][cam_id](pool_features)
                all_features.append(out_features)

            # Convert list to tensor for multihead attention
            all_features = torch.stack(all_features, dim=1)  # Shape: [B, num_cameras, feature_dim]

            # Apply multihead attention
            attn_output, _ = self.multihead_attn(all_features, all_features, all_features)
            attn_output = attn_output.mean(dim=1)  # Average over the camera dimension

            obs_cond = torch.cat([attn_output, qpos], dim=1)

            # initialize action from Gaussian noise
            noisy_action = torch.randn((B, Tp, action_dim), device=obs_cond.device)
            naction = noisy_action

            # Apply FiLM conditioning
            film_output = self.apply_film_conditioning(naction, obs_cond, self.num_inference_timesteps)

            # init scheduler
            self.noise_scheduler.set_timesteps(self.num_inference_timesteps)

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = nets["policy"]["noise_pred_net"](
                    sample=film_output, timestep=k, global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                film_output = self.noise_scheduler.step(
                    model_output=noise_pred, timestep=k, sample=film_output
                ).prev_sample

            return film_output

    def serialize(self):
        return {
            "nets": self.nets.state_dict(),
        }

    def deserialize(self, model_dict):
        if "nets" in model_dict:
            status = self.nets.load_state_dict(model_dict["nets"])
        else:  # backward compatibility
            nets_dict = {}
            for k, v in model_dict.items():
                if k.startswith("nets."):
                    nets_dict[k[5:]] = v
            status = self.nets.load_state_dict(nets_dict)
        print("Loaded diffusion model")
        return status