import torch
import gpytorch

# GP Models
# ----------------------

# Stationary Model
class VelGPModelStationary(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,
                 stat_params_lengthscale_depth_init = 1.0,
                 stat_params_lengthscale_z25_init   = 1.0,
                 stat_params_lengthscale_selev_init = 1.0,
                 stat_params_scale_init             = 0.1,
                 stat_params_scale_init_id          = 0.1,
                 noise_init                         = 0.1
                 ):

        super(VelGPModelStationary, self).__init__(train_x, train_y, likelihood)

        # mean and kernel
        self.mean_module  = gpytorch.means.ZeroMean()
        self.covar_module = KernelStationary()

        # set initialization values
        self._initialize_hyparm(stat_params_lengthscale_depth_init,
                                stat_params_lengthscale_z25_init,
                                stat_params_lengthscale_selev_init,
                                stat_params_scale_init,
                                stat_params_scale_init_id,
                                noise_init)

    def _initialize_hyparm(self, 
                           stat_params_lengthscale_depth_init = 1.0,
                           stat_params_lengthscale_z25_init   = 1.0,
                           stat_params_lengthscale_selev_init = 1.0,
                           stat_params_scale_init             = 0.1,
                           stat_params_scale_init_id             = 0.1,
                           noise_init                         = 0.1
                           ):

        # initialization hyperparameters
        self.covar_module.rbf_z.lengthscale = torch.tensor([stat_params_lengthscale_depth_init])
        self.covar_module.rbf_params.lengthscale = torch.tensor([stat_params_lengthscale_z25_init, 
                                                                stat_params_lengthscale_selev_init])
        self.covar_module.rbf_params_scl.outputscale = torch.tensor(stat_params_scale_init)
        self.covar_module.kid_scl.outputscale = torch.tensor(stat_params_scale_init_id)
        # initialization noise
        self.likelihood.noise = torch.tensor(noise_init)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# Stationary Model (with constrained noise)
class VelGPModelStationaryConstrainedNoise(VelGPModelStationary):
    def __init__(self, train_x, train_y, likelihood, min_noise=1e-4,
                 stat_params_lengthscale_depth_init = 1.0,
                 stat_params_lengthscale_z25_init   = 1.0,
                 stat_params_lengthscale_selev_init = 1.0,
                 stat_params_scale_init             = 0.1,
                 stat_params_scale_init_id           = 0.1,
                 noise_init                         = 0.1
                 ):
        
        # enforce lower bound on initial noise
        noise_init = max(noise_init, min_noise)

        # call parent constructor with the actual hyperparameters
        super().__init__(
            train_x, train_y, likelihood,
            stat_params_lengthscale_depth_init = stat_params_lengthscale_depth_init,
            stat_params_lengthscale_z25_init   = stat_params_lengthscale_z25_init,
            stat_params_lengthscale_selev_init = stat_params_lengthscale_selev_init,
            stat_params_scale_init             = stat_params_scale_init,
            stat_params_scale_init_id          = stat_params_scale_init_id,
            noise_init                         = noise_init,
        )

        # add constraint on noise
        self.likelihood.noise_covar.register_constraint(
            "raw_noise",
            gpytorch.constraints.GreaterThan(min_noise)
        )

    def forward(self, x):
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class VelGPModelStationaryConstrainedNoiseScaleSum(VelGPModelStationaryConstrainedNoise):
    def __init__(self, train_x, train_y, likelihood,
                 min_noise: float = 1e-4,
                 max_total: float = 1.0,
                 stat_params_lengthscale_depth_init=1.0,
                 stat_params_lengthscale_z25_init=1.0,
                 stat_params_lengthscale_selev_init=1.0,
                 stat_params_scale_init=0.1,
                 stat_params_scale_init_id= 0.1,
                 noise_init=0.1):
        noise_init = max(noise_init, min_noise)
        super().__init__(train_x, train_y, likelihood,
                         min_noise=min_noise,
                         stat_params_lengthscale_depth_init=stat_params_lengthscale_depth_init,
                         stat_params_lengthscale_z25_init=stat_params_lengthscale_z25_init,
                         stat_params_lengthscale_selev_init=stat_params_lengthscale_selev_init,
                         stat_params_scale_init=stat_params_scale_init,
                         stat_params_scale_init_id=stat_params_scale_init_id,
                         noise_init=noise_init)
        self.min_noise = float(min_noise)
        self.register_buffer("max_total", torch.tensor(float(max_total)))

    @torch.no_grad()
    def enforce_total_variance(self):

        noise_val = float(self.likelihood.noise.item())
        scale_val = float(self.covar_module.rbf_params_scl.outputscale.item())
        const_val = float(self.covar_module.kid_scl.outputscale.item())
        max_total_val = float(self.max_total.item())

        total = noise_val + scale_val + const_val
        if total <= max_total_val:
            noise_new = noise_val
            scale_new = scale_val
            const_new = const_val
        else:
            remaining = max_total_val - noise_val
            if remaining <= 0.0:
                noise_new = max_total_val
                scale_new = 0.0
                const_new = 0.0
            else:
                extra = scale_val + const_val
                if extra > 0.0:
                    f = remaining / extra
                    noise_new = noise_val
                    scale_new = scale_val * f
                    const_new = const_val * f
                else:
                    noise_new = noise_val
                    scale_new = 0.0
                    const_new = 0.0

        if noise_new < self.min_noise:
            noise_new = self.min_noise
            remaining = max_total_val - noise_new
            if remaining < 0.0:
                scale_new = 0.0
                const_new = 0.0
            else:
                extra = scale_new + const_new
                if extra > 0.0:
                    f = remaining / extra
                    scale_new *= f
                    const_new *= f
                else:
                    scale_new = 0.0
                    const_new = 0.0

        rn   = self.likelihood.noise_covar.raw_noise
        invN = self.likelihood.noise_covar.raw_noise_constraint.inverse_transform
        rn.data.copy_(invN(torch.full_like(rn, float(noise_new))))


        ro   = self.covar_module.rbf_params_scl.raw_outputscale
        invS = self.covar_module.rbf_params_scl.raw_outputscale_constraint.inverse_transform
        ro.data.copy_(invS(torch.full_like(ro, float(scale_new))))


        rc   = self.covar_module.kid_scl.raw_outputscale
        invC = self.covar_module.kid_scl.raw_outputscale_constraint.inverse_transform
        rc.data.copy_(invC(torch.full_like(rc, float(const_new))))
    
    def forward(self, x):
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class IdDeltaKernel(gpytorch.kernels.Kernel):
    is_stationary = False
    has_lengthscale = False

    def __init__(self, id_index: int, **kwargs):
        super().__init__(**kwargs)
        self.id_index = id_index

    def forward(self, x1, x2, diag=False, **params):
        id1 = x1[..., self.id_index].long()

        if diag:
            return torch.ones(
                x1.shape[:-1],
                dtype=x1.dtype,
                device=x1.device
            )

        id2 = x2[..., self.id_index].long()
        return (id1.unsqueeze(-1) == id2.unsqueeze(-2)).to(dtype=x1.dtype)

# GP Kernels
# ----------------------

class KernelStationary(gpytorch.kernels.Kernel):
    has_lengthscale = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # kr(d)
        self.rbf_z = gpytorch.kernels.RBFKernel(
            ard_num_dims=1, active_dims=[0]
        )

        # kr(z2.5, surface elevation)
        self.rbf_params = gpytorch.kernels.RBFKernel(
            ard_num_dims=2, active_dims=[1, 2]
        )
        self.rbf_params_scl = gpytorch.kernels.ScaleKernel(self.rbf_params)

        # kid
        self.kid = IdDeltaKernel(id_index=5)
        self.kid_scl = gpytorch.kernels.ScaleKernel(self.kid)

        # (scaled(kr(z2.5,selev)) + scaled(kid)) * kr(d)
        self.stationary_kernel = (self.rbf_params_scl + self.kid_scl) * self.rbf_z

    def forward(self, x1, x2, diag=False, **params):
        return self.stationary_kernel(x1, x2, diag=diag, **params)
