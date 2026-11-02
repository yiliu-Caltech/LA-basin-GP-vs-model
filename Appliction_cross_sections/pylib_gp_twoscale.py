import math
import torch
import gpytorch

class VelGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(VelGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = Genevarikernel()  

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class Matern3D(gpytorch.kernels.Kernel):
    has_lengthscale = True

    def __init__(self, xy_idx=(3, 4), z_idx=0, alpha_init=0.3, nu=2.5, **kwargs):
        super().__init__(**kwargs)
        if nu not in (0.5, 1.5, 2.5):
            raise RuntimeError("nu must be 0.5, 1.5, or 2.5")
        self.nu = nu
        self.xy_idx = tuple(xy_idx)
        self.z_idx = int(z_idx)
        self.register_parameter(
            name="raw_alpha",
            parameter=torch.nn.Parameter(torch.log(torch.tensor(float(alpha_init))))
        )
        self.register_constraint("raw_alpha", gpytorch.constraints.Positive())

    @property 
    def alpha(self): 
        return self.raw_alpha_constraint.transform(self.raw_alpha)


    def _select_xyz(self, X):
        x = X[..., self.xy_idx[0]:self.xy_idx[0]+1]
        y = X[..., self.xy_idx[1]:self.xy_idx[1]+1]
        z = X[..., self.z_idx:self.z_idx+1]
        return torch.cat([x, y, z], dim=-1)

    def forward(self, x1, x2, diag=False, **params):
        x1_xyz = self._select_xyz(x1)
        x2_xyz = self._select_xyz(x2)

        ell = self.lengthscale 
        a = self.alpha

        x1s = x1_xyz / ell
        x2s = x2_xyz / ell
        x1s = torch.cat([x1s[..., :2], x1s[..., 2:3] / a], dim=-1)
        x2s = torch.cat([x2s[..., :2], x2s[..., 2:3] / a], dim=-1)

        dist = self.covar_dist(x1s, x2s, diag=diag, **params)

        c = math.sqrt(2.0 * self.nu)
        exp_term = torch.exp(-c * dist)
        if self.nu == 0.5:
            poly = 1.0
        elif self.nu == 1.5:
            poly = (math.sqrt(3.0) * dist).add(1.0)
        else:  
            poly = (math.sqrt(5.0) * dist).add(1.0).add(5.0 / 3.0 * dist ** 2)

        return poly * exp_term


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

    

class Genevarikernel(gpytorch.kernels.Kernel):
    has_lengthscale = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
       
        matern_xyz = Matern3D(xy_idx=(3, 4), z_idx=0, alpha_init=0.3, nu=2.5)
        self.shape_kernel = gpytorch.kernels.ScaleKernel(matern_xyz)

        self.rbf_z         = gpytorch.kernels.RBFKernel(ard_num_dims=1, active_dims=[0])
        #Z2.5 and surface elevation scaling kernel
        self.rbf_params     = gpytorch.kernels.RBFKernel(ard_num_dims=2, active_dims=[1, 2]) 
        self.rbf_params_scl = gpytorch.kernels.ScaleKernel(self.rbf_params)

        self.kid = IdDeltaKernel(id_index=5)
        self.kid_scl = gpytorch.kernels.ScaleKernel(self.kid)

        #composite kernel
        self.stationary_kernel = (self.rbf_params_scl + self.kid_scl) * self.rbf_z
        

    def forward(self, x1, x2, diag=False, **params):
        k1 = self.shape_kernel(x1, x2, diag=diag, **params)
        k2 = self.stationary_kernel(x1, x2, diag=diag, **params)
        return k1 + k2

