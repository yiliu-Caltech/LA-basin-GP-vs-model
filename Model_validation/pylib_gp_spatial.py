import math
import torch
import gpytorch

class VelGPModel_spa(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(VelGPModel_spa, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = Genevarikernel_spa()  

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




class Genevarikernel_spa(gpytorch.kernels.Kernel):
    has_lengthscale = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
       
        matern_xyz = Matern3D(xy_idx=(3, 4), z_idx=0, alpha_init=0.3, nu=2.5)
        self.shape_kernel = gpytorch.kernels.ScaleKernel(matern_xyz)


        

    def forward(self, x1, x2, diag=False, **params):
        k1 = self.shape_kernel(x1, x2, diag=diag, **params)
        return k1 

