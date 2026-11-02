# A Data-Driven, Multi-Scale Sediment Velocity Model for Southern California

**Gaussian-process refinement of near-surface shear-wave velocity in the Los Angeles Basin using CVM-S4.26 as a regional reference model.**

This repository contains research notebooks, custom [GPyTorch](https://gpytorch.ai/) covariance kernels, model checkpoints, preprocessing utilities, and visualization workflows supporting the study **“A data-driven, multi-scale sediment velocity model for Southern California”** by Yi Liu, Grigorios Lavrentiadis, and Domniki Asimaki.

The framework integrates geological attributes and local spatial information to estimate observational corrections to the regional Community Velocity Model (CVM), rather than replacing its large-scale structure. It supports predictions of individual shear-wave velocity profiles, two-dimensional geological cross-sections, and basin-wide $V_{S30}$.

> **Research software:** This repository is a collection of experiment notebooks and supporting scripts, not a pip-installable package or a standalone UCVM distribution. The manuscript is provided as research context.

## Highlights

- **Multi-source observations:** 658 geotechnical $V_S$ profiles from the Web Portal Shear-Wave Velocity Profile Database (VPDB) and 33 deeper sonic-log profiles.
- **Reference-model correction:** Model $\ln(V_S)-\ln(V_S^{\mathrm{CVM}})$ relative to **CVM-S4.26**, so weakly constrained predictions revert toward the regional model.
- **Composite covariance:** Combine a geological/basin-aware stationary kernel with a localized **3D anisotropic Matérn-5/2** kernel.
- **Two-stage optimization:** Fit the stationary component first, freeze it, then optimize the spatial component using group-held-out predictive likelihoods.
- **Profile-level evaluation:** Hold out complete velocity profiles to test generalization at new locations, instead of randomly withholding depth samples from the same profiles.
- **Regional applications:** Compare site profiles and six cross-sections, and produce a continuous $V_{S30}$ map across the Los Angeles Basin.

## Scientific Motivation

Regional Community Velocity Models are widely used for earthquake-wave simulations and seismic-hazard studies, but their shallow sedimentary velocities can be overly smooth and weakly constrained by near-surface measurements. This is particularly important in the Los Angeles Basin, where low-velocity sediments and complicated basin geometry affect ground-motion amplification and duration.

The goal of this work is to preserve the physically meaningful, basin-scale structure of **CVM-S4.26** while correcting its systematic shallow-velocity discrepancies using independent velocity measurements. Gaussian processes offer a flexible, probabilistic description of these corrections, with covariance structures designed to share information across geologically similar sites and nearby spatial locations.

## Observations and Preprocessing

| Input | Amount | Primary role |
| --- | ---: | --- |
| Geotechnical $V_S$ profiles (VPDB) | **658 profiles** | Constrain near-surface velocity gradients; most investigate depths shallower than 100 m |
| Borehole sonic logs | **33 profiles** | Constrain deeper basin structure, extending beyond 2 km in some cases |
| **CVM-S4.26** | Regional 3D model | Provide baseline $V_P$, $V_S$, density, and reference $V_S$ estimates |
| Basin and surface attributes | Location-dependent | Provide basin identity, surface elevation, and $Z_{2.5}$ (depth to $V_S=2.5$ km/s) |


## Model Formulation

### 1. Logarithmic velocity residuals

Instead of learning $V_S$ directly, the model learns

$$
\epsilon(\mathbf{x})=\ln V_S^{\mathrm{obs}}(\mathbf{x})-\ln V_S^{\mathrm{CVM}}(\mathbf{x}),
\qquad \epsilon(\mathbf{x})\sim\mathcal{GP}\left(0,k(\mathbf{x},\mathbf{x}')\right).
$$

The corrected velocity at a prediction location is

$$
\widehat V_S(\mathbf{x})=V_S^{\mathrm{CVM}}(\mathbf{x})\exp\bigl(\mu_\epsilon(\mathbf{x})\bigr),
$$

where $\mu_\epsilon$ is the posterior mean of the log-residual field. The **zero-mean residual prior** makes the correction tend toward zero when there is little relevant observational support.

### 2. Multi-scale composite kernel

The full covariance model adds two complementary components:

$$
k(\mathbf{x},\mathbf{x}')=k_{\mathrm{stat}}(\mathbf{x},\mathbf{x}')+k_{\mathrm{spatial}}(\mathbf{x},\mathbf{x}').
$$

**Stationary geological component.** The broad, transferable correction depends on geological similarity, basin membership, and depth:

$$
\begin{aligned}
k_{\mathrm{stat}}(\mathbf{x},\mathbf{x}')={}&
\sigma_r^2\,k_{Z_{2.5}}(\mathbf{x},\mathbf{x}')
\,k_{e_s}(\mathbf{x},\mathbf{x}')\,k_d(d,d')\\
&+\sigma_b^2\,\mathbb{1}[b(\mathbf{x})=b(\mathbf{x}')]
\,k_d(d,d').
\end{aligned}
$$

Here $Z_{2.5}$ is the depth to a shear-wave velocity of 2.5 km/s, $e_s$ is surface elevation, $b$ is basin identity, and $d$ is depth below the ground surface. The continuous geological and depth terms use RBF covariance functions. The basin-identity term allows profiles in the same mapped basin to share an additional correction.

**Localized spatial component.** An anisotropic Matérn-$5/2$ kernel models residual heterogeneity that depends on three-dimensional separation:

$$
r=\sqrt{\frac{(X-X')^2+(Y-Y')^2}{\ell_h^2}
+\frac{(d-d')^2}{\ell_{d,\mathrm{sp}}^2}},
$$

$$
k_{\mathrm{spatial}}(\mathbf{x},\mathbf{x}')
=\sigma_m^2\left(1+\sqrt{5}r+\frac{5}{3}r^2\right)e^{-\sqrt{5}r}.
$$

The horizontal and vertical correlation lengths are learned separately. Together, the two kernel components account for **regional geological trends** and **localized corrections** without forcing one component to explain both scales.

### 3. Two-stage training and group-based loss

The paper uses a two-stage training strategy:

1. **Stage 1 — stationary kernel:** fit geological/basin-dependent covariance while imposing an observational noise variance floor ($\sigma_n^2\ge 0.1$) and a variance-budget constraint during this stage. Held-out groups are **spatial clusters** of profiles.
2. **Stage 2 — spatial kernel:** freeze the fitted stationary component and optimize the localized spatial kernel. Held-out groups are **entire individual velocity profiles**.

Both stages use a leave-one-group-out (LOGO) negative log predictive likelihood,

$$
\mathcal L_{\mathrm{LOGO}}(\theta)
=-\sum_{g=1}^{M}\log p(\boldsymbol\epsilon_g\mid
\mathbf X_g,\mathcal D_{-g},\theta).
$$

The paper derives an efficient evaluation based on blocks of the full covariance **precision matrix**, avoiding a separate GP fit for each omitted group. The related implementation is in [`Model_training/newloss.py`](Model_training/newloss.py); notebook-specific choices determine how this objective is applied.

## Evaluation and Results

### Generalization to held-out profiles

A **5% profile-level test set** was constructed by withholding all measurements from randomly selected profiles. The following RMSE values are reported for prediction errors in **log velocity space** in the accompanying manuscript:

| Kernel configuration | Held-out-profile RMSE |
| --- | ---: |
| **Composite (stationary + spatial)** | **0.235** |
| Stationary geological kernel | 0.243 |
| Localized spatial kernel | 0.399 |


### Residual diagnostics

The manuscript reports prediction errors approximately centered around zero over the sampled depth and $V_{S30}$ ranges, with larger near-surface dispersion reflecting unresolved shallow heterogeneity.


### Individual site predictions

The site-scale notebooks compare GP-corrected profiles against the original CVM-S4.26 prediction and other available reference profiles. The manuscript also discusses posterior prediction intervals to convey uncertainty.


### Six geological cross-sections

The paper evaluates **six sections** across different basin settings. For each section, its reported prediction grid uses **100 horizontal sample points** and **5 m vertical increments down to 500 m**. The GP results are compared with CVM-S4.26 and a Geotechnical Layer Model (GTL). The analysis emphasizes shallow low-velocity layers, smoother spatial continuity, and preservation of deeper basin trends.


### Regional $V_{S30}$ mapping

The study generates a basin-wide grid with **500 m horizontal spacing** and **2.5 m vertical spacing** for near-surface predictions, then computes the travel-time-average velocity in the upper 30 m:

$$
V_{S30}=\frac{30\ \mathrm{m}}{\sum_i h_i/V_{S,i}}.
$$

The manuscript compares these predictions with the California $V_{S30}$ model of **Wills et al. (2015)** and reports broadly consistent regional patterns with more localized structure in the GP-derived map. The relevant workflows are in `Application_vs30map/`.

## Repository Layout

```text
.
├── Preprocessing/
│   ├── Cleaning_vs_data.ipynb
│   ├── Log-test.ipynb
│   └── Assign_basinid.ipynb
├── Model_training/
│   ├── Model_trainning_step1.ipynb
│   ├── Model_trainning_step2.ipynb
│   ├── newloss.py
│   ├── pylib_gp_noconst.py
│   ├── pylib_gp_twoscale.py
│   └── *.pth, *.pkl
├── Model_validation/
│   └── Model_validation.ipynb
├── Residual/
│   └── Model_residual.ipynb
├── Application_selected_locations/
│   ├── Model_point1.ipynb
│   └── Model_point2.ipynb
├── Appliction_cross_sections/
│   ├── Model_cross_section-GP.ipynb
│   └── Model_cross_section-GTL.ipynb
├── Application_vs30map/
│   ├── Model_vs30map_prediction.ipynb
│   ├── Model_vs30map_plotting.ipynb
│   └── compare_vel_maps.py
├── Plotting/
│   └── Plotting.ipynb
└── Data/
    ├── Raw_data/
    ├── Vel_data/
    └── Figure/
```


## Environment and Installation

The notebooks primarily use Python, Jupyter, [PyTorch](https://pytorch.org/), and [GPyTorch](https://gpytorch.ai/), together with scientific computing and geospatial libraries. A starting environment is:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install numpy pandas scipy matplotlib torch gpytorch scikit-learn \
    jupyterlab pyproj geopandas shapely rasterio cartopy networkx openpyxl
jupyter lab
```

**Important:** The source archive does not include a verified lockfile or pinned package versions. This command is a practical starting point, **not a guaranteed reproducible environment**. Certain analyses depend on local datasets, geospatial files, or UCVM outputs.

## Suggested Research Workflow

1. **Preprocess data:** `Preprocessing/Cleaning_vs_data.ipynb`, `Log-test.ipynb`, and `Assign_basinid.ipynb`.
2. **Fit stage 1:** `Model_training/Model_trainning_step1.ipynb` for stationary covariance hyperparameters.
3. **Fit stage 2:** `Model_training/Model_trainning_step2.ipynb` for spatial covariance hyperparameters with the stationary component fixed.
4. **Evaluate:** `Model_validation/Model_validation.ipynb` and `Residual/Model_residual.ipynb`.
5. **Predict site profiles:** notebooks in `Application_selected_locations/`.
6. **Predict sections:** notebooks in `Appliction_cross_sections/` for GP and GTL comparisons.
7. **Generate $V_{S30}$ maps:** notebooks in `Application_vs30map/`.
8. **Plot outputs:** `Plotting/Plotting.ipynb` and figures in `Data/Figure/`.

Run the notebooks with attention to their existing relative paths and data assumptions; there is **no single verified end-to-end CLI command**. The repository contains `.pth` checkpoints and `.pkl` feature scalers. Keep model definitions, feature ordering, and fitted scalers consistent when loading them; only unpickle files from trusted sources.

## Reproducibility and Data Availability

- The archive includes experiment artifacts, figures, input files, and model checkpoints, but a clean installation may still require **UCVM** and underlying third-party geophysical datasets.
- Large rasters, checkpoints, and intermediate results may exceed GitHub web upload limits; consider **Git LFS** or separate archived releases.
- Avoid committing `__pycache__/`, `.ipynb_checkpoints/`, `.DS_Store`, virtual environments, or temporary outputs.
- **Check rights before redistribution:** VPDB profiles, sonic logs, CVM products, and California $V_{S30}$ reference grids may have distinct access, attribution, or license requirements.
- The results summarized above are from the accompanying manuscript. Users seeking exact reproduction should verify dataset versions, preprocessing, group splits, random seeds, checkpoint settings, and local path configurations.

## Citation

If you use this research, cite the associated manuscript when its final bibliographic details are available:

> Liu, Y., Lavrentiadis, G., and Asimaki, D. (2026). *A data-driven, multi-scale sediment velocity model for Southern California*. Manuscript dated October 15, 2026. **Publication venue/DOI to be updated.**

Please also cite the source observational datasets, the applicable SCEC/UCVM velocity model, and comparison models in accordance with their providers' requirements.

## Related Resources

- [SCEC Unified Community Velocity Model (UCVM)](https://github.com/SCECcode/UCVM)
- [GPyTorch documentation](https://docs.gpytorch.ai/)
- [PyTorch documentation](https://pytorch.org/docs/stable/index.html)

## License

No license was identified in the supplied archive. Add an appropriate `LICENSE` file after confirming that the code and redistributed third-party data can be released under the selected terms.
