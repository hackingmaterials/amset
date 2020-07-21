# Transport Properties


Electronic transport properties — namely, conductivity, Seebeck coefficient, 
and electronic component of thermal conductivity — are calculated through the 
[Onsager coefficients](https://www.sciencedirect.com/science/article/pii/S0010465518301632?via%3Dihub).
The spectral conductivity is calculated as
```math
\Sigma_{\alpha\beta}(\varepsilon) =  \sum_n \int \frac{\mathrm{d}{\mathbf{k}}}{8\pi^3} 
v_{n\mathbf{k},\alpha}v_{n\mathbf{k},\beta}\tau_{n\mathbf{k}}
\delta{\left(\varepsilon - \varepsilon_{n\mathbf{k}} \right )},
```
where $`\alpha`$ and $`\beta`$ denote Cartesian coordinates, 
$`\varepsilon_{n\mathbf{k}}`$ and $`v_{n\mathbf{k},\alpha}`$ are the energy and 
group velocity of band index $`n`$ and wave vector $`\mathbf{k}`$, respectively. 
The spectral conductivity can be used to compute the moments of the generalized 
transport coefficients
```math
\mathcal{L}^n_{\alpha\beta} = e^2 \int \Sigma_{\alpha\beta}(\varepsilon) (\varepsilon - \varepsilon_\mathrm{F})^n
\left [ -\frac{\partial f^0}{\partial \varepsilon} \right ] \mathrm{d}{\varepsilon},
```
where $`e`$ is the electron charge and $`\varepsilon_\mathrm{F}`$ is the Fermi level at a certain doping 
concentration and temperature $`T`$.
The Fermi–Dirac distribution is given by
```math
    f^0_{n\mathbf{k}} = \frac{1}{\exp\left[{(\varepsilon_{n\mathbf{k}}-\varepsilon_\mathrm{F})/k_\mathrm{B}T} \right] + 1},
```
where $`k_\mathrm{B}`$ is the Boltzmann constant.
Electrical conductivity ($`\sigma`$), Seebeck coefficient ($`S`$), and the 
charge carrier contribution to thermal conductivity ($`\kappa`$) are obtained as
```math
\begin{aligned}
\sigma_{\alpha\beta} ={}& \mathcal{L}_{\alpha\beta}^0, \\
S_{\alpha\beta} ={}& \frac{1}{eT} \frac{\mathcal{L}_{\alpha\beta}^1}{\mathcal{L}_{\alpha\beta}^0}, \\
\kappa_{\alpha\beta} = {}& \frac{1}{e^2T}
\left [ \frac{(\mathcal{L}_{\alpha\beta}^1)^2}{\mathcal{L}_{\alpha\beta}^0}
- \mathcal{L}_{\alpha\beta}^2 \right ] .
\end{aligned}
```
