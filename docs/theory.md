# Theory

## Boltzmann transport equation

The Boltzmann transport equation for electrons under a small electric field,
$\mathbf{F}$, can be written

$$
    \frac{e \mathbf{F}}{\hbar} \nabla_\mathbf{k} f_\mathrm{T} (\mathbf{k}, T) =
        \left ( \frac{\delta f_\mathrm{T} (\mathbf{k}, T)}{\delta t} \right ).
$$

In the constant relaxation time approximation (which we do **not** use), the
right hand side of the equation is set to a constant value, $1/\tau$,
where $\tau$ is the relaxation time. Under low field conditions, the total
distribution function, $f_\mathrm{T}$, is

$$
    f_\mathrm{T}(\mathbf{k}) = f(\mathbf{k}) + xg(\mathbf{k}),
$$

where $x$ is the cosine of the angle between $\mathbf{F}$ and
$\mathbf{k}$, $g$ is the perturbation part of the distribution.
and $f$ is the distribution at equilibrium (Fermi–Dirac distribution),
given as


$$
    f = \frac{1}{e^{(E-E_\mathrm{F})/k_\mathrm{B}T} + 1}
$$


## Elastic *versus* inelastic scattering

The perturbation is dependent on the scattering rate $s$.
The differential scattering rate from state $\mathbf{k}$ to
$\mathbf{k}^\prime$ is given as

$$
    s(\mathbf{k}, \mathbf{k}^\prime ) =
        s_\mathrm{inel} (\mathbf{k}, \mathbf{k}^\prime ) +
        s_\mathrm{el} (\mathbf{k}, \mathbf{k}^\prime )
$$

where the subscripts "$\mathrm{el}$" and "$\mathrm{inel}$"  refer
to elastic and inelastic scattering processes, respectively.

Elastic scattering is completely randomizing and has no effect on $f$, i.e.,

$$
    s(\mathbf{k}, \mathbf{k}^\prime) \equiv s(\mathbf{k}^\prime, \mathbf{k}),
$$

whereas ineleastic scattering is not randomizing and does affect $f$, namely

$$
    s(\mathbf{k}, \mathbf{k}^\prime) \neq s(\mathbf{k}^\prime, \mathbf{k}).
$$

This results from the different rates for emission and absorption processes in
inelastic scattering. Accordingly, we define the scattering *out* rate from
k-point, $\mathbf{k}$, as

$$
    s_\mathrm{out}(\mathbf{k}) = \int \left [
        s( \mathbf{k}, \mathbf{k}^\prime ) (1 - f^\prime) +
        s(\mathbf{k}^\prime, \mathbf{k}) f^\prime
        \right ] \mathrm{d}\mathbf{k}^\prime,
$$

and scattering *in* rate to k-point, $\mathbf{k}$, as

$$
    s_\mathrm{in}(\mathbf{k}) = \int X g^\prime \left [
        s(\mathbf{k}^\prime, \mathbf{k}) (1 - f) +
        s(\mathbf{k}, \mathbf{k}^\prime) f
        \right ] \mathrm{d}\mathbf{k}^\prime,
$$

where $f=f(\mathbf{k})$, $f^\prime = f(\mathbf{k}^\prime)$,
$g^\prime = g(\mathbf{k}^\prime)$ and $X$ is the angle between
$\mathbf{k}$ and $\mathbf{k}^\prime$.


## Iterative Boltzmann transport

As detailed in Rode[^Rode], the perturbation to the total distribution function
can be written,

$$
   g (\mathbf{k}) =
        \frac{s_\mathrm{in}(g, \mathbf{k})-
        \frac{e \mathbf{F}}{\hbar} \nabla_\mathbf{k} f_\mathrm{T} (\mathbf{k}, T)}
        {s_\mathrm{out}(\mathbf{k}) + s_\mathrm{el}(\mathbf{k})}
$$

As $g$  depends upon itself, an iterative procedure is needed to solve the
perturbation to the distribution function. Fortunately, the functional form of
$g$ ensures it converges exponentially, in most cases requiring fewer than
5 iterations to achieve accurate results.

In the absence of inelastic scattering, $s_\mathrm{in}(g) = 0$ and
$g_1$, is the exact solution. In addition, the $g_1$ solution is
often termed the relaxation time approximation (RTA).

## Calculating transport properties

Transport properties are calculated using the transport density of states (DOS),
defined as:

$$
    \sigma(E, T) =
        \int \sum_b \mathbf{v}_{b,\mathbf{k}} \otimes \mathbf{v}_{b,\mathbf{k}}
        \tau_{b,\mathbf{k},T} \delta (E - E_{b, \mathbf{k}})
        \frac{\mathrm{d}\mathbf{k}}{8 \pi^3},
$$

where $\mathbf{v}_{b, \mathbf{k}}$ is the group velocity of k-point,
$\mathbf{k}$ in band, $b$, $\tau$ is the lifetime defined as
$1/s(\mathbf{k})$ and $E$ is energy.

The transport DOS is used to calculate the moments of the generalized transport
coefficients


$$
    \mathcal{L}^{(\alpha)}(E_\mathrm{F}, T) =
        q^2 \int \sigma(E, T)(E - E_\mathrm{F})^\alpha \left (
        - \frac{ \delta f_\mathrm{T}(E, E_\mathrm{F}, T)}{\delta E}
        \right ) \mathrm{d}E,
$$

where $E_\mathrm{F}$ is the Fermi level at doping concentration, $c$.

Finally, the electrical conductivity, $\sigma$, Seebeck coefficient,
$S$, and electronic contribution to the thermal conductivity,
$\kappa_\mathrm{e}$, are calculated according to

$$
\begin{align}
    \sigma = {}& \mathcal{L}^{(0)},\\
    S = {}& \frac{1}{qT} \frac{\mathcal{L}^{(1)}}{\mathcal{L}^{(0)}},\\
    \kappa_\mathrm{e} = {}& \frac{1}{q^2T}
        \left [ \frac{(\mathcal{L}^{(1)})^2}{\mathcal{L}^{(0)}}
            - \mathcal{L}^{(2)} \right ] .
\end{align}
$$



[^Rode]: Rode, D. L. *Low-field electron transport. Semiconductors and semimetals* **10**, (Elsevier, 1975).
