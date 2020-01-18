# Integration


## Tetrahedron method weight derivatives

The Blöchl paper gives tetrahedron weights for integrating over the 
Brillouin zone. The energy derivative of the weights is not given but would
allow calculating the density of states in a similar manner of integration.

Here, we give the expressions that result in the energy derivative of the 
integration weights $i_{nj}$. By $i_1$, $i_2$, $i_3$, and $i_4$, we denote
the contribution to the integration weights at the four corners of a 
tetrahedron, which are ordered according to increasing energy. The band index
$n$ is omitted.

For a fully occupied tetrahedra, i.e. $\epsilon < \epsilon_1$, all the 
contributions are zero:

$$
i_1 = i_2 = i_3 = i_4 = 0
$$

For $\epsilon_1 < \epsilon < \epsilon_2$:

$$
\begin{align}
i_1 ={}& C \left[ \frac{\epsilon - \epsilon_2}{\epsilon_{12}} + 
      \frac{\epsilon - \epsilon_3}{\epsilon_{13}} + 
      \frac{\epsilon - \epsilon_4}{\epsilon_{14}} \right] \\\\
i_2 ={}& C \frac{\epsilon - \epsilon_1}{\epsilon_{21}} \\\\
i_3 ={}& C \frac{\epsilon - \epsilon_1}{\epsilon_{31}} \\\\
i_4 ={}& C \frac{\epsilon - \epsilon_1}{\epsilon_{41}} \\\\
C ={}& \frac{V_\mathrm{T}}{V_\mathrm{G}} 
    \frac{(\epsilon - \epsilon_1)^2}{\epsilon_{21}\epsilon_{31}\epsilon_{41}}
\end{align}
$$

For $\epsilon_2 < \epsilon < \epsilon_3$:

$$
\begin{align}
i_1 ={}& C \frac{\epsilon - \epsilon_4}{\epsilon_{14}} + 
           \frac{\frac{\epsilon - \epsilon_3}{\epsilon_{13}} 
                 \frac{\epsilon - \epsilon_1}{\epsilon_{31}} 
                 \frac{\epsilon - \epsilon_3}{\epsilon_{23}}}{\epsilon_{41}}\\\\
i_1 ={}& C \left[ \frac{\epsilon - \epsilon_2}{\epsilon_{12}} + 
      \frac{\epsilon - \epsilon_3}{\epsilon_{13}} + 
      \frac{\epsilon - \epsilon_4}{\epsilon_{14}} \right] \\\\
i_2 ={}& C \frac{\epsilon - \epsilon_1}{\epsilon_{21}} \\\\
i_3 ={}& C \frac{\epsilon - \epsilon_1}{\epsilon_{31}} \\\\
i_4 ={}& C \frac{\epsilon - \epsilon_1}{\epsilon_{41}} \\\\
\end{align}
$$
