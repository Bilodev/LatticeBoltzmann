# Lattice Boltzmann

Instead of solving **Navier-Stokes** directly, we don't track individual particles, but instead keep track of how many _simulated_ particles move in each direction at every point in the domain.

From these populations we can derive **density** and **velocity**.

Space is understood as a **lattice**, and velocities are taken from a small discrete set.

**D2Q9**: in two dimensions, each node has 9 directions.

$f_i(x,t)$ represents the distribution function (population); for each node we define $9$ values, i.e. taking $i \in [1,9]$, which represent the amount of fluid at node $x$ traveling at velocity $c_i$ at time $t$.

We can obtain the density at a node as the sum of its distributions, and the momentum as the velocity-weighted sum.

## Algorithm

At each step, an ideal equilibrium distribution $f_i^{eq}$ is computed, indicating how the populations would be distributed if the node were in local equilibrium with that velocity and density.

We need to compute the equilibrium and relax the current $f_i$ toward $f_i^{eq}$.

Each post-collision population ($f_i^*$) then moves to the neighboring node in its direction (streaming).

### Boundary conditions

- **Inlet (left)**: a fixed velocity $U_0$ is imposed every step, by directly setting the populations to their equilibrium value at density $1$ and velocity $U_0$.
- **Outlet (right, top, bottom)**: open, zero-gradient boundaries — populations are simply copied from the nearest interior neighbor, letting fluid leave freely without imposing a specific value.
- **Obstacle — Bounce-Back**: a population that hits the solid obstacle bounces straight back and takes on the **no-slip** condition, essentially assuming the same velocity as the wall (i.e., coming to rest).
