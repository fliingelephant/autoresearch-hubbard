# Exterior Attention Blocks for Fermionic NQS

## Purpose

This folder is for trial designs around exterior-algebra attention blocks for
fermionic neural quantum states. The immediate goal is not to replace the
current determinant/backflow implementation. The goal is to specify a clean
research path for testing whether determinant-like and Pfaffian-like structure
can emerge from exterior-algebra building blocks instead of being installed as
named heads.

The guiding rule is:

```text
Hardcode the fermionic algebra, not the fermionic ansatz.
```

Concretely, we want layers built from ordinary scalar attention plus fixed
exterior-algebra primitives:

```text
Attention -> Exterior block -> Attention -> Exterior block -> ...
```

The trainable parts decide what forms to construct, route, wedge, contract, and
mix. The fixed algebra supplies the antisymmetric inductive bias.

## Background

The current ByteDance/Gu et al. style Hubbard NQS is backflow-like:

```text
configuration n
  -> transformer
  -> configuration-dependent orbital matrices M_k(n)
  -> select occupied rows
  -> sum_k det(Phi_k(n))
  -> amplitude psi(n)
```

This is powerful, but determinant structure is an explicit readout. Also, since
the matrix is configuration dependent, it is not a single global exterior form
shared by all configurations. It is a map

```text
n -> M(n) -> scalar amplitude
```

with cross-configuration consistency coming from shared neural parameters.

The exterior-attention idea keeps the same VMC interface:

```text
n -> psi_theta(n)
```

but changes the internal mechanism. Instead of producing matrices and applying a
determinant or Pfaffian head, the network produces and composes exterior-valued
features. Determinants and Pfaffians should appear only as special low-complexity
regions of the learned exterior circuit:

```text
u_1 wedge ... wedge u_N                    determinant-like
P_N exp_wedge(Omega) for a learned 2-form  Pfaffian/AGP-like
```

They are reachable forms, not separate output modules.

## Two Tracks

### Track 1: Backflow-Like Exterior Amplitude Evaluator

This is the first trial target.

The amplitude remains configuration conditioned:

```text
n -> exterior features conditioned on n -> scalar amplitude psi_theta(n)
```

This is closest to the existing code and the Gu et al. ansatz. It does not
promise a single global exterior state across all configurations. Instead, it
tests whether exterior-algebra layers are useful as a more general algebraic
amplitude evaluator.

### Track 2: Strong Global Exterior Ansatz

This is a later target.

The model learns a shared exterior object or compressed generator:

```text
Psi_theta in Lambda(V), or an implicit generator of it
psi_theta(n) = <e_n, Psi_theta>
```

This has cleaner cross-configuration exterior semantics, but it is harder
because a dense N-electron sector has binomial(M, N) coefficients. This track
should wait until the backflow-like version clarifies which algebraic blocks are
stable and useful.

## Objects and Notation

Let `V` be the one-particle spin-orbital space. For a Hubbard lattice with
`N_sites` sites, a natural physical choice is

```text
dim(V) = M = 2 * N_sites
```

one mode for each site and spin.

The exterior algebra is

```text
Lambda(V) = direct_sum_g Lambda^g(V)
```

A configuration `n` with occupied spin-orbitals

```text
p_1 < p_2 < ... < p_Ne
```

corresponds to the occupation blade

```text
e_n = e_{p_1} wedge e_{p_2} wedge ... wedge e_{p_Ne}
```

A dense physical wavefunction would store coefficients of

```text
Psi = sum_{|I| = Ne} c_I e_I
```

but this is not feasible. The trial design therefore should not store dense
coefficients. It should store a compressed exterior computation.

## Key Design Principle: Higher Order by Depth

The analogy with ordinary attention is central.

Ordinary transformer layers start from simple pairwise routing, but stacked
layers compose those operations into higher-order functions. We want the same
idea with exterior algebra:

```text
simple exterior primitives + depth -> higher-order antisymmetric structure
```

Layer 1 may construct pair-like 2-form content from 1-form features. Later
attention layers redistribute that content across the lattice. Later exterior
blocks can combine or contract those features into effective higher-body
structure.

The model should not explicitly contain separate 2-body, 3-body, 4-body,
determinant, or Pfaffian modules. It should contain reusable algebraic
operations that can compose into those structures.

## Fixed Algebraic Primitives

The fixed primitives are:

```text
wedge product:        alpha wedge beta
grade projection:     Pi_g(alpha)
scalar pairing:       <alpha, beta>
interior product:     i_alpha beta
dual/Hodge map:       star alpha
```

For the first trial, the minimal block can use:

```text
wedge
interior or contraction-like product
grade projection
scalar pairing
```

The dual/Hodge map should be introduced only if needed for contraction or
complement information. If used, the metric/volume form must be fixed and simple
at first.

The primitives are not trainable.

## Variational Parameters

The layer is still variational because learnable maps surround the fixed
algebra, exactly like GATr.

For hidden exterior-valued token features `h_i`, a schematic block is:

```text
u_i = A_1 h_i
v_i = A_2 h_i
b_i = [u_i wedge v_i, interior(u_i, v_i), star(u_i)]
h_i_plus = W [h_i, b_i]
```

Trainable:

```text
A_Q, A_K, A_V     attention projections
A_1, A_2          maps selecting what to wedge/contract
W                 grade/channel mixer after algebraic products
gates             scalar gates or grade-wise gates
embeddings        site, spin, occupation, and optional positional features
readout maps      final scalar extraction parameters
```

Fixed:

```text
wedge
interior/contraction
star/dual, if used
grade projection
scalar pairing
canonical mode ordering
```

This follows the GATr pattern: geometric product and join are fixed, but
learned equivariant linear layers decide which multivectors enter those fixed
products and how their outputs are mixed.

## Hidden Representation

There are two possible hidden representations.

### Physical Exterior Fiber

Each token carries truncated forms over the physical spin-orbital space `V`.

Pros:

```text
most direct fermionic interpretation
e_p basis corresponds to physical mode p
configuration blade has literal meaning
```

Cons:

```text
Lambda^2(V) already has O(M^2) components
higher grades are impossible for large M
```

This may be useful only for small systems or toy tests.

### Latent Exterior Fiber

Each token carries forms over a small latent exterior space `E`:

```text
dim(E) = d_ext, for example 4, 6, or 8
hidden grades = 0, 1, 2 initially
```

Pros:

```text
cheap
compatible with transformer-like hidden channels
good first trial
```

Cons:

```text
less direct physical exterior semantics
readout must connect latent exterior features back to physical occupied modes
```

Recommendation for Track 1: start with the latent exterior fiber. Use physical
mode/site information only through token embeddings and readout conditioning.

## Tokenization

Two tokenization choices should be considered.

### Site Tokens

One token per lattice site. The input feature is the local occupation state:

```text
empty, up, down, double
```

This matches Gu et al. and the existing transformer setup.

Pros:

```text
small token count
close to existing model
natural positional/lattice embeddings
```

Cons:

```text
spin-orbital canonical ordering is less explicit
readout needs care to recover spin-up/spin-down sector structure
```

### Spin-Orbital Tokens

One token per spin-orbital mode:

```text
(site, spin)
```

Pros:

```text
closer to second-quantized exterior basis
occupied-mode reducer is simpler
```

Cons:

```text
twice as many tokens
site-level double occupancy information must be reconstructed or shared
```

Recommendation for first trial: site tokens if staying close to the current
code; spin-orbital tokens if the first prototype is isolated and tiny.

The current 4x4 prototype uses the site-token option to match the SiT input
basis:

```text
token_i = up_i + 2 * down_i
```

## Block Architecture

A single block should mirror a transformer block:

```text
h <- h + ScalarAttention(Norm(h))
h <- h + ExteriorBilinearFFN(Norm(h))
```

### Scalar Attention

Attention weights remain scalar:

```text
q_i = A_Q h_i
k_i = A_K h_i
v_i = A_V h_i

ell_ij = sum_g <Pi_g(q_i), Pi_g(k_j)> / sqrt(scale)
a_ij = softmax_j(ell_ij + optional_scalar_bias_ij)
m_i = sum_j a_ij v_j
```

The attention logits may combine:

```text
exterior scalar pairings
ordinary scalar-channel dot products
lattice relative-position bias, optional
```

Do not put wedge products, determinants, or Pfaffians in the softmax logits.

### Exterior Bilinear Feedforward Block

The exterior block should be local in token index after attention has mixed
information:

```text
u_i = A_1 m_i
v_i = A_2 m_i

b_i = concat(
    m_i,
    u_i wedge v_i,
    interior(u_i, v_i),
    optional star(u_i),
    optional scalar_pair(u_i, v_i)
)

out_i = W b_i
```

Then apply a residual update:

```text
h_i <- h_i + gated(out_i)
```

The activation should respect the algebraic structure. The GATr-style safe
choice is scalar-gated activation:

```text
g_i = GELU(scalar_channel_i)
h_i <- g_i * h_i
```

For the first implementation, use grade-wise scalar gates rather than
coordinate-wise nonlinearities on every exterior component.

## Readout for Track 1

The readout must produce a scalar amplitude:

```text
psi_theta(n)
```

It should not be a free scalar MLP that ignores exterior structure. It should
extract a scalar through an exterior-algebraic contraction.

A first readout candidate is an occupied-token reducer.

For occupied modes in canonical order:

```text
p_1 < p_2 < ... < p_Ne
```

initialize an accumulator:

```text
a_0 = learned scalar or vacuum form
```

then scan:

```text
a_t = R_theta(a_{t-1}, z_{p_t}(n))
```

where `z_p(n)` is the final exterior-valued feature associated with occupied
mode `p`, and `R_theta` is built from learned linear maps plus fixed exterior
operations, for example:

```text
u_t = B_1 a_{t-1}
v_t = B_2 z_{p_t}
a_t = W_R [a_{t-1}, z_{p_t}, u_t wedge v_t, interior(u_t, v_t)]
```

With site tokens, the scan still follows spin-orbital canonical order. The
prototype projects each contextual site exterior feature into spin-resolved
down/up exterior mode features before the reducer update:

```text
down occupied sites: 0, ..., N_sites - 1
up occupied sites:   0, ..., N_sites - 1
z_(site,down)(n), z_(site,up)(n) = P_spin(z_site(n))
```

Finally:

```text
A_theta(n) = <a_Ne, r_theta>
log psi_theta(n) = log_complex(A_theta(n))
```

This is backflow-like because `z_p(n)` depends on the whole configuration
through previous attention layers.

Important: the scan order must be canonical and fixed, so sign conventions are
consistent.

## Readout Alternatives

### Exterior Occupied Reducer

The reducer described above is the most aligned with the goal.

Pros:

```text
does not call determinant/Pfaffian
uses exterior primitives all the way to scalar amplitude
can compose higher-order structure by depth and by occupied scan
```

Cons:

```text
sequential over occupied modes unless parallelized with associative structure
needs careful stability tests
```

### Exterior Global Register

Add one or more global register tokens initialized as vacuum/scalar forms.
Let attention and exterior blocks update them. The amplitude is the scalar part
or selected grade projection of the final register conditioned on `n`.

Pros:

```text
simple and parallel
very transformer-like
```

Cons:

```text
may collapse into a generic scalar network if not constrained
less explicit occupied-mode composition
```

### Determinant/Pfaffian Compatibility Head

Keep determinant/Pfaffian only as an ablation or compatibility baseline, not as
the main exterior design.

Pros:

```text
easy comparison to current model
stable known readout
```

Cons:

```text
violates the main goal if treated as the primary head
```

## What Must Not Be Hardcoded

Do not add named ansatz heads:

```text
DetHead
PfaffianHead
AGPHead
explicit k-body modules
```

Do not add separate handcrafted paths for:

```text
2-body correlations
3-body correlations
4-body correlations
```

The model may learn those structures through repeated exterior blocks.

## What Is Allowed to Be Hardcoded

Allowed:

```text
canonical fermionic ordering
grade bookkeeping
wedge multiplication table
interior/contraction table
grade projection
scalar pairing
simple fixed metric/dual if needed
```

These are algebra definitions, not ansatz families.

## Expected Emergent Structures

A Slater determinant appears if the learned exterior circuit effectively builds:

```text
Psi = u_1 wedge ... wedge u_Ne
```

The coefficient of a configuration blade is then a determinant.

A Pfaffian/AGP appears if the learned circuit effectively builds:

```text
Psi = P_N exp_wedge(Omega)
```

with learned 2-form:

```text
Omega = sum_{p < q} F_pq e_p wedge e_q
```

The coefficient is then a Pfaffian.

More general structures appear when the learned exterior circuit is neither
decomposable nor a pure pair exponential.

## Computational Constraints

Full physical exterior algebra is not viable for large lattices:

```text
dim Lambda^g(V) = binomial(M, g)
```

The first trial should therefore avoid dense physical high-grade tensors.

Recommended first constraints:

```text
latent exterior dimension d_ext <= 8
stored grades: 0, 1, 2
number of exterior channels modest, e.g. 8 to 32
number of blocks initially 1 to 2
site tokens first, unless testing a tiny isolated spin-orbital prototype
```

The experiment should check whether depth and occupied-mode readout create useful
higher-order structure without storing high grades explicitly.

## Initial Milestones

### Milestone 0: Algebra Unit Tests

Before any VMC integration:

```text
wedge antisymmetry
wedge associativity on basis blades
grade projection correctness
interior product consistency
scalar pairing consistency
canonical occupied ordering signs
```

### Milestone 1: Tiny Exterior Block

Implement a standalone latent exterior block with:

```text
grades 0, 1, 2
scalar attention disabled or replaced by identity
learned A_1, A_2, W
fixed wedge/contraction
```

Verify:

```text
JAX jit works
gradients flow through learned parameters
shapes are stable
zero input does not produce non-scalar bias unless intended
```

### Milestone 2: Attention + Exterior Block

Add scalar attention around exterior features:

```text
exterior scalar-pairing logits
ordinary scalar-channel logits, optional
residual attention update
exterior bilinear FFN update
```

Verify:

```text
permutation behavior matches token design
relative-position features are handled only through scalar channels/biases
attention logits stay finite
```

### Milestone 3: Backflow-Like Scalar Readout

Implement occupied-mode exterior reducer:

```text
input config -> final token features -> occupied reducer -> scalar psi(n)
```

Start on tiny systems where exact diagonalization is available.

Verify:

```text
negative signed scalar amplitudes are carried as complex log phases
local Metropolis updates produce finite amplitude ratios
model can fit simple known amplitudes
```

### Milestone 4: Compare to Baselines

Compare against:

```text
current NNB determinant ansatz
current SiTBackflow determinant ansatz
plain transformer scalar readout
exterior block with wedge disabled
exterior block with contraction disabled
```

The important ablation is whether exterior primitives improve expressivity or
training behavior beyond a generic scalar network.

## Open Questions

1. Should the first prototype use site tokens or spin-orbital tokens?
2. Should the first exterior fiber be latent `E` or truncated physical `V`?
3. Should contraction require a fixed Hodge star, or should we implement an
   explicit interior product table directly?
4. Should the readout be an occupied scan, a global register, or both as
   ablations?
5. Should holomorphic complex parameters be introduced, or should the first
   trial keep real parameters with complex logs of signed scalar amplitudes?
6. How strict should particle-number and spin-sector conservation be inside
   hidden grades versus only at readout?

## Recommended First Trial

Start with:

```text
Track: backflow-like exterior amplitude evaluator
Tokens: site tokens, to stay close to the existing model
Exterior fiber: small latent exterior space E
Stored grades: 0, 1, 2
Blocks: Attention -> ExteriorBilinearFFN, repeated 1 or 2 times
Readout: spin-resolved exterior projection, then canonical occupied down/up scan
Amplitude: signed scalar, returned through a complex log
```

This is the smallest trial that still tests the central idea:

```text
Can determinant/Pfaffian-like structure emerge from stacked exterior building
blocks rather than from explicit determinant/Pfaffian heads?
```

## References

- Geometric Algebra Transformer, arXiv:2305.18415
- Solving the Hubbard model with Neural Quantum States, arXiv:2507.02644
- Neural Network-Augmented Pfaffian Wave-functions for Scalable Simulations of
  Interacting Fermions, arXiv:2507.10705
- Fermionic wave functions from neural-network constrained hidden states,
  arXiv:2111.10420
- Determinant-free fermionic wave function using feed-forward neural networks,
  arXiv:2108.08631
