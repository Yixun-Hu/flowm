# Method notes: continuous-motion estimation and co-moving visual memory

This document describes the implemented model in `../model.py` and the controls
in `mechanism.py`. It is intended as material for a methods section, not a claim
that the combination is novel or empirically superior. In particular, the
implementation estimates motion **after** displacement attention; it does not
feed an estimated velocity back into the initial attention scores.

The associated evidence campaign is complete: 44 training runs, 280,000 optimizer
updates, 106 evaluation jobs, 41,408 model sequences, and 101,152 metric rows were
artifact-validated; the full 43-test suite passed. See the
[completion record](../../../outputs/continuous_velocity/paper_v1/completion.json)
and [research findings](../../../outputs/continuous_velocity/paper_v1/research_findings.md)
for final evidence and interpretation. Completion does not mean every proposed
architectural advantage was confirmed.

## Problem and notation

We observe grayscale context images \(I_s:\Omega\to[0,1]\), timestamps
\(t_s\), and known camera-window origins \(\mathbf c_s\in\mathbb Z^2\), where
\(\Omega=\{0,\ldots,31\}^2\). The world is the periodic domain
\(\mathbb T_L^2\) with \(L=50\). A view pixel \(\mathbf x\) corresponds to
world position \(\mathbf r=\mathbf x+\mathbf c_s\pmod L\). Let
\(t_0=\max_s t_s\) denote the last observed timestamp. Future camera origins
and query timestamps are provided as known controls.

The environment contains one rigid appearance moving at an unknown constant
velocity \(\mathbf v\in\mathbb R^2\):

\[
\mathbf p(t)=\mathbf p(t_0)+(t-t_0)\mathbf v.
\]

The unwrapped position obeys this law even when the rendered position crosses a
periodic boundary. The model receives neither \(\mathbf p\), \(\mathbf v\)
nor an object template. It predicts images at requested future timestamps.

The model's sequence-specific state consists of one inferred velocity and a
periodic canonical memory. The canonical coordinate system moves with the
object and coincides with world coordinates at \(t_0\). **It is not an
object-centered crop:** the implementation does not explicitly infer an object
center \(\mathbf p(t_0)\) or use an object query/slot. The object's reference
location is represented by where its appearance lies in the 50-by-50 field.

## Camera-compensated features

A shared image encoder \(\phi_\theta\) produces four learned channels per
pixel. The implemented encoder is a bias-free 3-by-3 convolution from 1 to 12
channels, a SiLU, and a bias-free 3-by-3 convolution from 12 to 4 channels. Raw
intensity is concatenated as an appearance skip:

\[
\mathbf f_s(\mathbf x)
=\left[I_s(\mathbf x),\phi_\theta(I_s)(\mathbf x)\right]\in\mathbb R^5.
\]

Using the camera origin, these features are scattered into a world field
\(\mathbf F_s\) with zeros outside the observed window. The binary mask
\(m_s\) marks the camera-covered world pixels:

\[
\mathbf F_s(\mathbf r)=
\begin{cases}
\mathbf f_s(\mathbf r-\mathbf c_s),&\mathbf r\text{ is covered by the view},\\
0,&\text{otherwise}.
\end{cases}
\]

All world indexing is periodic. Camera origins are integer-valued in this
experiment; this embedding does not implement arbitrary subpixel camera
translations. The camera mask indicates **observation coverage**, not a learned
object mask and not an estimate of foreground occlusion.

## Masked displacement attention

For context frame pairs \(k=(i,j)\), define \(\Delta t_k=t_j-t_i>0\). The
implementation selects 12 approximately evenly spaced pairs for each index lag
1, 2 and 4, yielding 36 pairs (some pairs repeat when context is very short).
With irregular observations, the index lag is not
the physical time interval; the actual timestamps determine \(\Delta t_k\).

For each pair, evaluate periodic integer displacements
\(\boldsymbol\delta\in\{-25,\ldots,24\}^2\). The normalized matching score
is computed from

\[
N_k(\boldsymbol\delta)
=\sum_{\mathbf r}
\mathbf F_i(\mathbf r)^\top\mathbf F_j(\mathbf r+\boldsymbol\delta),
\]

\[
D_k(\boldsymbol\delta)
=\sum_{\mathbf r}\left[
\|\mathbf F_i(\mathbf r)\|^2m_j(\mathbf r+\boldsymbol\delta)
+m_i(\mathbf r)\|\mathbf F_j(\mathbf r+\boldsymbol\delta)\|^2
\right],
\]

\[
S_k(\boldsymbol\delta)
=\operatorname{clip}_{[-1,1]}
\frac{2N_k(\boldsymbol\delta)}{\max(D_k(\boldsymbol\delta),10^{-5})}.
\]

Thus appearance similarity is evaluated only over jointly observed support
after a candidate translation. These are global masked feature correlations
for each frame pair, computed using FFTs. They are **not** independent
query-to-key attention for each spatial token.

The raw-intensity support statistic is

\[
R_k(\boldsymbol\delta)=\sum_{\mathbf r}\left[
F_{i,0}(\mathbf r)^2m_j(\mathbf r+\boldsymbol\delta)
+m_i(\mathbf r)F_{j,0}(\mathbf r+\boldsymbol\delta)^2
\right].
\]

Numerically, \(R_k\) is clamped below at zero. A candidate is admissible if

\[
|\delta_x|,|\delta_y|\leq 2.5\Delta t_k+1,
\qquad R_k(\boldsymbol\delta)>2.
\]

The displacement attention is

\[
a_k(\boldsymbol\delta)
=\operatorname{softmax}_{\boldsymbol\delta}
\left(\beta S_k(\boldsymbol\delta)+\text{admissibility mask}\right),
\qquad
\beta=\operatorname{clip}_{[5,150]}e^{b_\theta},
\]

followed by the continuous displacement expectation

\[
\hat{\mathbf d}_k
=\sum_{\boldsymbol\delta}a_k(\boldsymbol\delta)\boldsymbol\delta.
\]

The mask is implemented by setting inadmissible logits to \(-10^4\). If every
candidate is invalid, the subsequent pair weight is forced to zero; the
resulting softmax expectation is not treated as motion evidence.

There is no fixed bank of velocity channels, and \(\hat{\mathbf d}_k\) is not
rounded to a pixel or velocity channel. Nevertheless, the correspondence
distribution is defined on a **finite integer displacement lattice** and has a
bounded search support. The method does not eliminate all spatial
discretization, aliasing, or multimodal matching. A soft expectation of two
incompatible displacement modes need not be a physically correct match.

The architecture also assumes that the relevant displacement is identifiable
within the chosen periodic representative. A temporal gap large enough to
allow half-world or larger displacement can create winding ambiguity; the
current estimator does not explicitly enumerate or solve those winding counts.

## Confidence and differentiable constant-velocity fitting

For each pair, compute expected matching strength, expected support and scaled
entropy:

\[
q_k=\sum_{\boldsymbol\delta}a_kS_k,
\quad
r_k=\sum_{\boldsymbol\delta}a_kR_k,
\quad
h_k=-\frac18\sum_{\boldsymbol\delta}
a_k\log\max(a_k,10^{-10}).
\]

A small MLP \(g_\theta:\mathbb R^3\to\mathbb R\), with a 12-unit SiLU
hidden layer, predicts reliability:

\[
\gamma_k=\operatorname{sigmoid}
g_\theta\!\left(q_k,h_k,\frac{\log(1+r_k)}6\right),
\]

\[
w_k=\gamma_k\,[\max(q_k,0)]^4
\frac{r_k}{r_k+10}
\,\mathbf1\{\text{some candidate is admissible}\}.
\]

For fixed positive weights, the constant-velocity fit minimizes

\[
\min_{\mathbf v}\sum_k w_k
\|\hat{\mathbf d}_k-\Delta t_k\mathbf v\|_2^2,
\]

with closed-form solution

\[
\hat{\mathbf v}^{(0)}
=\frac{\sum_k w_k\Delta t_k\hat{\mathbf d}_k}
{\max(\sum_k w_k\Delta t_k^2,10^{-6})}.
\]

Two unrolled robust refinement steps downweight disagreements in velocity units:

\[
w_k^{(r+1)}
=\frac{w_k}
{1+\|\hat{\mathbf d}_k/\Delta t_k-\hat{\mathbf v}^{(r)}\|_2^2/0.15^2},
\qquad r=0,1,
\]

\[
\hat{\mathbf v}^{(r+1)}
=\frac{\sum_k w_k^{(r+1)}\Delta t_k\hat{\mathbf d}_k}
{\max(\sum_k w_k^{(r+1)}\Delta t_k^2,10^{-6})}.
\]

The final estimate is \(\hat{\mathbf v}=\hat{\mathbf v}^{(2)}\). This is an
unrolled robust estimator with fixed refinement count; it is not a claim that
a global robust objective has been exactly optimized. Gradients propagate
through the expectations, confidence weights, fits and refinements. Hard
admissibility masks and clamps make the full implementation piecewise
differentiable rather than smooth everywhere.

The physical bias enters through the common relationship
\(\hat{\mathbf d}_k\approx\Delta t_k\mathbf v\), shared by all selected
pairs. The learned components determine which appearances match and how
strongly to trust their evidence. Importantly, the robust iterations update
**pair weights and the velocity only**: they do not recompute displacement
attention, narrow its candidate search around the fitted velocity, or add a
velocity-dependent term to its logits.

If all evidence weights vanish, the safeguarded fit returns zero velocity.
This is a numerical fallback for unobservability, not a calibrated posterior
claim of stationarity.

## Trajectory-constrained alignment and memory attention

Let \(\mathcal B[X](\mathbf q)\) denote periodic bilinear sampling of a grid
\(X\) at continuous coordinate \(\mathbf q\). At a canonical coordinate
\(\mathbf u\), observation \(s\) is read along the inferred trajectory:

\[
\mathbf q_s(\mathbf u)
=\mathbf u+(t_s-t_0)\hat{\mathbf v},
\]

\[
\mathbf A_s(\mathbf u)
=\mathcal B[\mathbf F_s](\mathbf q_s(\mathbf u)),
\qquad
\mu_s(\mathbf u)=\mathcal B[m_s](\mathbf q_s(\mathbf u)).
\]

The aligned value divides out attenuation caused by interpolating zero-padded
view boundaries:

\[
\mathbf V_s(\mathbf u)
=\frac{\mathbf A_s(\mathbf u)}{\max(\mu_s(\mathbf u),10^{-4})}.
\]

A pointwise learned network \(h_\theta:\mathbb R^5\to\mathbb R\), implemented
as 1-by-1 convolutions with an 8-unit SiLU hidden layer, supplies temporal fusion
scores:

\[
\rho_s(\mathbf u)
=\mu_s(\mathbf u)
\exp\!\left(\operatorname{clip}_{[-8,8]}
h_\theta(\mathbf V_s(\mathbf u))\right),
\]

\[
\mathbf M(\mathbf u)
=\frac{\sum_s\rho_s(\mathbf u)\mathbf V_s(\mathbf u)}
{\max(\sum_s\rho_s(\mathbf u),10^{-5})}.
\]

This is content-dependent, visibility-weighted temporal attention at an aligned
coordinate. It does not use a separate query vector, future time, velocity, or
independent per-frame offset in the fusion score. The shared velocity determines
where every context frame is sampled. Setting all learned scores to zero gives
the visibility-weighted mean control exactly.

The canonical image is produced once:

\[
C(\mathbf u)
=\operatorname{clip}_{[0,1]}
\left(M_0(\mathbf u)+D_\theta(\mathbf M)(\mathbf u)\right),
\]

where \(D_\theta\) is a 3-by-3 circular convolution from 5 to 16 channels, SiLU,
and a 3-by-3 circular convolution from 16 to 1 channel. \(M_0\) is the fused raw
intensity channel. The residual decoder is time-independent and is applied to
the canonical grid, not separately conditioned on each forecast horizon.

Neither the canonical feature memory nor its decoded image is recursively
updated during prediction. This enforces appearance-state persistence, but does
not guarantee that the inferred appearance is complete, sharp or free of ghost
copies. Incorrect alignment or fusion can already corrupt the canonical image,
and that corruption is then transported consistently.

## Analytic continuous transport and rendering

For a requested future time \(t_0+\tau\) and view pixel \(\mathbf x\), predict

\[
\boxed{
\hat I_{t_0+\tau}(\mathbf x)
=\mathcal B[C]\!\left(
\mathbf x+\mathbf c_{t_0+\tau}-\tau\hat{\mathbf v}
\right).
}
\]

Coordinates are taken modulo \(L\) for sampling. Each future image is sampled
directly from the same canonical image using its **absolute elapsed time**.
No previous predicted image is warped, and no velocity-channel argmax or
integer roll is used. The underlying representation remains a finite grid
with a piecewise bilinear extension; it is not an implicit coordinate MLP.

This avoids a chain of repeated forecast-to-forecast resampling. It does not
eliminate interpolation error: source images may already be bilinearly
rasterized, each observation is resampled during alignment, and the canonical
image is sampled again for rendering. Learned sharpening can compensate for
some effects, but exact reconstruction and arbitrary-resolution accuracy are
not architectural guarantees.

## Mathematical properties and their limits

### Composition of latent motion

For a frozen velocity and canonical appearance, the latent transition can be
written

\[
T_\tau(C,\mathbf p,\mathbf v)
=(C,\mathbf p+\tau\mathbf v,\mathbf v).
\]

It satisfies

\[
T_b\circ T_a=T_{a+b},
\qquad
\mathbf d(a+b)=\mathbf d(a)+\mathbf d(b),
\quad \mathbf d(\tau)=\tau\mathbf v.
\]

The implemented model stores reference position implicitly in \(C\); the
equivalent property is additive query-coordinate displacement. This is an
algebraic property of the transition, up to floating-point arithmetic. It is
not evidence that inferred velocities are correct or that image prediction
improves empirically.

In particular, repeatedly rasterizing and bilinearly warping an image generally
does **not** obey exact composition:

\[
\mathcal W_b(\mathcal W_a(C))\ne\mathcal W_{a+b}(C)
\]

for sampled bilinear warp operators \(\mathcal W\). The implementation avoids
that repeated-rasterization path and queries \(C\) directly. Comparing direct
and composed latent coordinates is a correctness check; comparing long-horizon
forecasts against learned-offset controls establishes whether the property is
useful for this task.

### Growth of position error

Suppose an identifiable reference point has initial position error
\(\mathbf e_0=\hat{\mathbf p}_0-\mathbf p_0\) and velocity error
\(\mathbf e_v=\hat{\mathbf v}-\mathbf v\). Before periodic reduction,

\[
\mathbf e(\tau)=\mathbf e_0+\tau\mathbf e_v,
\qquad
\|\mathbf e(\tau)\|_2
\leq\|\mathbf e_0\|_2+|\tau|\|\mathbf e_v\|_2.
\]

With identical initial appearance/location and only a velocity quantization
intervention \(Q_\Delta\), the unwrapped reference-point discrepancy is

\[
\|\mathbf e_Q(\tau)\|_2
=|\tau|\|Q_\Delta(\mathbf v)-\mathbf v\|_2.
\]

This equality applies to ideal single-trajectory transport with shared initial
state. It is not a general error law for FloWM's multiple channels, for
corrupted memory, or for detector-based metrics. Quantizing the velocity used
for **context alignment** can additionally alter the initial canonical state,
which is why the forecast-only intervention keeps memory fixed.

If the canonical continuous appearance \(A\) were \(K\)-Lipschitz, a pure
translation error would also bound a pointwise appearance discrepancy by
\(K\|\mathbf e(\tau)\|\), plus any appearance reconstruction error. Sharp
nearest-rendered edges and finite-grid sampling complicate applying this bound
to observed pixel errors. Missing and ghost detections are thresholded events
and need not grow linearly with position error.

Periodic image similarity can repeat after a full-world displacement while
unwrapped trajectory error continues to grow. Evaluation must therefore retain
winding information when identifiable and separately report failure/ambiguity.

### Coordinate and time structure

Known camera origins enter through world embedding and the final query, so
camera and object motion have distinct roles. Integer translations of the
global coordinate origin are compatible with periodic embedding, correlations
and circular convolution; numerical consistency is testable. This does not
establish exact equivariance to arbitrary subpixel camera transforms.

Timestamps explicitly govern the physical fit, context alignment and future
transport. However, the bounded candidate search, fixed robust scale 0.15 and
learned image features prevent claiming unrestricted temporal-rescaling or
Galilean equivariance for the entire estimator. Useful irregular-time or
long-horizon generalization remains an empirical question.

## Learning and physical initialization

Formal training uses only future image reconstruction. For target pixels
\(I\), the implemented loss is

\[
\mathcal L
=\frac{\sum_{t,\mathbf x}
\left(1+5\mathbf1\{I_t(\mathbf x)>0.05\}\right)
\left(\hat I_t(\mathbf x)-I_t(\mathbf x)\right)^2}
{\sum_{t,\mathbf x}\left(1+5\mathbf1\{I_t(\mathbf x)>0.05\}\right)}.
\]

This foreground-sensitive weighting uses target pixel values, not object
position, velocity, template identity or bounding-box supervision. In the
current campaign, 50 regular context frames are paired with six training
forecast horizons: 1, 5, 10, 20, 30 and 40 frames after the last observation.

At initialization the confidence MLP's last layer is zero, so
\(\gamma_k=0.5\). Memory-score outputs are zero, giving visibility-weighted
averaging; the residual decoder output is zero, giving the fused raw image.
The matching inverse temperature starts at \(e^{3.7}\), while learned
descriptor convolutions are randomly initialized. Consequently an untrained
physical model already performs raw-supported registration, analytic fitting,
motion compensation, averaging and direct transport. Good zero-update results
are an expected consequence of substantial algorithmic structure and must not
be attributed to learned attention.

The optional true-velocity override is reserved for labeled diagnostic runs.
It replaces the inferred velocity before the specified alignment/transport
stage and is excluded from formal training and ordinary test predictions.
It is **not a guaranteed image-error bound**: a trained memory or residual
decoder can compensate for systematic velocity-estimation bias, so replacing
that estimate by the true velocity can change the distribution on which those
modules operate. Oracle interventions diagnose mechanisms; exact-GT-frame
predictions separately measure detector/evaluator ceilings.
Evaluation may use an oracle digit template for localization; that template is
not available to this predictor or the template-free raw baseline.

## Evaluation and replication protocol

Learning rates 0.0003 and 0.001 are compared on 64 validation sequences using
tuning seed 2026, with best checkpoints selected within a 5,000-update budget.
The core campaign trained fresh seeds 2027/2028/2029 for every model type;
physical and relaxed additionally use 2030. Thus those two models have five
total fits, four post-HPO, and the other four core controls have four total fits,
three post-HPO. The 2029 runs for offset, MLP velocity, mean memory and raw
correlation were specified in an explicit additional protocol. These selected
fits and their evaluations are now complete. Alternative learning-rate tuning
runs are not additional post-HPO replicates. The tuning seed
is shown separately from primary cross-training-seed comparisons.

The completed 20,000-update extensions use seed 2026 with learning rates selected
at 5,000 updates. They are longer-budget optimization diagnostics, not proofs of
convergence; this limitation applies to the completed MLP extension too. Rendering/2-D
variants use the original three seeds 2026/2027/2028 and reuse the selected
horizontal-bilinear learning rate; their post-HPO cohort has two seeds.

Metric schema 2 separates what can be observed in images from internal motion
outputs and from winding assumptions:

1. **Image MSE** on all target pixels and **frame-localization success** are the
   primary image outcomes. Frame success requires a sufficiently visible target,
   a detected non-duplicated object, and periodic localization error at most
   2 pixels. This score does not fit a velocity, but cannot detect whole-world
   winding errors.
2. **Observed continuity** unwraps detections from a single context anchor using
   bounded successive motion, without a constant-velocity fit. Its ADE is
   accompanied by coverage and permanent winding-ambiguity flags after
   unresolvable gaps. GT sets the initial winding only, never later reanchors.
3. **Internal transport error** compares the output displacement with
   \((t-t_0)\mathbf v\). It assesses the model's motion state, not object
   localization in generated images; it excludes appearance and reference
   position error.
4. **CV-certified trajectory metrics** retain the older global-motion fitting
   assumption as explicitly named secondary diagnostics. They can reject correct
   frames or nonlinear predictions and are not primary model-neutral outcomes.

Missing and detected-ghost rates remain visible. Perfect-frame controls expose
detector and winding-evaluator limitations. Version-1 evaluations were archived;
every included model was reevaluated with schema 2 for the primary comparisons.
Each model's score pools frame/pixel numerators and denominators; summaries then
report mean and sample SD across trained models. Paired test-seed bootstraps
describe test uncertainty conditional on those fits, not uncertainty over future
training runs.

## What the ablations identify

| Comparison | Directly tested mechanism | Interpretation limit |
| --- | --- | --- |
| Trained physical vs raw-untrained | Benefit of the learned system over a wholly nonlearned registration/transport pipeline | Changes learned features, fusion and decoder together |
| Physical vs trained raw-correlation | Learned matching versus fixed raw matching, with the same memory/decoder architecture | End-to-end training can adapt other modules |
| Loaded physical vs raw-matching swap | Change only matching; retain fitted memory features, fusion and decoder | Swap changes descriptors, temperature and confidence together and can introduce distribution shift |
| Physical vs MLP velocity | Analytic fit versus direct learned velocity regression on the same correspondence observations | Different parameter count and optimization difficulty; budgets/tuning matter |
| Physical vs mean-memory | Learned temporal fusion versus visibility-weighted averaging | Retrained and fixed-checkpoint swaps answer different questions |
| Continuous vs quantized velocity, transport only | Future transport precision with identical canonical memory | Single trajectory; does not reproduce FloWM's readout competition |
| Continuous vs quantized velocity, both stages | Combined impact on context alignment and future transport | Cannot attribute all improvement to future drift alone |
| Physical vs relaxed | Strict transition versus the same initialized transition plus learned corrections | A tie does not establish superiority of strict enforcement |
| Physical vs offset, with learning curves | Shared analytic motion structure and physical initialization versus unconstrained temporal offsets | A short-budget gap may reflect optimization rather than final achievable error |

The unconstrained-offset and relaxed controls share the same nine-dimensional
observation summary and offset MLP. Their learned offset is
\(40[g_\theta(\mathbf s,\Delta t/40)-g_\theta(\mathbf s,0)]\), which forces
zero displacement at zero elapsed time. The relaxed control adds
\(\Delta t\hat{\mathbf v}\) and initializes its last offset layer to zero;
the offset-only control does not add the analytic term. Thus relaxed and
physical have exactly matched initial motion, while offset-only starts with a
different motion prior. Their comparison should be described accordingly.

The importance of reasonable tuning is already visible in completed validation
records: at tuning seed 2026, offset-only best weighted validation loss changes
from 0.078321 at learning rate 0.0003 to 0.004436 at 0.001, at the same 5,000-update
budget. This approximately 17.7-fold improvement is an optimization observation,
not a final test result. It prevents treating the original weak pilot offset
baseline as evidence of an expressivity limit. The completed 20k MLP run provides
a longer-budget diagnostic, while its single tuning seed, fixed selected
learning rate, and finite budget still preclude a convergence or expressivity
claim.

## Reviewer-facing claim matrix

| Level | Statement | Required qualification/evidence |
| --- | --- | --- |
| Architectural property | The fitted velocity is continuous and shared by alignment and direct transport | Initial correspondence attention is not conditioned on that fitted velocity; its search remains discrete and bounded |
| Architectural property | Canonical memory stays fixed and latent displacements compose in time | This does not guarantee exact image-warp composition or better empirical forecasts |
| Empirical question; completed evidence available | Which learned components improve image prediction? | Matched-budget comparisons are complete; mean fusion matches core image outcomes, so learned memory attention is not shown necessary |
| Empirical question; completed evidence available | Does removing velocity quantization benefit generated images? | The controlled fractional-speed interventions show clear image losses from quantization; fixed-memory transport comparisons identify that contribution |
| Empirical question; completed evidence available | Does strict physics improve over learned corrections or ensure transfer? | No clear strict-versus-relaxed primary image advantage; bilinear-to-nearest transfer remains a failure case, so a general robustness claim is unsupported |
| Unsupported extension | The MLP cannot learn the task, the oracle is an image-quality bound, or the architecture is universally superior | Finite-budget failure is not an expressivity proof; oracle inputs can interact with learned compensation |
| Unsupported extension | Results establish general world modeling or superiority to matched-trained FloWM | Multiple motion components, new assumptions and equally trained references remain untested |

## Alternative explanations and decisive evidence

The simple environment may be solved largely by motion-compensated copying.
If the raw-untrained baseline is competitive, the contribution cannot be
described as discovering motion solely through learned attention. Learned
features might add value mainly under partial overlap, rendering shifts or
ambiguous appearances; those conditions need direct measurements.

Training on bilinear images can specialize matching to interpolation-induced
appearance patterns. Nearest rendering introduces quantized per-pair
displacements even when the underlying velocity is constant. A rendering
train/test matrix, matcher swaps, and oracle-velocity diagnosis distinguish this
observation shift from failure of continuous transport.

The raw intensity skip and fixed canonical decoder may explain improvements
over a recurrent predictor independently of the physical fit. The matched MLP,
memory averaging and relaxed controls are therefore more informative about the
new mechanism than a comparison against a frozen FloWM checkpoint alone.

This implementation builds the canonical memory from a batch of context
observations, then queries that fixed memory throughout the forecast. It does
not evaluate a streaming recurrent update rule that incorporates newly arriving
observations indefinitely. Persistence within one forecast should therefore
be distinguished from FloWM's online recurrent memory setting; memory capacity,
streaming computation and online adaptation are not matched in this campaign.

The completed evidence should be interpreted component by component. The clear
controlled cost of velocity quantization supports continuous transport in this
task, while the physical/relaxed tie and competitive mean fusion narrow claims
about strict enforcement and learned memory attention. Nearest-rendering
transfer remains a limitation. Long-horizon and directional results require
their own stated domains, seed counts, uncertainty and evaluator coverage;
they cannot erase those negative controls. The final findings report preserves
these distinctions rather than treating all completed experiments as positive
confirmation.

These experiments address one rigid object, a single global velocity, known
camera translations, periodic boundaries, and informative context. They do not
establish multi-object association, acceleration, deformation, uncertainty
representation, or general-purpose world-model capabilities. Those extensions
would require additional state structure and experiments.
