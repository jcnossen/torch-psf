# Estimating the PSF from the data itself: sparse SMLM, arbitrary samples, dense blinking movies

Theory note, 2026-11-15. No code. The focus is case (c): a movie of blinking emitters that are too dense to
localize. Section 6 proposes how to get the PSF out of such a movie, and section 8 brainstorms the requirements
for a web tool. Statements marked **[speculative]** are my own reasoning and not taken from a paper. References at
the end: DOIs were checked by web search on 2026-11-15 unless marked *(m)*, meaning written from memory and not
checked.

**Contents.** 1 Image formation and the blind ambiguity · 2 Symmetries that no data set can break ·
3 Sparse SMLM · 4 Arbitrary samples · 5 Dense blinking: what the cumulants contain ·
6 Proposed estimator for a dense movie · 7 Comparison table · 8 Web tool: requirements brainstorm ·
9 Summary.

---

## 1. Image formation and the blind-deconvolution ambiguity

Incoherent fluorescence imaging is linear in the emitter density $o(\mathbf r, z)$. In a space-invariant model
(index-matched; see Gibson & Lanni for the mismatched case) the image taken at focus position $z_f$ is

$$
I(\mathbf r; z_f) = \int\!\!\int o(\mathbf r', z')\, h(\mathbf r - \mathbf r';\, z' - z_f)\, d^2 r'\, dz' + b ,
$$

so each object plane is convolved with its own defocus slice of the 3D PSF $h$. In a scalar model the PSF follows
from the pupil:

$$
h(\mathbf r; z) = \Big| \mathcal F^{-1}\big\{ A(\boldsymbol\rho)\, e^{i W(\boldsymbol\rho)}\, e^{i k_z(\boldsymbol\rho) z} \big\}(\mathbf r) \Big|^2 ,
\qquad W = 2\pi \sum_j c_j Z_j ,
\qquad k_z = \tfrac{2\pi}{\lambda}\sqrt{n^2 - \lambda^2\lvert\boldsymbol\rho\rvert^2} ,
$$

where the pupil coordinate $\boldsymbol\rho$ is a spatial frequency, the pupil is a disk of radius $\mathrm{NA}/\lambda$, the $Z_j$ are Noll-normalized Zernikes [Noll] and the $c_j$ are in waves. A vectorial model is a sum of such terms over the field components, with real
pupil-amplitude factors that are either even or odd in $\boldsymbol\rho$.

For a thin object (one plane), the 2D Fourier transform gives $\hat I(\mathbf k) = \hat O(\mathbf k)\,\hat H(\mathbf k)$.
The OTF $\hat H$ is the autocorrelation of the pupil, supported on $\lvert\mathbf k\rvert \le 2\mathrm{NA}/\lambda$.
**Blind deconvolution** must split one product into two factors. For any invertible kernel $g$,

$$
(o,\, h) \;\to\; (o * g^{-1},\; h * g)
$$

gives the same noise-free image, and the power spectrum shows only $\lvert\hat O\rvert\,\lvert\hat H\rvert$. Every
method below is a way to shrink this family of solutions to one point. Three things can do that:
(i) **constraints on $h$**: positivity, band limit, and above all $h = \lvert\text{FT of a pupil}\rvert^2$ with
about 15-20 parameters; (ii) **constraints on $o$**: sparsity, positivity, statistics; (iii) **diversity**:
several data sets in which the same unknown object is seen through different, partly known kernels.

A counting comparison sets the scale. With a pupil model the PSF has $\sim 17$ unknowns (Noll 5-21). An arbitrary
2D object has as many unknowns as there are pixels. A sparse SMLM frame has $\sim 5$ unknowns per spot of $P^2$
pixels.

Noise-free 2D convolutions with compact support are generically factorizable, up to trivial ambiguities
[Lane & Bates 1987]. That result is of little practical use: it depends on the zero sheets of $\hat I$, which noise
destroys. Naive joint MAP estimation of $(o, h)$ under sparse image priors also has a known failure: its global
optimum is the "no blur" solution $h = \delta$, $o = I$. What works is to estimate $h$ with $o$ marginalized or
profiled out, which is well constrained because $h$ has few unknowns [Levin et al. 2009]. That principle runs
through all of the cases below.

## 2. Symmetries that no data set of this kind can break

These hold for any data on emitters at **unknown** depths: sparse SMLM, dense blinking movies, and thick samples.

1. **Gauge modes.** Tip and tilt (Noll 2, 3) are equivalent to a lateral shift of all emitters. Defocus (the exact
   $k_z z$ phase, which at high NA is approximately Noll 4) is equivalent to a common $z$ offset. Piston has no effect. That
   is why the fit uses Noll 5-21. A free spline PSF has more such freedoms: a lateral shift per plane, a
   relabelling of $z$, and a brightness per plane (`docs/insitu_spline_psf.md` §2.4). The Fisher matrix is
   singular along these directions, and no unbiased finite-variance estimator exists for them
   [Stoica & Marzetta 2001]. They have to be fixed by convention.

2. **Mirror symmetry (twin image).** Assume the pupil amplitude is even ($A(-\boldsymbol\rho) = A(\boldsymbol\rho)$)
   and the dipoles rotate freely. Then
   $$ W(\boldsymbol\rho) \to -W(-\boldsymbol\rho), \qquad z \to -z $$
   leaves every slice $h(\mathbf r; z)$ unchanged. The proof uses $\mathcal F\{f^*(-\boldsymbol\rho)\} = (\mathcal F f)^*$
   on each field component. In Zernike terms the **even** modes (defocus, astigmatism, spherical, ...) change sign,
   the **odd** modes (coma, trefoil, ...) do not, and all emitter depths are mirrored. Blinking data alone cannot
   tell the two solutions apart. One needs an outside reference, such as the known orientation of an astigmatic
   lens, a biplane offset of known sign, or the known direction of a stage step. INSPR uses biplane data or astigmatism
   with known orientation, and Vectorfit/NAT fixes the sign by convention (`docs/insitu_psf_literature.md`).
   Tilted fixed dipoles and pupil apodization that is not even break the symmetry, but only weakly.

3. **No first-order signal from even modes in focus.** Near $W = 0$ and $z = 0$, the derivative is
   $\partial h/\partial c_j = 4\pi\,\mathrm{Re}\{a^*\, \mathcal F[i A Z_j]\}$ with $a = \mathcal F[A]$ real. For an even
   $Z_j$, $\mathcal F[AZ_j]$ is real, so the derivative vanishes. For an odd $Z_j$ it is imaginary, so the derivative
   does not vanish. In-focus data therefore give Fisher information on even modes that goes to zero as $c_j^2$ near
   zero: the magnitude is fixed, the sign is not, and the CRLB diverges. Defocus, meaning emitters spread in $z$,
   or a known bias aberration turns this into a first-order effect. That is the role of **phase diversity**
   [Gonsalves 1982]. A sample confined to one plane, such as TIRF on a coverslip, needs an engineered bias
   (astigmatism) to measure even modes.

## 3. Sparse SMLM: why it is identifiable

**Counting.** $N$ isolated spots on $P\times P$ pixels give $NP^2$ data values for $5N + 17$ unknowns
($x, y, z, N_\text{ph}, b$ per spot, plus shared Zernikes). With $P = 13$ that is 169 values per spot against 5
nuisance parameters. Each spot is in effect a sample of the PSF itself: sparsity reduces the object to a sum of
deltas, and the blind ambiguity disappears because $g$ would have to map deltas to deltas.

**Diversity.** The emitters sit at different unknown depths, so the data set is a phase-diversity
experiment with unknown but estimable diversities $k_z z_n$. Item 3 of §2 is why a spread in $z$ is needed, and
VISPR "fails if all molecules [are] at one z" (`docs/insitu_psf_literature.md`).

**Information.** For Poisson data,
$I_{ab} = \sum_\text{pix} \partial_a\mu\, \partial_b\mu / \mu$. Let $\theta$ be the pupil parameters and $\eta_n$
the nuisance parameters of spot $n$. Profiling $\eta_n$ out gives the *efficient* information

$$
J(\theta) = \sum_n \Big[ I_{\theta\theta} - I_{\theta\eta}\, I_{\eta\eta}^{-1}\, I_{\eta\theta} \Big]_n (z_n, N_n, b_n),
\qquad \mathrm{Cov}(\hat\theta) \succeq J^{-1}.
$$

The information adds over spots, and each spot's share depends on its depth. $J$ is thus a weighted average over
the $z$ distribution of the emitters, which says directly which modes a given sample can determine. Photon statistics
are rarely the limit **[order-of-magnitude, own estimate]**. The quantum Fisher information per photon for a phase
mode $2\pi c_j Z_j$ with unit-RMS $Z_j$ is at most $4\,\mathrm{Var}(2\pi Z_j) = 16\pi^2$, so
$\sigma_{c_j} \gtrsim 1/(4\pi\sqrt{N_\text{tot}})$ waves. For $10^4$ spots of $10^3$ photons that is about
$3\times10^{-5}$ waves. The errors actually reported are much larger: INSPR reports 15 ± 6 mλ RMS pupil error
[Xu et al. 2020]. The limits are **model error** (field and depth dependence, pupil amplitude, dipoles, camera
model, background model) and the decisions about which spots count as single. Detection statistics are not.

**Moderate density: the bridge.** The deviance test in `InSituEstimator` rejects ROIs that a single emitter does
not explain. Those ROIs still contain PSF information. Fitting $K$ emitters per ROI jointly with the shared pupil
[Huang et al. 2011; Holden et al. 2011] keeps them, and the information adds as before, with a Schur complement
over $5K$ nuisance parameters. Emitters with near-identical PSF footprints make $I_{\eta\eta}$ ill-conditioned. Their
share then goes to zero, but it never becomes negative, so including them cannot hurt in principle. In practice
the risk is a wrong $K$, so model selection (likelihood ratio or deviance, as now) stays essential. This works up
to the point where $5K$ approaches the number of informative pixels. Beyond that only the time axis can help (§5).

## 4. Arbitrary samples (one image or a z-stack of an unknown object)

**A single image** constrains the OTF *support* (the edge at $2\mathrm{NA}/\lambda$ appears in the power spectrum
as the transition to the noise floor) and, given a model of the object's power spectrum
$\mathbb E\lvert\hat O\rvert^2$ (smooth, isotropic, often a power law), the **MTF** $\lvert\hat H\rvert$. The
**phase transfer function** is lost, and with it the odd modes and the sign of the even modes. Cryo-EM is the
instructive contrast: CTFFIND estimates defocus and astigmatism from single images of unknown objects
[Rohou & Grigorieff 2015], because the phase-contrast CTF oscillates and leaves zero rings in the power spectrum.
A fluorescence OTF is non-negative and smooth for moderate aberrations, so a single fluorescence image carries far
less PSF information **[analogy; own reasoning]**. Parametric blind deconvolution with few PSF parameters
[Markham & Conchello 1999] or a pupil-parameterized 3D PSF [Soulez et al. 2012] makes the problem well posed in
principle. Its accuracy depends on the object prior.

**Phase diversity** [Gonsalves 1982; Paxman, Schulz & Fienup 1992] takes $D$ images of the *same* object with known
added pupil phases $\phi_d$: $\hat I_d = \hat O\, \hat H_d(\theta)$. For fixed $\theta$ the object can be eliminated
in closed form, which gives the object-free error metric

$$
E(\theta) = \sum_{\mathbf k} \Big[ \sum_d \lvert\hat I_d\rvert^2 - \frac{\lvert\sum_d \hat H_d^* \hat I_d\rvert^2}{\sum_d \lvert\hat H_d\rvert^2 + \gamma} \Big],
\qquad D = 2:\;\; \frac{\lvert\hat I_1 \hat H_2 - \hat I_2 \hat H_1\rvert^2}{\lvert\hat H_1\rvert^2 + \lvert\hat H_2\rvert^2}.
$$

This is the cleanest object-independent PSF estimator there is. Its Cramér-Rao analysis, including the choice of
diversity, is in [Lee, Roggemann & Welsh 1999]. In a microscope, diversity *in the pupil plane* (a deformable mirror
or SLM) keeps the object fixed. A **z-step in a thick sample is not pure diversity.** Moving the focus brings other
object planes into focus, so $\hat I_d(\mathbf k) = \sum_z \hat O_z(\mathbf k)\, \hat H_{z - z_d}(\mathbf k)$, a 3D
convolution with $N_z$ unknowns per $\mathbf k$ instead of one. With index mismatch the aberration also changes with
focus depth [Gibson & Lanni 1991]. Only a thin layer (beads on glass, a single membrane) turns a z-stack into true
defocus diversity, which is what bead phase retrieval does [Hanser et al. 2004]. For a thick widefield sample the
elimination still works at high $\mathbf k$, where only near-focus planes contribute, and fails at low $\mathbf k$
**[own reasoning]**. Optical sectioning (confocal, two-photon, light sheet) makes the effective object thin, which
is why image-based methods work best there.

**Sensorless / modal AO** [Booth 2014; Débarre et al. 2007; Facomprez et al. 2012] estimates the **correction**,
not the PSF. The metric $M(\mathbf b) = \sum_{\mathbf k} w(\mathbf k)\, \lvert\hat I(\mathbf k; \mathbf b)\rvert^2$
has expectation $\sum_{\mathbf k} w\, S_O\, \lvert\hat H(\mathbf k; \mathbf c + \mathbf b)\rvert^2$. The MTF of an
aberrated system never exceeds the diffraction-limited MTF at any frequency [Goodman, §6.4], so $M$ is maximal at
$\mathbf b = -\mathbf c$ **for every object**. The object only sets the curvature of the maximum, hence the
precision. Near the optimum $M$ is quadratic in $\mathbf c + \mathbf b$, so about $2N_\text{modes}+1$ images
suffice. Débarre et al. tune which aberration magnitudes the metric is sensitive to through the frequency band of
$w$, and Facomprez et al. report accurate correction up to about 2 rad RMS with $10^4$-$10^5$ photons. In SMLM the
same idea has been applied to blinking data [Burke et al. 2015]. Learned estimators [Saha et al. 2020] trade
explicit diversity for a prior. Saha et al. find that the symmetry of the mode decides what can be predicted, which
is item 3 of §2.

## 5. Dense blinking movies: what the cumulants contain

### 5.1 Model

$$
F(\mathbf r_i, t) = \sum_k \epsilon_k\, s_k(t)\, h(\mathbf r_i - \mathbf x_k;\, z_k) + b(\mathbf r_i) + n_i(t),
$$

with independent blinking traces $s_k(t)$, static background $b$, and noise $n_i(t)$ that is independent between
pixels and frames: shot noise, plus read noise on sCMOS. Cumulants are additive over independent sources and blind to constants. For
pixels $\mathbf r_1 \dots \mathbf r_n$ at time lags $\tau_j$,

$$
\kappa_n(\mathbf r_1, \dots, \mathbf r_n) = \sum_k \epsilon_k^n\, \kappa_n[s_k](\boldsymbol\tau) \prod_{j=1}^n h(\mathbf r_j - \mathbf x_k; z_k) \;+\; \text{noise terms only when all } (\mathbf r_j, t_j) \text{ coincide}.
$$

Background drops out at every order $n\ge2$. Shot and read noise drop out of every *cross*-cumulant (distinct pixels or
distinct frames) [Dertinger et al. 2009, 2010]. This is the property that matters most here: **temporal
independence of the emitters takes the place of spatial sparsity.** In SMLM the emitters are separated because
they are switched on at different times. In a cumulant they are separated because terms from different emitters
never multiply.

### 5.2 Second order: built-in diversity

Write the zero-lag covariance of two pixels as a function of their midpoint $\mathbf m$ and separation $\mathbf d$:

$$
G(\mathbf m, \mathbf d) = \sum_k \rho_k\, h(\mathbf m + \tfrac{\mathbf d}{2} - \mathbf x_k)\, h(\mathbf m - \tfrac{\mathbf d}{2} - \mathbf x_k)
= (\rho * h_{\mathbf d})(\mathbf m),
\qquad h_{\mathbf d}(\mathbf u) := h(\mathbf u + \tfrac{\mathbf d}{2})\, h(\mathbf u - \tfrac{\mathbf d}{2}),
$$

with the positive map $\rho_k = \epsilon_k^2\, \mathrm{Var}[s_k]$. For a thin sample, **each pixel separation
$\mathbf d$ is a separate image of the same unknown object $\rho$ seen through a different kernel $h_{\mathbf d}$,
and every kernel is a known function of the one unknown PSF.** This is exactly the phase-diversity structure of §4,
obtained for free from the movie:

$$
\hat G_{\mathbf d}(\mathbf k) = \hat R(\mathbf k)\, \hat H_{\mathbf d}(\mathbf k)
\quad\Longrightarrow\quad
\hat G_{\mathbf d_1} \hat H_{\mathbf d_2} - \hat G_{\mathbf d_2} \hat H_{\mathbf d_1} = 0 \quad\text{(object-free)}.
$$

Things to note:

* For a **Gaussian** PSF of width $\sigma$, $h_{\mathbf d}(\mathbf u) = e^{-\lvert\mathbf d\rvert^2/4\sigma^2} h^2(\mathbf u)$.
  The kernels differ only by a scalar, the "distance factor" of cross-cumulant SOFI [Dertinger et al. 2010], and
  the ratio between separations gives $\sigma$ without knowing the object. SOFIevaluator uses the decay of the
  cross-cumulants with pixel distance to estimate the PSF size [Moeyaert et al. 2020].
* For an **aberrated** PSF, $h_{\mathbf d}$ changes *shape* with $\mathbf d$ (coma, for example, makes it
  asymmetric along $\mathbf d$), so the set of separations carries shape information beyond a width.
  $\hat H_{\mathbf d}(\mathbf k)$ as a function of $(\mathbf d, \mathbf k)$ is the ambiguity function of $h$, and it
  determines a non-negative $h$ uniquely. With $\rho$ unknown, the claim that the *ratios*
  $\hat H_{\mathbf d_1}/\hat H_{\mathbf d_2}$ still determine a pupil-model $h$ (apart from the symmetries of §2)
  is **[conjecture, no proof found]**. Counting supports it: the data are 4D, $G(\mathbf m, \mathbf d)$, the
  unknowns 2D plus 17.
* The support of $\hat H_{\mathbf d}$ extends to $4\mathrm{NA}/\lambda$. That is the $\sqrt 2$ resolution gain of
  second-order SOFI, and it also means the PSF is probed at twice the usual frequency range.
* $\rho$ is brightness-weighted ($\epsilon^2$) and depends on blinking ($\mathrm{Var}[s]$). That is harmless for
  PSF estimation within one order, because $\rho$ is just some positive map. Heterogeneity only matters when orders
  are combined (bSOFI's balancing [Geissbuehler et al. 2012]).
* In a web search on 2026-11-15 I found **no method that estimates aberrations (Zernikes) from cumulants**. Only
  the PSF width (distance factor, SOFIevaluator) or a PSF used for deconvolution has been estimated that way.

### 5.3 Higher orders and the source-separation view

In matrix form the movie is $F = H\,S + \text{noise}$, where column $k$ of $H$ is the image of emitter $k$ and row
$k$ of $S$ is its trace. The pixel covariance $C = H\,\mathrm{diag}(\rho)\,H^\top$ determines $H$ only up to a
rotation, as in factor analysis. Third- and fourth-order cumulant tensors
$\sum_k \kappa_n[s_k]\epsilon_k^n\, \mathbf h_k^{\otimes n}$ remove that rotation. This is the identifiability of
independent component analysis, which requires non-Gaussian sources [Comon 1994]. Blinking is strongly
non-Gaussian. Lidke et al. (2005) separated closely spaced quantum dots this way and localized them.
**Each separated column is the PSF of one emitter at its own depth**, which leads back to the sparse in-situ
problem of §3. deconSTORM [Mukamel et al. 2012] and SPARCOM [Solomon et al. 2018] use the same independence for
the *object*, but they assume a known PSF.

### 5.4 The noise budget: why second order is the workhorse

Let $M$ be the mean number of emitters within one PSF area and $T$ the number of frames. The signal of an $n$-th
order cumulant grows as $M$. Its estimator standard deviation scales as $\mathrm{Var}^{n/2}/\sqrt T \propto M^{n/2}/\sqrt T$
(central limit theorem: the sum of many blinkers becomes Gaussian). Hence

$$
\mathrm{SNR}_n \;\propto\; \sqrt T\; M^{\,1 - n/2}:\qquad n=2\text{: independent of density};\quad n=3:\ M^{-1/2};\quad n=4:\ M^{-1}.
$$

This is my derivation, consistent with the scaling reported by Mikhalychev et al. (2024): a second-order SNR with
no clear density dependence, and fourth order $\propto \rho^{-1}$. **For dense samples, second-order cross-cumulants
are the robust statistic; third order is a refinement on small ROIs; ICA-style separation only works where the
local density is moderate.**

### 5.5 Depth: why z is poorly constrained, and what helps

In 3D, $G(\mathbf m, \mathbf d) = \int (\rho_z * h_{\mathbf d, z})(\mathbf m)\, dz$, with an unknown density per
depth. Two effects weaken the $z$ diversity on which item 3 of §2 depends:

1. **Cumulants down-weight out-of-focus emitters.** At defocus $\Delta$ the PSF spreads over an area $\propto \Delta^2$,
   so $\int h_{\mathbf d}\,d^2u \propto \Delta^{-2}$, while the mean image keeps $\int h = $ const. This is the
   optical sectioning of SOFI [Dertinger et al. 2009]. But the defocused emitters are exactly the ones that
   measure the even modes.
2. **Depth is no longer labelled per emitter.** In SMLM every spot has its own $z_n$. Here, for each $\mathbf k$,
   the data are $N_{\mathbf d}$ numbers $\hat G_{\mathbf d}(\mathbf k) = \sum_z \hat R_z(\mathbf k)\, \hat H_{\mathbf d,z}(\mathbf k)$
   with $N_z$ unknown $\hat R_z$. Density and PSF are jointly determined only if the $N_{\mathbf d} \times N_z$
   matrix $[\hat H_{\mathbf d,z}(\mathbf k)]$ is well conditioned **[own formulation]**. Without an engineered PSF and with small even
   aberrations, $h_z pprox h_{-z}$ (item 2 of §2), so the matrix is close to rank-deficient.

Remedies: an astigmatic or other engineered PSF (it decorrelates the depths and fixes the sign), a biplane or
multiplane camera (true known diversity; multiplane SOFI exists [Geissbuehler et al. 2014]), or a thin sample (TIRF,
where $\rho$ is 2D and the problem reduces to §5.2; it then needs an engineered bias to see the even modes).

### 5.6 Confounders specific to fluctuation data

* **Bleaching** makes the mean non-stationary and creates spurious positive covariance. Compute cumulants in
  blocks of a few hundred frames, as SOFI tools do.
* **Drift** smears $\rho$ (harmless for a static fit if corrected per block) but also mixes separations (harmful).
  Register the blocks first.
* **Camera.** Pixel-independent noise (shot, sCMOS read noise, EMCCD excess noise) enters only $\mathbf d = 0$,
  zero lag. **Correlated read noise** (sCMOS row or column patterns) enters $G(\mathbf m, \mathbf d)$ along one
  axis and mimics astigmatism. It has to be measured from a dark movie and subtracted. Charge spreading between
  pixels would look like PSF blur.
* **Blinking faster than the frame time** lowers $\mathrm{Var}[s]$ (less signal), but does not change the spatial
  structure. Time-lagged cumulants remove even the $\mathbf d = 0$ noise term, but need correlation across frames.
* **Field dependence.** The estimate is local: fit patches of about 10-20 µm and interpolate, as the sparse
  methods do.

## 6. Proposed estimator for a dense movie **[speculative: a design, not a published method]**

The information is there. The plan below gets it out with the statistic that survives density (second order)
and the structure that removes the object (diversity across separations):

1. **Pre-processing.** Convert to photons (offset, gain), subtract the dark-movie covariance (correlated read
   noise), register drift per block, and split the movie into blocks for bleaching.
2. **Accumulate** per block the mean $\bar F(\mathbf r)$ and the cross-covariances $G(\mathbf m, \mathbf d)$ for
   all separations with $0 < \lvert\mathbf d\rvert \le R$ ($R \approx$ PSF radius, e.g. 5 px gives about 40
   distinct $\mathbf d$ up to sign). Optionally add the lag-1 covariance including $\mathbf d = 0$. This is one
   streaming pass. The model for $G$ includes the pixel integration of $h$ and, for a pixel pair, the product of
   two pixel-integrated PSFs.
3. **Thin-sample fit (2D $\rho$).** For given $\theta$, eliminate $\hat R(\mathbf k)$ in closed form exactly as in
   phase diversity, with the separations $\mathbf d$ as diversity channels:
   $$
   E(\theta) = \sum_{\mathbf k} \Big[ \sum_{\mathbf d} w_{\mathbf d} \lvert\hat G_{\mathbf d}\rvert^2 - \frac{\lvert\sum_{\mathbf d} w_{\mathbf d} \hat H_{\mathbf d}^*(\theta)\, \hat G_{\mathbf d}\rvert^2}{\sum_{\mathbf d} w_{\mathbf d} \lvert\hat H_{\mathbf d}(\theta)\rvert^2 + \gamma} \Big].
   $$
   Minimize over $\theta$ (Noll 5-21) by gradient descent. This needs the gradient through $h \to h_{\mathbf d} \to \hat H_{\mathbf d}$,
   which the torch pupil model already provides. Then refine with the positivity of $\rho$ restored: a variable
   projection with non-negative least squares (NNLS) for $\rho$ in real space. The weights $w_{\mathbf d}$ should
   approximate the inverse covariance of the estimates, obtained by a block bootstrap. Least squares on moments is
   not maximum likelihood, but it is consistent, and its uncertainty can be bootstrapped.
4. **3D fit.** Replace $\hat R$ by $N_z$ depth layers and the scalar elimination by a small least-squares problem
   per $\mathbf k$. Report the condition number of $[\hat H_{\mathbf d, z}]$ as the "can this sample give z?"
   diagnostic (§5.5). Fix the even-mode sign from the user-supplied astigmatism sign or biplane offset.
5. **Optional, where the local density allows.** Use ICA or third-order cumulants on small ROIs, or multi-emitter
   fits on the sparsest frames, to obtain single-emitter footprints. Feed them, with their own $z$, to the
   existing in-situ estimator as extra data. This gives the per-emitter $z$ labels that the cumulant fit lacks.
6. **Validation.** Simulate a movie with the fitted PSF and the fitted $\rho$, then compare held-out blocks:
   cumulant residuals per $\mathbf d$, analogous to the held-out NLL of `InSituEstimator`.

What I expect, **[speculative]**: odd modes (coma, trefoil) and astigmatism should be well determined in a thin
sample, through the asymmetry and anisotropy of $h_{\mathbf d}$. Spherical aberration and higher even modes will
be weak unless there is depth spread or an engineered bias. The accuracy will be limited by the camera-noise
model and bleaching long before it is limited by photons. Before any of this is built, a Fisher-information
calculation for $\theta$ from $G$ on simulated dense movies (with $\rho$ profiled out, as in §3) would show which
modes are recoverable at which density, number of frames and $z$ spread. It costs little and should be done first.

## 7. Comparison

| Data | What is identifiable | Priors / diversity needed | Typical limit |
|---|---|---|---|
| Sparse SMLM, single-emitter ROIs | Full pupil (Noll 5-21), field-dependent; not tip, tilt, defocus; even-mode sign only with a reference | Pupil model; spread in $z$ or an engineered bias; known astigmatism or biplane sign | Model error (depth/field dependence, amplitude, dipoles); reported ~15 mλ RMS (INSPR) |
| Moderately dense SMLM, multi-emitter ROIs | Same as above, with information from overlapping spots kept | As above, plus model selection on $K$ | Wrong $K$; ill-conditioning when footprints coincide |
| Dense blinking, 2nd-order cross-cumulants, thin sample | Width (proven); full pupil shape plausible (conjecture); odd modes and astigmatism strongest | Pupil model; engineered bias for even modes; camera noise model | Bleaching, drift, correlated read noise; SNR independent of density but needs many frames |
| Dense blinking, 3D | PSF and density per depth only if the separation x depth matrix is well conditioned | Engineered PSF or multiplane; sign reference | $z$ weakly constrained; out-of-focus emitters down-weighted |
| Dense blinking, 3rd/4th order or ICA | Individual emitter footprints, then as in SMLM | Non-Gaussian blinking; moderate local density | SNR falls as $M^{1-n/2}$ |
| Arbitrary sample, one image | Cutoff (NA); MTF given an object-spectrum prior; no phase | Object prior (power spectrum, sparsity) or a parametric PSF | Object prior; sign of even modes and odd modes lost |
| Arbitrary sample, pupil-plane diversity (DM/SLM) | Full pupil, object-free (thin or sectioned sample) | Known diversities | Thick widefield samples: low $\mathbf k$ fails |
| Arbitrary sample, z-stack | Thin layer: as above. Thick: 3D blind deconvolution with a parametric PSF | Parametric pupil; positivity | Depth-dependent aberrations; missing cone |
| Sensorless AO | The *correction* (the metric maximum), not the PSF | Quadratic metric model, mode set | ~$2N+1$ images; good up to ~2 rad RMS (Facomprez) |

## 8. Web tool: requirements brainstorm (no design decisions yet)

The existing page (`web/`, WebGPU, an `InSituEstimator` port) already has an optics panel, a movie loader, Zernike
bars and a PSF JSON export. A dense-movie mode could reuse all of that. Points to settle:

**Inputs**
* Raw frames: no denoising, no background subtraction, and no binning after acquisition. Gain and offset in
  photons, or a dark and a flat movie. A **dark movie is needed** for the correlated-read-noise covariance (§5.6).
* Optics as now (NA, λ, refractive indices, pixel size), plus the **sign reference**: astigmatism orientation,
  biplane offset, or "2D only".
* Sample geometry hint: thin (TIRF) versus 3D, and the expected z range. This decides between the fits of
  steps 3 and 4 in §6.
* Frame time and approximate blinking rate, which decides between zero-lag and lagged statistics.

**Computation (sizes as orders of magnitude)**
* A movie of $10^4$ frames at $256^2$ px in uint16 is about 1.3 GB, so it has to be **streamed** (the movie
  loader already streams TIFF). Second-order accumulators for ~40 separations take $256^2 \times 40 \times$
  (sums, plus per-block storage) and fit in tens of MB on the GPU. The accumulation is about $T \cdot P \cdot N_{\mathbf d} \approx 3\times10^{10}$
  multiply-adds, seconds on a GPU. Reading the file will dominate.
* Third order on the full field is about 1 GB of accumulators, so it would run only on a user-selected ROI.
* The fit works on the accumulated $G$, not on the frames, so it can be repeated cheaply with other settings
  (mode set, thin versus 3D, patches).
* The model is $\hat H_{\mathbf d}(\theta)$ for all $\mathbf d$ (and $z$ layers) and needs gradients with respect
  to the Zernikes. This is new WGSL code. Decide whether to port it after it has been proven in torch.

**Outputs and diagnostics**
* Zernikes with block-bootstrap error bars, plus the Fisher eigen-spectrum: "which modes this movie determines".
* The 3D PSF in the existing view and a PSF JSON compatible with the localizer.
* The fitted $\rho$, which is a second-order SOFI-like image, as a by-product. Residual maps of
  $G(\cdot, \mathbf d)$ per separation, so that camera artifacts become visible.
* Warnings: non-stationarity (bleaching or drift detected), correlated noise found in the dark movie, and an
  ill-conditioned depth matrix ("z not determinable from this sample").

**Validation**
* Extend the demo simulator to dense blinking (density, on/off rates, bleaching, correlated read noise) with a
  known PSF. Test whether the truth is recovered as a function of density and $T$, and whether the result agrees
  with the sparse `InSituEstimator` on a movie that is dense early and sparse late.

**Open questions for you**
1. Typical sample: TIRF/2D or volumetric? And with or without an astigmatic lens?
2. What is the PSF for: localizing the sparse part of the same movie, SOFI deconvolution, or AO correction?
   The required accuracy differs a lot.
3. Camera type, and are dark movies available?
4. Should the estimator be proven in torch first, starting with the Fisher-information study of §6, and ported
   to the browser only afterwards?

## 9. Summary

1. Blind PSF estimation is identifiable only through a low-dimensional pupil model combined with sparsity,
   temporal independence or diversity. Gauge modes and the mirror symmetry (even-mode sign against z flip) need a
   reference in every case.
2. Sparse SMLM works because each spot is a sample of the PSF at its own depth. The precision is limited by model
   error, not by photons.
3. For arbitrary samples, only pupil-plane diversity on a thin or sectioned object gives an object-free full PSF.
   A single image gives about the MTF, and sensorless AO gives the correction, not the PSF.
4. Dense blinking movies replace spatial sparsity with temporal independence. Each pixel separation in the
   second-order cross-cumulants is a free diversity channel. A phase-diversity-style fit that eliminates the
   object (§6) is a plausible, apparently unpublished, way to get the Zernikes, and its SNR does not degrade with
   density.
5. Depth is the weak point of the fluctuation route: cumulants down-weight defocused emitters and carry no
   per-emitter z. An engineered PSF or multiplane detection, plus a sign reference, is needed for 3D.
   A Fisher-information study should precede any implementation.

---

## References

* Booth MJ. Adaptive optical microscopy: the ongoing quest for a perfect image. *Light Sci. Appl.* 3, e165 (2014). doi:10.1038/lsa.2014.46
* Burke D, Patton B, Huang F, Bewersdorf J, Booth MJ. Adaptive optics correction of specimen-induced aberrations in single-molecule switching microscopy. *Optica* 2, 177 (2015). doi:10.1364/OPTICA.2.000177
* Comon P. Independent component analysis, a new concept? *Signal Process.* 36, 287 (1994). doi:10.1016/0165-1684(94)90029-9
* Débarre D, Booth MJ, Wilson T. Image based adaptive optics through optimisation of low spatial frequencies. *Opt. Express* 15, 8176 (2007). doi:10.1364/OE.15.008176
* Dertinger T, Colyer R, Iyer G, Weiss S, Enderlein J. Fast, background-free, 3D super-resolution optical fluctuation imaging (SOFI). *PNAS* 106, 22287 (2009). doi:10.1073/pnas.0907866106
* Dertinger T, Colyer R, Vogel R, Enderlein J, Weiss S. Achieving increased resolution and more pixels with Superresolution Optical Fluctuation Imaging (SOFI). *Opt. Express* 18, 18875 (2010). doi:10.1364/OE.18.018875 *(m: DOI and pages)*
* Facomprez A, Beaurepaire E, Débarre D. Accuracy of correction in modal sensorless adaptive optics. *Opt. Express* 20, 2598 (2012). doi:10.1364/OE.20.002598
* Geissbuehler S, Bocchio NL, Dellagiacoma C, Berclaz C, Leutenegger M, Lasser T. Mapping molecular statistics with balanced super-resolution optical fluctuation imaging (bSOFI). *Opt. Nanoscopy* 1, 4 (2012). doi:10.1186/2192-2853-1-4
* Geissbuehler S, et al. Live-cell multiplane three-dimensional super-resolution optical fluctuation imaging. *Nat. Commun.* 5, 5830 (2014). doi:10.1038/ncomms6830
* Gibson SF, Lanni F. Experimental test of an analytical model of aberration in an oil-immersion objective lens used in three-dimensional light microscopy. *JOSA A* 8, 1601 (1991). doi:10.1364/JOSAA.8.001601 *(m)*
* Gonsalves RA. Phase retrieval and diversity in adaptive optics. *Opt. Eng.* 21(5) (1982). doi:10.1117/12.7972989 (DOI from an aggregator)
* Goodman JW. *Introduction to Fourier Optics*, 3rd ed., §6.4 (Roberts & Co., 2005) *(m: section number)*
* Hanser BM, Gustafsson MGL, Agard DA, Sedat JW. Phase-retrieved pupil functions in wide-field fluorescence microscopy. *J. Microsc.* 216, 32 (2004). PMID 15369481
* Holden SJ, Uphoff S, Kapanidis AN. DAOSTORM: an algorithm for high-density super-resolution microscopy. *Nat. Methods* 8, 279 (2011). doi:10.1038/nmeth0411-279
* Huang F, Schwartz SL, Byars JM, Lidke KA. Simultaneous multiple-emitter fitting for single molecule super-resolution imaging. *Biomed. Opt. Express* 2, 1377 (2011). doi:10.1364/BOE.2.001377
* Lane RG, Bates RHT. Automatic multidimensional deconvolution. *JOSA A* 4, 180 (1987). (DOI not checked)
* Lee DJ, Roggemann MC, Welsh BM. Cramér-Rao analysis of phase-diverse wave-front sensing. *JOSA A* 16, 1005 (1999). doi:10.1364/JOSAA.16.001005
* Levin A, Weiss Y, Durand F, Freeman WT. Understanding and evaluating blind deconvolution algorithms. *CVPR* 2009, 1964. doi:10.1109/CVPR.2009.5206815
* Lidke KA, Rieger B, Jovin TM, Heintzmann R. Superresolution by localization of quantum dots using blinking statistics. *Opt. Express* 13, 7052 (2005). doi:10.1364/OE.13.007052
* Markham J, Conchello JA. Parametric blind deconvolution: a robust method for the simultaneous estimation of image and blur. *JOSA A* 16, 2377 (1999). doi:10.1364/JOSAA.16.002377
* Mikhalychev A, Saushin A, Ulyanenkov A, Kuzhir P. Spatial super-resolution in nanosensing with blinking emitters. arXiv:2402.17391 (2024)
* Moeyaert B, Vandenberg W, Dedecker P. SOFIevaluator: a strategy for the quantitative quality assessment of SOFI data. *Biomed. Opt. Express* 11, 636 (2020). doi:10.1364/BOE.382278
* Mukamel EA, Babcock H, Zhuang X. Statistical deconvolution for superresolution fluorescence microscopy. *Biophys. J.* 102, 2391 (2012). doi:10.1016/j.bpj.2012.03.070
* Noll RJ. Zernike polynomials and atmospheric turbulence. *JOSA* 66, 207 (1976). doi:10.1364/JOSA.66.000207 *(m)*
* Paxman RG, Schulz TJ, Fienup JR. Joint estimation of object and aberrations by using phase diversity. *JOSA A* 9, 1072 (1992). (DOI not checked; PDF on the Fienup lab site)
* Rohou A, Grigorieff N. CTFFIND4: fast and accurate defocus estimation from electron micrographs. *J. Struct. Biol.* 192, 216 (2015). doi:10.1016/j.jsb.2015.08.008
* Saha D, Schmidt U, Zhang Q, et al. Practical sensorless aberration estimation for 3D microscopy with deep learning. *Opt. Express* 28, 29044 (2020). doi:10.1364/OE.401933
* Solomon O, Mutzafi M, Segev M, Eldar YC. Sparsity-based super-resolution microscopy from correlation information. *Opt. Express* 26, 18238 (2018). doi:10.1364/OE.26.018238
* Soulez F, Denis L, Tourneur Y, Thiébaut É. Blind deconvolution of 3D data in wide field fluorescence microscopy. *Proc. IEEE ISBI* 2012, 1735 (HAL hal-00691249)
* Stoica P, Marzetta TL. Parameter estimation problems with singular information matrices. *IEEE Trans. Signal Process.* 49, 87 (2001). doi:10.1109/78.890346
* Xu F, Ma D, et al. Three-dimensional nanoscopy of whole cells and tissues with in situ point spread function retrieval. *Nat. Methods* 17, 531 (2020). doi:10.1038/s41592-020-0816-x
* In-situ SMLM methods (INSPR, VISPR, uiPSF, Vectorfit/NAT, ...): see `docs/insitu_psf_literature.md`.
