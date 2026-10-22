# V8: detail is not export resolution

This branch retains the provided numerical V7 parent solver. Its additional
fine scalar stage addresses the small number of transported flame features,
not just the number of pixels in a camera image.

## What actually runs

1. A 160 x 280 x 160 staggered-grid parent advances velocity, pressure, chemistry,
   and sources. Its immutable velocity snapshots carry physical timestamps.
2. A separate 384 x 672 x 384 grid stores four **FP32** fields: fuel mass density,
   oxygen deficit, sensible energy density, and soot mass density. There are
   99,090,432 fine cells, spaced 5 mm apart over the same 1.92 x 3.36 x 1.92 m box.
3. Time-interpolated parent velocities and an explicit synthetic transverse-mode
   turbulence model drive fine-grid scalar advection. Scalars undergo their own
   source exchange, nonlinear chemistry, and thermal losses on the fine grid.
4. A bounded tensor-product cubic Hermite interpolant carries those fields.
   This has a different diffusion/bounds tradeoff from the parent's bounded
   MacCormack path. It is not a conservative finite-volume method.
5. The ray integrator samples transported temperature/soot before nonlinear
   Planck/CIE conversion. It does not warp density, add detail to images, sharpen
   pixels, use image generation, or blend old screenshots.

This is **not** a 99-million-cell pressure/velocity solve. The pressure grid has
7,168,000 cells. Fine scalar heat does not feed back into the parent momentum.
The transverse modes are a disclosed VFX turbulence closure, not experimentally
resolved turbulence or a validated LES model. They are not the wavelet method of
Kim et al.; neither wavelet energy estimation nor advected wavelet coordinates
is implemented here.

The flow cache has 192 x 336 x 192 cells. The shortest mode wavelength is 4 cm,
which spans four cache cells and eight fine scalar cells. Mode direction and
polarization satisfy k dot p = 0 analytically. The floor taper is applied to a
vector potential with the additional gradient-of-window curl term retained; interpolation of cached sampled
velocities is not a promise of machine-exact divergence-free fine transport.

## Ownership and memory

The fine layer owns two complete FP32 scalar textures, a fine-flow cache,
and one temporally mixed parent velocity texture. The two scalar textures are
RGBA32F; the sampled parent-velocity and fine-flow caches are RGBA16F. The
parent solver and its disk snapshots remain FP32. Cache quantization is a
disclosed memory/accuracy tradeoff, not full-precision flow reconstruction. It never shares write access with the parent
solver. The parent finishes and closes before the fine process allocates its
large arrays. Parent data is loaded in bounded z slabs.

Rendering borrows the next-step scratch texture for the optical cache. The
current scalar state is read-only. Before the next transport operation every
scratch cell is overwritten. Lighting and render images have independent owners.
High-resolution images are integrated in native-coordinate row bands to keep
output buffers below the SSBO size limit. This is native rendering, not scaling.

Image-unit bindings and sampled-texture bindings have separate APIs. The initial
development test exposed the risk of assigning both through the same helper:
an output-image bind changed the input sampler. `bind_image()` now only changes
image binding state. The failing and repaired test reports are retained.

## Controls, invariants and evidence

`DetailConfig` declares shape, flow-cache ratio, mode RMS speed, wavelengths,
mode count, seed, cubic/linear reconstruction and characteristic work budget.
No shader switches on the example name. Zero mode strength is an explicit
ablation. An unsupported solid-containing parent is rejected, not ignored.

The initial fine state is interpolated from the provided starting state. New
fine cells do not instantly constitute new simulated detail: the capture runs
sources, advection and chemistry through a recorded warm-up before publishing
images. The frame journal records all accepted fine steps and physical times. It does
not save a resumable fine-field checkpoint; a failed fine process is rerun from
the declared parent cache.

The captured frame rate is 24 Hz, with two fine steps per captured interval.
Every displayed frame corresponds to a newly advanced state. No temporal
interpolation, AI upscaling, or slow-motion re-encoding is used.

Only tests that were actually executed are listed in the release report.
Test counts and full MP4 decoding are engineering checks, not physical validation.
No hardware-GPU or EmberGen performance win is established by software OpenGL.

## Reference principles (not implementation certification)

- Fedkiw, Stam, Jensen, *Visual Simulation of Smoke* (2001): monotonic cubic
  interpolation as a numerical option for reducing smoke transport dissipation.
  https://graphics.stanford.edu/papers/smoke/
- Kim, Thuerey, James, Gross, *Wavelet Turbulence for Fluid Simulation* (2008):
  separating low-frequency dynamics from modeled fine-scale transport.
  https://www.cs.cornell.edu/~tedkim/WTURB/
- Bridson, Hourihan, Nordenstam, *Curl-Noise for Procedural Fluid Flow* (2007):
  divergence-free modeled velocity fields. This branch uses transverse Fourier
  modes rather than the paper's Perlin-potential construction.
  https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph2007-curlnoise.pdf
- GPU Gems, chapter 39: volume reconstruction, classification and integration.
  https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-39-volume-rendering-techniques

## Accuracy diagnostics

The cached parent has localized high component speeds. These can exceed the
nominal displacement target used to choose a bounded number of characteristic
segments. That target is a work/accuracy control, not a demonstrated convergence
criterion. The release records the sampled maximum and the nominal budget.
Neither nonnegative fields nor a large cell count proves the reacting flow is
resolved in space or time.

The reported per-frame fine-simulation timings exclude the earlier parent-cache
build. Its separate measured snapshot/advance durations are included in the
validation and parent manifest. The whole pipeline is not claimed to be faster
than the previous full solver or a production effects package.
