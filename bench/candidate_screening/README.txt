Round 2: certified candidate screening and the original exact FP32 statistic

Baseline: 50cb1db (round 1: no padded tail batches; adjacent weighted-window reuse).
No constant-dy assumption. patchWeighted still produces z=FP32(w*r), w=FP32(1/max(dy^2,1e-10)).
The exact amplitude remains mean_depth*overshoot*2. It is not replaced by A/B.
The final margin=0 fit retains the round-1 implementation.

For each duration, approximate the two sampled targets independently on joint adaptive knots:
  s = stored FP32 template; b = FP32(s*s).
Their sparse second differences c_s,c_b evaluate the two correlations using FP64 double prefixes.
The nominal 1% fitting tolerance chooses knots only. The certificate uses the maximum measured
error against the actual rounded coefficient representation, including reconstruction roundoff.

Let a be the exact FP32 amplitude used by the old kernel, A=sum(s*z), B=sum(b*w).
For each window compute an interval about delta_proxy = a*(a*B_proxy-2*A_proxy):
  E_A = eps_s*sum(abs(z)) + FP64_prefix_evaluation_budget + gamma_m*max(abs(s))*sum(abs(z)) + tiny;
  E_B = eps_b*sum(w)      + FP64_prefix_evaluation_budget + gamma_m*max(abs(b))*sum(w)      + tiny;
  E_delta = a^2*E_B + 2*abs(a)*E_A + final_FP32_roundoff_budget.
m = ceil(width/G)+log2(G)+2 covers the original per-lane FMA chain and XOR reduction.
gamma_m = m*2^-24/(1-m*2^-24). The final expression uses a conservative 2^-21 relative budget.
Subnormal absolute errors, the FP64 expression, and outward conversion to FP32 are included.
Unsafe gamma values, nonfinite intervals, and possible FP32 overflow keep the candidate.

The FP64 prefix bound uses P*max(abs(x)), not the local window mass. Therefore it remains
valid for small windows following a huge weighted outlier. Two levels of the explicit 512-thread
scan have bounded arithmetic depth <=2*(ceil(P/512)+16); the budget uses four times the associated
gamma times P^2*max(abs(x)), plus coefficient evaluation error. First-prefix subtraction receives
its own absolute global allowance. Both w and z retain their individual supplied dy weights.

The screening kernel stores every lower bound and atomically minimizes the rounded-up upper
bounds per period. If U_min = min_i U_i, the global minimizer has delta_min <= U_min.
Discard only candidates with L_i > U_min. Equality is always retained. Thus an evaluated seed
candidate is not required, and at least one finite winning candidate survives. EMPTY upper keys
or unbounded lower intervals retain the original path. No top-K restriction is imposed.

Survivors run searchChi2Lanes<G> with its original input values, amplitude, FMA order, XOR tree,
and packed (delta,duration,start) tie order. Screening buffers are included in bytes_per_period.
Curves shorter than 8192 points, templates below 512 samples, and zero-margin fits bypass
screening where it would add overhead; the existing adjacent-window path remains available.

Verification:
  tests/test_candidate_screening.py checks each original candidate against its interval for
  G=1,2,4,8,16,32, unequal dy, variance-floor outliers, zero-template ties and overflow fallback;
  tests complete output fields, batch-size invariance, peak live memory and multiple GPUs.
  validate.py compares all 24 returned fields/masks, including complete raw/refined periodograms
  and fitted quantities, with the immutable first-round snapshot. Input files are hash verified.
  Warmup is excluded. Recorded E2E wall time includes validation, templates, transfers, all search
  batches, refinement, final fitting, and statistics; input-file I/O and cooling are excluded.

Campaign directory:
  /home/farthing/GTLS/GTLS-Compare/tls2-gtls-delta001-20261114/round2-candidate-screening-20261114
Benchmark example (use the campaign's installed Python/CUDA paths as in profile.sh):
  python bench/candidate_screening/validate.py --case 0365d_100s --pattern mixed --repeats 3 \
    --cool-to 65 --output <new-output-directory>
  CUDA_VISIBLE_DEVICES=<GPU-UUID> bash bench/candidate_screening/profile.sh nsys
  CUDA_VISIBLE_DEVICES=<GPU-UUID> bash bench/candidate_screening/profile.sh ncu
  python bench/candidate_screening/summarize.py
Profiler scripts preserve existing reports. NCU compares the same 815 periods starting at
coarse-grid index 12225 on one GPU, irrespective of either implementation's memory batch size.
NCU captures baseline G8/G16/G32 and the new prefix, screening, and exact G8/G16/G32 kernels.
NCU replay durations are not E2E timings; throughput percentages are not absolute traffic counts.
