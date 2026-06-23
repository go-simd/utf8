# Performance parity — go-simd/utf8 vs stdlib

**Methodology.** Apple M4 Max (arm64, NEON), macOS (Darwin 25.5.0), Go 1.26.4,
single core. Reference: `unicode/utf8` (Go stdlib — also the scalar fallback of
go-simd). `Valid` implements the Lemire–Keiser lookup validator; `RuneCount`
counts non-continuation bytes. Inputs: printable-ASCII fast-path and a
multibyte-mixed buffer (a/é/λ/世/🚀), seeds 2/3, sizes 64 B … 1 MiB;
`-benchtime=0.3s -count=3`, median reported. Correctness: `go test` byte-matches
`unicode/utf8.Valid` / `RuneCount` over ASCII, multibyte and every invalid class
(overlong, surrogate, too-large, lone continuation), plus a fuzz-validated
every-offset boundary sweep through the NEON kernel. Reproduce:

```
GOWORK=off go test -run='^$' -bench=Parity -benchmem -benchtime=0.3s -count=3 .
```

> **arm64 NEON kernel (this host).** As of 2026-06-22 go-simd/utf8 ships a real
> **arm64/NEON** kernel, alongside amd64/ppc64le/s390x: `validBlocks` is the
> Lemire–Keiser validator ported to NEON (VTBL nibble lookups; PSUBUSB, PCMPGTB
> and PALIGNR synthesised from VUMAX/VSUB, the sign-flip + VUMIN/VCMEQ trick, and
> VEXT, since the released arm64 assembler lacks those ops directly), `countCont`
> counts continuation bytes via VUADDLV, and a cheap 64-byte-unrolled ASCII
> pre-scan keeps the pure-ASCII fast path memory-bound. The numbers below are
> measured natively on this NEON host.

## Valid — ASCII fast path

| size | go-simd (GB/s) | stdlib (GB/s) | ratio | verdict |
|------|---------------:|--------------:|------:|---------|
| 64 B   |  17.7 |  8.0 | 2.21× | NEON wins |
| 1 KiB  |  99.0 | 62.8 | 1.58× | NEON wins |
| 16 KiB | 116.0 | 89.2 | 1.30× | NEON wins |
| 1 MiB  | 105.3 | 87.1 | 1.21× | NEON wins |

## Valid — multibyte mixed

| size | go-simd (GB/s) | stdlib (GB/s) | ratio | verdict |
|------|---------------:|--------------:|------:|---------|
| 64 B   | 3.94 | 2.07 | 1.90× | NEON wins |
| 1 KiB  | 3.94 | 1.62 | 2.43× | NEON wins |
| 16 KiB | 3.88 | 1.48 | 2.62× | NEON wins |
| 1 MiB  | 3.87 | 0.43 | **9.0×** | NEON wins |

## RuneCount — multibyte mixed

| size | go-simd (GB/s) | stdlib (GB/s) | ratio | verdict |
|------|---------------:|--------------:|------:|---------|
| 64 B   | 3.50 | 0.92 | 3.81× | NEON wins |
| 1 KiB  | 3.60 | 1.00 | 3.59× | NEON wins |
| 16 KiB | 3.59 | 1.00 | 3.58× | NEON wins |
| 1 MiB  | 3.59 | 0.41 | **8.7×** | NEON wins |

## Before → after (arm64)

Prior to the NEON kernel, go-simd/utf8 *was* `unicode/utf8` on arm64
(zero-overhead stdlib fallback), so go-simd ≈ stdlib by construction:

| op (1 MiB) | before (×stdlib) | after (×stdlib) |
|------------|-----------------:|----------------:|
| Valid ASCII | 1.00× (fallback) | **1.21×** |
| Valid mixed | 1.00× (fallback) | **9.0×** |
| RuneCount   | 1.00× (fallback) | **8.7×** |

## amd64 (AVX2, GitHub Actions x86_64 runner — ratios valid, absolute ns/op CI-noisy)

**Methodology.** GitHub Actions `ubuntu-latest` runner, **AMD EPYC** (`avx2`
present, **no `avx512*`** — confirmed from `/proc/cpuinfo`; the pool serves
different EPYC parts run-to-run, e.g. 7763 / 9V74, so absolute MB/s shifts
between runs), `GOAMD64=v1` baseline, Go stable, single core. Same parity
harness, `-count=6` (ASCII rows min-of-6, mixed/RuneCount max-of-6 throughput).
The runner is shared, so absolute throughput is noisy and **not comparable to the
arm64 M4 Max rows above** (different hardware/ISA); the **ratios vs stdlib** are
measured back-to-back on the *same* CPU in the *same* job and are valid.
Reproduce via `gh workflow run bench-amd64.yml`.

### Valid — ASCII fast path (amd64 AVX2)

The amd64 AVX2/SSE kernels now carry the same high-bit ASCII pre-scan as the
arm64 NEON path (`asciiBlocksAVX2` / `asciiBlocksSSE`: a group-of-4
`[V]PMOVMSKB` fast lane). The pure-ASCII case now streams at memory bandwidth
instead of paying full Lemire–Keiser validation, turning the prior regression
into a clear win.

| size | go-simd (MB/s) | stdlib | ×stdlib | verdict |
|------|---------------:|-------:|--------:|---------|
| 64 B   |   9820 |  3780 | 2.60× | wins |
| 1 KiB  |  90615 | 27129 | 3.34× | **wins** |
| 16 KiB | 153319 | 42021 | 3.65× | **wins** |
| 1 MiB  |  78284 | 37954 | 2.06× | **wins** |

> **Before → after (amd64 ASCII pre-scan).** Prior to this kernel the amd64
> Lemire–Keiser validator ran the full per-block check on every byte even on pure
> ASCII, with **no cheap pre-scan**, so it *lost* to stdlib's word-at-a-time
> high-bit scan by ~5–8× at ≥1 KiB (1 KiB 0.18×, 16 KiB 0.12×, 1 MiB 0.14×).
> Porting the arm64 ASCII pre-scan to amd64 (`asciiBlocksAVX2`/`asciiBlocksSSE`)
> lets the all-ASCII case stream at bandwidth (78–153 GB/s), so it now beats
> stdlib **2.1–3.7×** across all sizes. Both runs are min/max-of-6 on the same
> `ubuntu-latest` x86_64 runner; absolute ns/op is CI-noisy but the same-run
> ratios vs stdlib are valid.

| op (×stdlib) | 64 B | 1 KiB | 16 KiB | 1 MiB |
|--------------|-----:|------:|-------:|------:|
| before | 1.01× | 0.18× | 0.12× | 0.14× |
| after  | 2.60× | 3.34× | 3.65× | 2.06× |

### Valid — multibyte mixed (amd64 AVX2)

The ASCII pre-scan does not touch this path (a non-ASCII block is found in the
first group, after which the full validator runs as before); the numbers are
unchanged within shared-runner noise.

| size | go-simd (MB/s) | stdlib | ×stdlib | verdict |
|------|---------------:|-------:|--------:|---------|
| 64 B   | 4152 | 1134 |  3.66× | wins |
| 1 KiB  | 5259 | 1163 |  4.52× | wins |
| 16 KiB | 5183 | 1084 |  4.78× | wins |
| 1 MiB  | 5196 |  263 | 19.74× | **wins big** |

### RuneCount — multibyte mixed (amd64 AVX2)

| size | go-simd (MB/s) | stdlib | ×stdlib | verdict |
|------|---------------:|-------:|--------:|---------|
| 64 B   | 2811 | 584 |  4.81× | wins |
| 1 KiB  | 4460 | 677 |  6.59× | wins |
| 16 KiB | 4592 | 646 |  7.11× | wins |
| 1 MiB  | 4593 | 218 | 21.08× | **wins big** |

* On amd64 the **multibyte/RuneCount paths win 3.7–21.1×** vs stdlib (same shape
  as arm64) and the **pure-ASCII fast path now wins 2.1–3.7×** thanks to the
  ported ASCII pre-scan (was a ~5–8× regression — see the before→after note
  above).

## Summary

* The new **arm64/NEON kernel wins on every workload**: the multibyte/RuneCount
  paths — where the stdlib decoder drops to ~0.4–1.5 GB/s — see 1.9–9× speedups,
  and the ASCII fast path (the case a naive Lemire pass would *regress*) stays
  ahead of stdlib at 1.2–2.2× thanks to the 64-byte-unrolled high-bit pre-scan.
* Results are byte-identical to `unicode/utf8.Valid` / `RuneCount` on every
  input, including overlong / surrogate / too-large / lone-continuation errors
  and every ASCII→multibyte boundary (fuzz-clean, 100% coverage).

### Action items
1. ~~Add an arm64/NEON kernel for Valid and RuneCount.~~ **Done** (this revision).
2. ~~**amd64/AVX2 follow-up:** quantify the Lemire–Keiser speedup vs stdlib.~~
   **Done** (see the amd64 section) — on the GitHub Actions x86_64 runner (AVX2):
   multibyte Valid/RuneCount win **3.7–21.1×**.
3. ~~**amd64 ASCII pre-scan:** port the arm64 NEON path's high-bit ASCII pre-scan
   into the amd64 kernel so the pure-ASCII fast path stops regressing against
   `unicode/utf8.Valid`'s word-at-a-time scan.~~ **Done** — `asciiBlocksAVX2` /
   `asciiBlocksSSE` (group-of-4 `[V]PMOVMSKB` fast lane) turn the prior ~5–8×
   regression into a **2.1–3.7× win** while leaving the mixed/RuneCount paths
   unchanged.
