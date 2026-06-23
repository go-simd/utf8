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

**Methodology.** GitHub Actions `ubuntu-latest` runner, **AMD EPYC 7763** (`avx2`
present, **no `avx512*`** — confirmed from `/proc/cpuinfo`), `GOAMD64` baseline,
Go stable, single core. Same parity harness, `-count=6`, **min-of-6**. The runner
is shared, so absolute throughput is noisy and **not comparable to the arm64 M4
Max rows above** (different hardware/ISA); the **ratios vs stdlib** are measured
back-to-back on the *same* CPU and are valid. Reproduce via
`gh workflow run bench-amd64.yml`.

### Valid — ASCII fast path (amd64 AVX2)

| size | go-simd (MB/s) | stdlib | ×stdlib | verdict |
|------|---------------:|-------:|--------:|---------|
| 64 B   | 4571 |  4541 | 1.01× | parity |
| 1 KiB  | 5650 | 31856 | 0.18× | **regresses vs stdlib** |
| 16 KiB | 5667 | 46841 | 0.12× | **regresses vs stdlib** |
| 1 MiB  | 5659 | 41473 | 0.14× | **regresses vs stdlib** |

> **Honest finding (amd64).** On pure ASCII, **go-simd *loses* to stdlib by
> ~5–8×** at ≥1 KiB. The Go `unicode/utf8.Valid` ASCII fast path on amd64 is a
> word-at-a-time high-bit scan the compiler turns into ~30–47 GB/s; the amd64
> Lemire–Keiser kernel here runs the full validator on every block and has **no
> cheap ASCII pre-scan** (unlike the arm64 NEON path, which added a 64-byte
> high-bit pre-scan). This is the one go-simd workload that regresses on amd64 —
> the fix is to port the same ASCII pre-scan into the amd64 kernel (action 3).

### Valid — multibyte mixed (amd64 AVX2)

| size | go-simd (MB/s) | stdlib | ×stdlib | verdict |
|------|---------------:|-------:|--------:|---------|
| 64 B   | 4534 | 1199 |  3.78× | wins |
| 1 KiB  | 5638 | 1156 |  4.88× | wins |
| 16 KiB | 5672 | 1009 |  5.62× | wins |
| 1 MiB  | 5654 |  300 | 18.83× | **wins big** |

### RuneCount — multibyte mixed (amd64 AVX2)

| size | go-simd (MB/s) | stdlib | ×stdlib | verdict |
|------|---------------:|-------:|--------:|---------|
| 64 B   | 2987 | 615 |  4.86× | wins |
| 1 KiB  | 4787 | 706 |  6.78× | wins |
| 16 KiB | 4993 | 668 |  7.47× | wins |
| 1 MiB  | 4991 | 253 | 19.76× | **wins big** |

* On amd64 the **multibyte/RuneCount paths win 3.8–19.8×** vs stdlib (same shape
  as arm64). The **ASCII fast path regresses ~5–8×** because the amd64 kernel
  lacks the ASCII pre-scan the arm64 path has — see the note above (action 3).

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
   **Done** (see the amd64 section) — on the GitHub Actions x86_64 runner (EPYC
   7763, AVX2): multibyte Valid/RuneCount win **3.8–19.8×**, but the **pure-ASCII
   path regresses ~5–8×** vs stdlib's word-at-a-time fast path.
3. **amd64 ASCII pre-scan:** port the arm64 NEON path's 64-byte high-bit ASCII
   pre-scan into the amd64 kernel so the pure-ASCII fast path stops regressing
   against `unicode/utf8.Valid`'s word-at-a-time scan.
