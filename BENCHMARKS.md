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
2. **amd64/AVX2 follow-up:** run this harness on a real x86_64 VM to quantify the
   Lemire–Keiser SIMD speedup vs stdlib there (Rosetta lacks AVX2).
