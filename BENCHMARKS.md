# Performance parity — go-simd/utf8 vs stdlib

**Methodology.** Apple M4 Max (arm64, NEON), macOS (Darwin 25.5.0), Go 1.26.4,
single core. Reference: `unicode/utf8` (Go stdlib — also the scalar fallback of
go-simd). `Valid` implements the Lemire–Keiser lookup validator; `RuneCount`
counts non-continuation bytes. Inputs: printable-ASCII fast-path and a
multibyte-mixed buffer (a/é/λ/世/🚀), seeds 2/3, sizes 64 B … 1 MiB;
`-benchtime=0.3s -count=3`, median reported. Correctness: `go test` byte-matches
`unicode/utf8.Valid` / `RuneCount` over ASCII, multibyte and every invalid class
(overlong, surrogate, too-large, lone continuation). Reproduce:

```
GOWORK=off go test -run='^$' -bench=Parity -benchmem -benchtime=0.3s -count=3 .
```

> **arm64 caveat (this host).** go-simd/utf8 ships SIMD kernels for **amd64,
> ppc64le and s390x only** — there is *no* arm64/NEON kernel yet. On this host
> `Valid`/`RuneCount` fall back to `unicode/utf8` (zero-overhead), which the
> numbers below confirm (gosimd ≈ stdlib to within noise). The real Lemire–Keiser
> SIMD speedup must be measured on **amd64/AVX2** (follow-up — needs an x86_64
> host).

## Valid — ASCII fast path

| size | go-simd (GB/s) | stdlib (GB/s) | ratio | verdict |
|------|---------------:|--------------:|------:|---------|
| 64 B   |  9.1  |  9.1  | 1.00× | arm64 fallback = stdlib |
| 1 KiB  | 62.5  | 63.1  | 0.99× | arm64 fallback = stdlib |
| 16 KiB | 91.2  | 90.6  | 1.01× | arm64 fallback = stdlib |
| 1 MiB  | 89.0  | 89.5  | 0.99× | arm64 fallback = stdlib |

## Valid — multibyte mixed

| size | go-simd (GB/s) | stdlib (GB/s) | ratio | verdict |
|------|---------------:|--------------:|------:|---------|
| 64 B   | 2.12 | 2.04 | 1.04× | arm64 fallback = stdlib |
| 1 KiB  | 1.59 | 1.57 | 1.01× | arm64 fallback = stdlib |
| 16 KiB | 1.48 | 1.46 | 1.01× | arm64 fallback = stdlib |
| 1 MiB  | 0.43 | 0.44 | 0.99× | arm64 fallback = stdlib |

## RuneCount — multibyte mixed

| size | go-simd (GB/s) | stdlib (GB/s) | ratio | verdict |
|------|---------------:|--------------:|------:|---------|
| 64 B   | 0.91 | 0.92 | 0.99× | arm64 fallback = stdlib |
| 1 KiB  | 1.00 | 1.00 | 1.00× | arm64 fallback = stdlib |
| 16 KiB | 1.00 | 1.00 | 1.00× | arm64 fallback = stdlib |
| 1 MiB  | 0.42 | 0.42 | 1.00× | arm64 fallback = stdlib |

## Summary

* On **arm64 this is a stdlib fallback**, so go-simd ≈ stdlib by construction
  (0.99–1.04× across ASCII, mixed and RuneCount). This **confirms the fallback
  is zero-overhead** — no regression — but it is **not** a SIMD result.
* The ASCII fast path is memory-bandwidth-bound (~90 GB/s); the mixed/RuneCount
  paths are branch-bound (~0.4–2 GB/s) and are exactly where the Lemire–Keiser
  SIMD validator pays off — on the arches that have it.

### Action items
1. **Add an arm64/NEON kernel** for `Valid` (Lemire–Keiser table lookup via
   `TBL`) and `RuneCount` (popcount of non-continuation mask). The mixed/multibyte
   path is the target: stdlib drops to ~0.4–1.5 GB/s where AVX2 reaches several
   GB/s. Same gap as hex.
2. **amd64/AVX2 follow-up:** run this harness on a real x86_64 VM to quantify the
   Lemire–Keiser SIMD speedup vs stdlib there (Rosetta lacks AVX2).
3. The RuneCount harness shows 1 alloc/op on *both* paths (mixed-buffer test data
   capture) — equal on both sides, so parity is unaffected; not a code-path
   allocation.
