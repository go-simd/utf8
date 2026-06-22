package utf8

// Standardized performance-parity harness: go-simd/utf8 (SIMD dispatch) vs
// unicode/utf8 (stdlib). Valid uses the Lemire-Keiser one-instruction-per-byte
// validator; RuneCount counts non-continuation bytes over a validated prefix.
//
// NOTE: go-simd/utf8 ships SIMD kernels for amd64 / ppc64le / s390x only. On
// arm64 (this host) there is no NEON kernel yet, so the dispatch falls back to
// unicode/utf8 and go-simd == stdlib by construction. The arm64 numbers here
// therefore confirm the zero-overhead fallback, not a SIMD speedup; the real
// SIMD parity (Lemire-Keiser AVX2) must be measured on amd64 (follow-up, needs
// an x86 host).
//
//	GOWORK=off go test -run=^$ -bench='Parity' -benchmem .

import (
	"bytes"
	"math/rand"
	"testing"
	stdutf8 "unicode/utf8"
)

var paritySizes = []int{64, 1024, 16384, 1 << 20}

// parityASCII is the all-valid fast-path input the SIMD validator is built for.
func parityASCII(n int) []byte {
	b := make([]byte, n)
	r := rand.New(rand.NewSource(2))
	for i := range b {
		b[i] = byte(32 + r.Intn(95)) // printable ASCII
	}
	return b
}

// parityMixed interleaves multibyte runes to exercise the structural validator.
func parityMixed(n int) []byte {
	var buf bytes.Buffer
	r := rand.New(rand.NewSource(3))
	runes := []rune{'a', 'é', 'λ', '世', '🚀'}
	for buf.Len() < n {
		buf.WriteRune(runes[r.Intn(len(runes))])
	}
	return buf.Bytes()[:n] // may split a trailing rune; both impls treat it identically
}

func sizeLabel(n int) string {
	switch n {
	case 64:
		return "64B"
	case 1024:
		return "1KiB"
	case 16384:
		return "16KiB"
	case 1 << 20:
		return "1MiB"
	}
	return "?"
}

func BenchmarkParityValidASCII(b *testing.B) {
	for _, n := range paritySizes {
		p := parityASCII(n)
		b.Run(sizeLabel(n)+"/gosimd", func(b *testing.B) {
			b.SetBytes(int64(n))
			for i := 0; i < b.N; i++ {
				Valid(p)
			}
		})
		b.Run(sizeLabel(n)+"/stdlib", func(b *testing.B) {
			b.SetBytes(int64(n))
			for i := 0; i < b.N; i++ {
				stdutf8.Valid(p)
			}
		})
	}
}

func BenchmarkParityValidMixed(b *testing.B) {
	for _, n := range paritySizes {
		p := parityMixed(n)
		b.Run(sizeLabel(n)+"/gosimd", func(b *testing.B) {
			b.SetBytes(int64(n))
			for i := 0; i < b.N; i++ {
				Valid(p)
			}
		})
		b.Run(sizeLabel(n)+"/stdlib", func(b *testing.B) {
			b.SetBytes(int64(n))
			for i := 0; i < b.N; i++ {
				stdutf8.Valid(p)
			}
		})
	}
}

func BenchmarkParityRuneCount(b *testing.B) {
	for _, n := range paritySizes {
		p := parityMixed(n)
		b.Run(sizeLabel(n)+"/gosimd", func(b *testing.B) {
			b.SetBytes(int64(n))
			for i := 0; i < b.N; i++ {
				RuneCount(p)
			}
		})
		b.Run(sizeLabel(n)+"/stdlib", func(b *testing.B) {
			b.SetBytes(int64(n))
			for i := 0; i < b.N; i++ {
				stdutf8.RuneCount(p)
			}
		})
	}
}
