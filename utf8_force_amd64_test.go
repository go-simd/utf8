//go:build amd64

package utf8

import (
	"bytes"
	"math/rand"
	"testing"
	stdutf8 "unicode/utf8"

	scutf8 "github.com/stuartcarnie/go-simd/unicode/utf8"
)

// BenchmarkValidStuartcarnie benchmarks github.com/stuartcarnie/go-simd's
// unicode/utf8.Valid — a 2018 Go port of Lemire's reference SSE4/AVX2 validator
// (pure-Go-callable assembly, no cgo; Apache-2.0). It is the only pre-existing
// pure-Go SIMD UTF-8 validator found, so it is the competitor of record.
func BenchmarkValidStuartcarnie(b *testing.B) {
	p := benchData()
	b.SetBytes(int64(len(p)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		scutf8.Valid(p)
	}
}

// TestStuartcarnieMatches asserts the competitor agrees with the standard
// library on the table cases, so the benchmark compares like for like.
func TestStuartcarnieMatches(t *testing.T) {
	for _, c := range cases {
		if got, want := scutf8.Valid(c.in), stdutf8.Valid(c.in); got != want {
			t.Errorf("%s: stuartcarnie=%v stdlib=%v", c.name, got, want)
		}
	}
}

// validForce mirrors valid but lets the test pick the kernel, so the AVX2 path
// can be exercised even when the (Rosetta) CPU reports no AVX2.
func validForce(p []byte, avx2 bool) bool {
	n := len(p)
	if avx2 && n >= 32 {
		blocks := n / 32
		if int(asciiBlocksAVX2(p, blocks)) == blocks {
			return stdutf8.Valid(p[blocks*32:])
		}
		if validBlocksAVX2(p, blocks) == 0 {
			return false
		}
		return stdutf8.Valid(p[runeStart(p, blocks*32):])
	}
	if n >= 16 {
		blocks := n / 16
		if int(asciiBlocksSSE(p, blocks)) == blocks {
			return stdutf8.Valid(p[blocks*16:])
		}
		if validBlocksSSE(p, blocks) == 0 {
			return false
		}
		return stdutf8.Valid(p[runeStart(p, blocks*16):])
	}
	return stdutf8.Valid(p)
}

func testForce(t *testing.T, avx2 bool) {
	t.Helper()
	for _, c := range cases {
		if got, want := validForce(c.in, avx2), stdutf8.Valid(c.in); got != want {
			t.Errorf("%s: avx2=%v got=%v want=%v in=%q", c.name, avx2, got, want, c.in)
		}
	}
	rng := rand.New(rand.NewSource(7))
	for n := 0; n <= 300; n++ {
		b := make([]byte, n)
		rng.Read(b)
		if got, want := validForce(b, avx2), stdutf8.Valid(b); got != want {
			t.Fatalf("avx2=%v random n=%d got=%v want=%v %x", avx2, n, got, want, b)
		}
		var sb bytes.Buffer
		for sb.Len() < n {
			sb.WriteRune(rune(rng.Intn(0x10FFFF + 1)))
		}
		vb := sb.Bytes()
		if got, want := validForce(vb, avx2), stdutf8.Valid(vb); got != want {
			t.Fatalf("avx2=%v valid n=%d got=%v want=%v", avx2, n, got, want)
		}
	}
}

func TestValidForceSSE(t *testing.T)  { testForce(t, false) }
func TestValidForceAVX2(t *testing.T) { testForce(t, true) }

// runeCountForce mirrors runeCount but lets the test pick the kernel, so the
// AVX2 count path can be exercised even when the (Rosetta / older-VM) CPU
// reports no AVX2 and the live dispatch never reaches it.
func runeCountForce(p []byte, avx2 bool) int {
	n := len(p)
	if avx2 && n >= 32 {
		blocks := n / 32
		if int(asciiBlocksAVX2(p, blocks)) == blocks {
			return blocks*32 + stdutf8.RuneCount(p[blocks*32:])
		}
		if validBlocksAVX2(p, blocks) != 0 {
			return countValidPrefix(p, blocks*32, countContAVX2(p, blocks))
		}
		return stdutf8.RuneCount(p)
	}
	if n >= 16 {
		blocks := n / 16
		if int(asciiBlocksSSE(p, blocks)) == blocks {
			return blocks*16 + stdutf8.RuneCount(p[blocks*16:])
		}
		if validBlocksSSE(p, blocks) != 0 {
			return countValidPrefix(p, blocks*16, countContSSE(p, blocks))
		}
		return stdutf8.RuneCount(p)
	}
	return stdutf8.RuneCount(p)
}

func runeCountForceTest(t *testing.T, avx2 bool) {
	t.Helper()
	for _, c := range cases {
		if got, want := runeCountForce(c.in, avx2), stdutf8.RuneCount(c.in); got != want {
			t.Errorf("%s: avx2=%v got=%d want=%d in=%q", c.name, avx2, got, want, c.in)
		}
	}
	rng := rand.New(rand.NewSource(11))
	for n := 0; n <= 300; n++ {
		b := make([]byte, n)
		rng.Read(b)
		if got, want := runeCountForce(b, avx2), stdutf8.RuneCount(b); got != want {
			t.Fatalf("avx2=%v random n=%d got=%d want=%d %x", avx2, n, got, want, b)
		}
		var sb bytes.Buffer
		for sb.Len() < n {
			sb.WriteRune(rune(rng.Intn(0x10FFFF + 1)))
		}
		vb := sb.Bytes()
		if got, want := runeCountForce(vb, avx2), stdutf8.RuneCount(vb); got != want {
			t.Fatalf("avx2=%v valid n=%d got=%d want=%d", avx2, n, got, want)
		}
	}
}

func TestRuneCountForceSSE(t *testing.T)  { runeCountForceTest(t, false) }
func TestRuneCountForceAVX2(t *testing.T) { runeCountForceTest(t, true) }

// FuzzRuneCountForceAVX2 exercises the AVX2 count kernel directly against the
// standard library, since the test CPU may report no AVX2 and never reach it
// through the live dispatch.
func FuzzRuneCountForceAVX2(f *testing.F) {
	for _, c := range cases {
		f.Add(c.in)
	}
	f.Fuzz(func(t *testing.T, p []byte) {
		if got, want := runeCountForce(p, true), stdutf8.RuneCount(p); got != want {
			t.Fatalf("avx2 RuneCount(%q)=%d want %d", p, got, want)
		}
	})
}

// FuzzValidForceAVX2 exercises the AVX2 kernel directly, since the (Rosetta)
// test CPU may report no AVX2 and never reach it through the live dispatch.
func FuzzValidForceAVX2(f *testing.F) {
	for _, c := range cases {
		f.Add(c.in)
	}
	f.Fuzz(func(t *testing.T, p []byte) {
		if got, want := validForce(p, true), stdutf8.Valid(p); got != want {
			t.Fatalf("avx2 Valid(%q)=%v want %v", p, got, want)
		}
	})
}
