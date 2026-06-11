package utf8

import (
	"bytes"
	"math/rand"
	"strings"
	"testing"
	stdutf8 "unicode/utf8"
)

// cases exercises ASCII, multibyte, boundary/incomplete and every invalid class
// (overlong, surrogate, too-large, lone continuation). Each is checked against
// unicode/utf8.Valid for byte-identical behaviour.
var cases = []struct {
	name string
	in   []byte
}{
	{"empty", nil},
	{"ascii", []byte("hello, world")},
	{"ascii-16", []byte("0123456789abcdef")},
	{"ascii-31", []byte("0123456789abcdef0123456789abcde")},
	{"ascii-32", []byte("0123456789abcdef0123456789abcdef")},
	{"ascii-33", []byte("0123456789abcdef0123456789abcdef!")},
	{"2byte", []byte("café résumé naïve")},
	{"3byte", []byte("日本語のテキスト ☃ €")},
	{"4byte", []byte("emoji 😀😁😂🎉 𝕳𝖊𝖑𝖑𝖔")},
	{"mixed-long", []byte(strings.Repeat("aé日😀", 40))},
	{"max-bmp", []byte{0xEF, 0xBF, 0xBF}},         // U+FFFF
	{"max-rune", []byte{0xF4, 0x8F, 0xBF, 0xBF}},  // U+10FFFF
	{"min-2byte", []byte{0xC2, 0x80}},             // U+0080
	{"min-3byte", []byte{0xE0, 0xA0, 0x80}},       // U+0800
	{"min-4byte", []byte{0xF0, 0x90, 0x80, 0x80}}, // U+10000

	// invalid: lone continuation
	{"lone-cont", []byte{0x80}},
	{"lone-cont-mid", []byte("abc\x80def")},
	{"two-cont", []byte{0x80, 0x80}},

	// invalid: incomplete / truncated sequences
	{"trunc-2byte", []byte{0xC2}},
	{"trunc-3byte", []byte{0xE0, 0xA0}},
	{"trunc-4byte", []byte{0xF0, 0x90, 0x80}},
	{"trunc-at-16", []byte("0123456789abcde\xC2")},
	{"trunc-at-32", []byte("0123456789abcdef0123456789abcde\xE0")},

	// invalid: overlong encodings
	{"overlong-2", []byte{0xC0, 0x80}}, // overlong NUL
	{"overlong-2b", []byte{0xC1, 0xBF}},
	{"overlong-3", []byte{0xE0, 0x80, 0x80}}, // < U+0800
	{"overlong-3b", []byte{0xE0, 0x9F, 0xBF}},
	{"overlong-4", []byte{0xF0, 0x80, 0x80, 0x80}}, // < U+10000
	{"overlong-4b", []byte{0xF0, 0x8F, 0xBF, 0xBF}},

	// invalid: surrogates U+D800..U+DFFF
	{"surrogate-low", []byte{0xED, 0xA0, 0x80}},  // U+D800
	{"surrogate-high", []byte{0xED, 0xBF, 0xBF}}, // U+DFFF
	{"surrogate-mid", []byte{0xED, 0xAF, 0xBF}},

	// invalid: too large (> U+10FFFF)
	{"too-large-f4", []byte{0xF4, 0x90, 0x80, 0x80}}, // U+110000
	{"too-large-f5", []byte{0xF5, 0x80, 0x80, 0x80}},
	{"too-large-ff", []byte{0xFF}},
	{"too-large-fe", []byte{0xFE, 0x80, 0x80, 0x80}},

	// invalid: continuation where a leading byte is required
	{"e0-bad-2nd", []byte{0xE0, 0xC0, 0x80}},
	{"missing-cont", []byte{0xE1, 0x80, 0x41}}, // ASCII where cont expected

	// boundary: valid sequence straddling block boundary
	{"straddle-16", append([]byte("0123456789abcde"), []byte("日")...)},
	{"straddle-32", append([]byte("0123456789abcdef0123456789abcde"), []byte("😀")...)},
}

func TestValid(t *testing.T) {
	for _, c := range cases {
		got := Valid(c.in)
		want := stdutf8.Valid(c.in)
		if got != want {
			t.Errorf("%s: Valid(%q)=%v want %v", c.name, c.in, got, want)
		}
		if gs, ws := ValidString(string(c.in)), stdutf8.ValidString(string(c.in)); gs != ws {
			t.Errorf("%s: ValidString(%q)=%v want %v", c.name, c.in, gs, ws)
		}
	}
}

// TestValidRandomLengths checks every length 0..300 with random bytes (mostly
// invalid) and random valid UTF-8, so block-boundary handling at all alignments
// is exercised.
func TestValidRandomLengths(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for n := 0; n <= 300; n++ {
		b := make([]byte, n)
		// random bytes
		rng.Read(b)
		if got, want := Valid(b), stdutf8.Valid(b); got != want {
			t.Fatalf("random n=%d: got=%v want=%v bytes=%x", n, got, want, b)
		}
		// valid UTF-8 of about this length
		var sb bytes.Buffer
		for sb.Len() < n {
			sb.WriteRune(rune(rng.Intn(0x10FFFF + 1)))
		}
		vb := sb.Bytes()
		if got, want := Valid(vb), stdutf8.Valid(vb); got != want {
			t.Fatalf("valid n=%d: got=%v want=%v", n, got, want)
		}
	}
}

func TestRuneCount(t *testing.T) {
	for _, c := range cases {
		if got, want := RuneCount(c.in), stdutf8.RuneCount(c.in); got != want {
			t.Errorf("%s: RuneCount(%q)=%d want %d", c.name, c.in, got, want)
		}
		s := string(c.in)
		if got, want := RuneCountInString(s), stdutf8.RuneCountInString(s); got != want {
			t.Errorf("%s: RuneCountInString(%q)=%d want %d", c.name, s, got, want)
		}
	}
}

// TestRuneCountRandomLengths checks every length 0..300 with random bytes
// (mostly invalid, so the scalar fallback is exercised) and random valid UTF-8
// (so the SIMD fast path and its rune-boundary tail split are exercised at all
// alignments), each against unicode/utf8.RuneCount.
func TestRuneCountRandomLengths(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for n := 0; n <= 300; n++ {
		b := make([]byte, n)
		rng.Read(b)
		if got, want := RuneCount(b), stdutf8.RuneCount(b); got != want {
			t.Fatalf("random n=%d: got=%d want=%d bytes=%x", n, got, want, b)
		}
		var sb bytes.Buffer
		for sb.Len() < n {
			sb.WriteRune(rune(rng.Intn(0x10FFFF + 1)))
		}
		vb := sb.Bytes()
		if got, want := RuneCount(vb), stdutf8.RuneCount(vb); got != want {
			t.Fatalf("valid n=%d: got=%d want=%d", n, got, want)
		}
	}
}

func FuzzRuneCount(f *testing.F) {
	for _, c := range cases {
		f.Add(c.in)
	}
	f.Add([]byte("the quick brown fox jumps over the lazy dog"))
	f.Fuzz(func(t *testing.T, p []byte) {
		if got, want := RuneCount(p), stdutf8.RuneCount(p); got != want {
			t.Fatalf("RuneCount(%q)=%d want %d", p, got, want)
		}
	})
}

func FuzzRuneCountInString(f *testing.F) {
	for _, c := range cases {
		f.Add(string(c.in))
	}
	f.Add("the quick brown fox jumps over the lazy dog")
	f.Fuzz(func(t *testing.T, s string) {
		if got, want := RuneCountInString(s), stdutf8.RuneCountInString(s); got != want {
			t.Fatalf("RuneCountInString(%q)=%d want %d", s, got, want)
		}
	})
}

// benchData returns ~1 MiB of valid mixed UTF-8 (ASCII-heavy, like real text,
// with a steady sprinkling of 2/3/4-byte runes) for throughput benchmarks.
func benchData() []byte {
	var b bytes.Buffer
	rng := rand.New(rand.NewSource(2))
	for b.Len() < 1<<20 {
		switch rng.Intn(10) {
		case 0:
			b.WriteRune(rune(0x80 + rng.Intn(0x700))) // 2-byte
		case 1:
			b.WriteRune(rune(0x800 + rng.Intn(0x8000))) // 3-byte
		case 2:
			b.WriteRune(rune(0x10000 + rng.Intn(0x1000))) // 4-byte
		default:
			b.WriteByte(byte(rng.Intn(0x80))) // ASCII
		}
	}
	return b.Bytes()
}

func BenchmarkValid(b *testing.B) {
	p := benchData()
	b.SetBytes(int64(len(p)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Valid(p)
	}
}

func BenchmarkValidStdlib(b *testing.B) {
	p := benchData()
	b.SetBytes(int64(len(p)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stdutf8.Valid(p)
	}
}

func BenchmarkRuneCount(b *testing.B) {
	p := benchData()
	b.SetBytes(int64(len(p)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		RuneCount(p)
	}
}

func BenchmarkRuneCountStdlib(b *testing.B) {
	p := benchData()
	b.SetBytes(int64(len(p)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stdutf8.RuneCount(p)
	}
}

func FuzzValid(f *testing.F) {
	for _, c := range cases {
		f.Add(c.in)
	}
	f.Add([]byte("the quick brown fox jumps over the lazy dog"))
	f.Fuzz(func(t *testing.T, p []byte) {
		if got, want := Valid(p), stdutf8.Valid(p); got != want {
			t.Fatalf("Valid(%q)=%v want %v", p, got, want)
		}
	})
}
