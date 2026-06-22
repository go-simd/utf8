//go:build arm64

package utf8

import (
	"testing"
	stdutf8 "unicode/utf8"
)

// TestASCIIBoundaryARM64 stresses the NEON ASCII pre-scan's group-rewind logic:
// a buffer that is ASCII up to some offset and then carries a non-ASCII byte at
// every position (including every 64-byte group boundary). asciiBlocks must
// report exactly the leading all-ASCII block count, and Valid / RuneCount must
// stay byte-identical to the standard library regardless of where the first
// high bit lands.
func TestASCIIBoundaryARM64(t *testing.T) {
	for n := 0; n <= 320; n++ {
		// All-ASCII buffer of length n: Valid==true, RuneCount==n.
		b := make([]byte, n)
		for i := range b {
			b[i] = byte('a' + i%26) // < 0x80
		}
		if !Valid(b) {
			t.Fatalf("all-ASCII n=%d: Valid=false", n)
		}
		if got := RuneCount(b); got != n {
			t.Fatalf("all-ASCII n=%d: RuneCount=%d want %d", n, got, n)
		}
		if blocks := n / 16; blocks > 0 {
			if got := int(asciiBlocks(b, blocks)); got != blocks {
				t.Fatalf("all-ASCII n=%d: asciiBlocks=%d want %d", n, got, blocks)
			}
		}
		// Inject a valid 2-byte rune (0xC3 0xA9 = é) at every even offset so the
		// first high bit walks across group/block boundaries; the rest stays
		// ASCII. Compare against the stdlib for both Valid and RuneCount.
		for off := 0; off+2 <= n; off += 2 {
			c := make([]byte, n)
			for i := range c {
				c[i] = byte('a' + i%26)
			}
			c[off] = 0xC3
			c[off+1] = 0xA9
			if got, want := Valid(c), stdutf8.Valid(c); got != want {
				t.Fatalf("n=%d off=%d: Valid=%v want %v", n, off, got, want)
			}
			if got, want := RuneCount(c), stdutf8.RuneCount(c); got != want {
				t.Fatalf("n=%d off=%d: RuneCount=%d want %d", n, off, got, want)
			}
			// asciiBlocks must stop at the block containing the injected byte.
			if blocks := n / 16; blocks > 0 {
				wantAB := off / 16
				if wantAB > blocks {
					wantAB = blocks
				}
				if got := int(asciiBlocks(c, blocks)); got != wantAB {
					t.Fatalf("n=%d off=%d: asciiBlocks=%d want %d", n, off, got, wantAB)
				}
			}
		}
	}
}
