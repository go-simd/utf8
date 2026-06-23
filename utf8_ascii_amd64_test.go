//go:build amd64

package utf8

import (
	"testing"
	stdutf8 "unicode/utf8"
)

// TestASCIIBoundaryAMD64 stresses the SSE and AVX2 ASCII pre-scan kernels'
// group-rewind logic: a buffer that is ASCII up to some offset and then carries
// a non-ASCII byte at every position (including every group/block boundary).
// asciiBlocksSSE / asciiBlocksAVX2 must report exactly the leading all-ASCII
// block count, and the SSE/AVX2 Force dispatch must stay byte-identical to the
// standard library regardless of where the first high bit lands.
func TestASCIIBoundaryAMD64(t *testing.T) {
	// Sizes up to 320 bytes exercise: <1 group, exactly N groups, group+tail,
	// and the per-block tail rescan after a fast-lane high-bit hit for both the
	// 16-byte SSE block and the 32-byte AVX2 block.
	for n := 0; n <= 320; n++ {
		// All-ASCII buffer of length n: Valid==true, RuneCount==n.
		b := make([]byte, n)
		for i := range b {
			b[i] = byte('a' + i%26) // < 0x80
		}
		if sseBlocks := n / 16; sseBlocks > 0 {
			if got := int(asciiBlocksSSE(b, sseBlocks)); got != sseBlocks {
				t.Fatalf("all-ASCII n=%d: asciiBlocksSSE=%d want %d", n, got, sseBlocks)
			}
		}
		if avxBlocks := n / 32; avxBlocks > 0 {
			if got := int(asciiBlocksAVX2(b, avxBlocks)); got != avxBlocks {
				t.Fatalf("all-ASCII n=%d: asciiBlocksAVX2=%d want %d", n, got, avxBlocks)
			}
		}
		for _, avx2 := range []bool{false, true} {
			if !validForce(b, avx2) {
				t.Fatalf("all-ASCII n=%d avx2=%v: validForce=false", n, avx2)
			}
			if got := runeCountForce(b, avx2); got != n {
				t.Fatalf("all-ASCII n=%d avx2=%v: runeCountForce=%d want %d", n, avx2, got, n)
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
			for _, avx2 := range []bool{false, true} {
				if got, want := validForce(c, avx2), stdutf8.Valid(c); got != want {
					t.Fatalf("n=%d off=%d avx2=%v: Valid=%v want %v", n, off, avx2, got, want)
				}
				if got, want := runeCountForce(c, avx2), stdutf8.RuneCount(c); got != want {
					t.Fatalf("n=%d off=%d avx2=%v: RuneCount=%d want %d", n, off, avx2, got, want)
				}
			}
			// asciiBlocksSSE must stop at the 16-byte block containing the byte.
			if sseBlocks := n / 16; sseBlocks > 0 {
				wantAB := off / 16
				if wantAB > sseBlocks {
					wantAB = sseBlocks
				}
				if got := int(asciiBlocksSSE(c, sseBlocks)); got != wantAB {
					t.Fatalf("n=%d off=%d: asciiBlocksSSE=%d want %d", n, off, got, wantAB)
				}
			}
			// asciiBlocksAVX2 must stop at the 32-byte block containing the byte.
			if avxBlocks := n / 32; avxBlocks > 0 {
				wantAB := off / 32
				if wantAB > avxBlocks {
					wantAB = avxBlocks
				}
				if got := int(asciiBlocksAVX2(c, avxBlocks)); got != wantAB {
					t.Fatalf("n=%d off=%d: asciiBlocksAVX2=%d want %d", n, off, got, wantAB)
				}
			}
		}
	}
}
