//go:build ppc64le

package utf8

import (
	"testing"
	stdutf8 "unicode/utf8"

	"golang.org/x/sys/cpu"
)

// TestDispatchPPC64LE drives valid and runeCount down both ppc64le branches —
// the standard-library fallback and the VSX Lemire–Keiser kernel — by toggling
// hasVSX, restoring it with defer, and comparing against unicode/utf8. The
// kernels load blocks with LXVB16X, an ISA-3.0 (POWER9) instruction that raises
// SIGILL on POWER8, so the kernel-forcing branch runs only when the host is
// actually POWER9+ (mirroring the amd64 force tests). The standard-library
// fallback branch is always exercised. The power9-targeted QEMU CI job and the
// native POWER9/POWER10 farm runs cover the kernel branch.
func TestDispatchPPC64LE(t *testing.T) {
	saved := hasVSX
	defer func() { hasVSX = saved }()

	// big mixes ASCII and 2/3/4-byte runes so the block-aligned kernel prefix and
	// the straddling-rune/tail scalar path are both exercised at length.
	big := benchData()
	check := func(label string) {
		for _, c := range cases {
			if got, want := Valid(c.in), stdutf8.Valid(c.in); got != want {
				t.Fatalf("%s Valid(%s): got %v want %v", label, c.name, got, want)
			}
			if got, want := RuneCount(c.in), stdutf8.RuneCount(c.in); got != want {
				t.Fatalf("%s RuneCount(%s): got %d want %d", label, c.name, got, want)
			}
		}
		// Many lengths around the 16-byte block boundary plus a long buffer.
		for _, n := range []int{0, 1, 15, 16, 17, 31, 32, 64, 1000, len(big)} {
			p := big[:n]
			if got, want := Valid(p), stdutf8.Valid(p); got != want {
				t.Fatalf("%s Valid n=%d: got %v want %v", label, n, got, want)
			}
			if got, want := RuneCount(p), stdutf8.RuneCount(p); got != want {
				t.Fatalf("%s RuneCount n=%d: got %d want %d", label, n, got, want)
			}
		}
	}

	// Standard-library fallback: always safe, exercised on every ppc64le host.
	hasVSX = false
	check("fallback")

	// VSX kernel: only force it on when the CPU is POWER9+, otherwise the LXVB16X
	// in the kernels would SIGILL (e.g. on a POWER8 farm node).
	if !cpu.PPC64.IsPOWER9 {
		t.Log("CPU is pre-POWER9; VSX kernel branch not exercised on this host")
		return
	}
	hasVSX = true
	check("vsx")
}
