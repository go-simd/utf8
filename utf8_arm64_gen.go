//go:build ignore

// Command gen produces utf8_arm64.s with go-asmgen: the Lemire–Keiser
// "Validating UTF-8 In Less Than One Instruction Per Byte" lookup-based SIMD
// validator ported to arm64 NEON (16 bytes per 128-bit block), plus a NEON
// rune-count (continuation-byte) kernel.
//
// The algorithm is the same one the amd64 SSSE3 path implements (see
// utf8_gen.go): classify each byte by high nibble via a VTBL nibble lookup,
// then across (prev,curr) blocks check continuation-length carry, over/under
// long continuations, the ED/F4 first-continuation maxima, overlong minima and
// "byte > 0xF4"; every check ORs into a running error accumulator, valid iff
// that accumulator is all-zero.
//
// The released arm64 assembler lacks several of the x86 building blocks, so we
// synthesise them from the available NEON ops:
//   - PSUBUSB (unsigned saturating subtract): subs_epu8(a,b) = umax(a,b) - b,
//     via VUMAX then VSUB.
//   - PCMPGTB (signed compare-greater): NEON-Go has only VCMEQ, so we flip the
//     sign bit of both operands with VEOR 0x80 and do an unsigned compare:
//     signed_gt(a,b) = NOT( umin(a^80,b^80) == (a^80) ), the inner equality
//     being a<=b and the outer NOT (VEOR all-ones) the strict greater-than.
//   - PSHUFB (nibble table lookup) -> VTBL; PALIGNR $k(prev,cur) -> VEXT $k with
//     operand order (cur, prev) so each lane sees the byte k positions earlier.
//   - PTEST -> reduce the error vector to a GPR (VMOV D-lanes + ORR).
//
// As in the amd64 kernel there is no final incomplete-sequence check: the Go
// caller backs the split point to a rune boundary and re-validates the
// straddling rune (and the tail) with the standard library, so the kernel only
// reports structural errors within its processed range. Run: go run
// utf8_arm64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/arm64"
	"github.com/go-asmgen/asmgen/emit"
)

// continuationLengths LUT, indexed by high nibble: 0xxx=>1 (ASCII),
// 10xx=>0 (continuation), 110x=>2, 1110=>3, 1111=>4.
var contLenLUT = []byte{1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4}

// checkOverlong initial_mins LUT, indexed by off1 high nibble.
var initialMinsLUT = []byte{
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0xC2, 0x80, 0xE1, 0xF1,
}

// checkOverlong second_mins LUT, indexed by off1 high nibble.
var secondMinsLUT = []byte{
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x7F, 0x7F, 0xA0, 0x90,
}

func sig() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Slice("src"), abi.Scalar("n", abi.Int64)},
		[]abi.Arg{abi.Scalar("ret", abi.Uint8)},
	)
}

func countSig() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Slice("src"), abi.Scalar("n", abi.Int64)},
		[]abi.Arg{abi.Scalar("ret", abi.Int64)},
	)
}

func main() {
	f := emit.NewFile("arm64")

	contLen := f.Data("contLen", contLenLUT)
	initMins := f.Data("initMins", initialMinsLUT)
	secMins := f.Data("secMins", secondMinsLUT)

	// ====================================================================
	// validBlocks: 16 bytes per block.
	// Persistent state: V15=has_error, V13=prev rawbytes,
	//   V14=prev high_nibbles, V12=prev carried_continuations.
	// Constants (loaded once): V20=contLen, V21=initMins, V22=secMins,
	//   V23=0x0F, V24=0xF4, V25=0xED, V26=0x9F, V27=0x8F, V28=allOnes,
	//   V29=0x80, V30=0x01, V31=0x02.
	// Scratch: V0..V11.
	// ====================================================================
	b := arm64.NewFunc("validBlocks", sig(), 0)
	b.LoadArg("src_base", "R1")
	b.LoadArg("n", "R2")
	b.Raw("MOVD $%s(SB), R3", contLen).Raw("VLD1 (R3), [V20.B16]")
	b.Raw("MOVD $%s(SB), R3", initMins).Raw("VLD1 (R3), [V21.B16]")
	b.Raw("MOVD $%s(SB), R3", secMins).Raw("VLD1 (R3), [V22.B16]")
	b.Raw("VMOVI $15, V23.B16")
	b.Raw("MOVD $0xF4, R3").Raw("VDUP R3, V24.B16")
	b.Raw("MOVD $0xED, R3").Raw("VDUP R3, V25.B16")
	b.Raw("MOVD $0x9F, R3").Raw("VDUP R3, V26.B16")
	b.Raw("MOVD $0x8F, R3").Raw("VDUP R3, V27.B16")
	b.Raw("VMOVI $255, V28.B16")
	b.Raw("MOVD $0x80, R3").Raw("VDUP R3, V29.B16")
	b.Raw("VMOVI $1, V30.B16")
	b.Raw("VMOVI $2, V31.B16")
	b.Raw("VMOVI $0, V15.B16") // has_error = 0
	b.Raw("VMOVI $0, V13.B16") // prev rawbytes = 0
	b.Raw("VMOVI $0, V14.B16") // prev high_nibbles = 0
	b.Raw("VMOVI $0, V12.B16") // prev carried_continuations = 0
	b.Raw("CBZ R2, vdone")
	b.Label("vloop")
	// --- load current block (rawbytes -> V0) ---
	b.Raw("VLD1 (R1), [V0.B16]")
	// --- high_nibbles = bytes >> 4 (V1) ---
	b.Raw("VUSHR $4, V0.B16, V1.B16")
	// --- checkSmallerThan0xF4: has_error |= subs_epu8(bytes, 0xF4) ---
	b.Raw("VUMAX V24.B16, V0.B16, V3.B16").Raw("VSUB V24.B16, V3.B16, V3.B16")
	b.Raw("VORR V3.B16, V15.B16, V15.B16")
	// --- initial_lengths = VTBL(contLen, high_nibbles) (V2) ---
	b.Raw("VTBL V1.B16, [V20.B16], V2.B16")
	// --- carryContinuations ---
	// right1 = subs_epu8( ext15(prev_carries, init_len), 1 )
	b.Raw("VEXT $15, V2.B16, V12.B16, V3.B16")                                 // [prev_carries[15], init_len[0..14]]
	b.Raw("VUMAX V30.B16, V3.B16, V3.B16").Raw("VSUB V30.B16, V3.B16, V3.B16") // -1 sat
	b.Raw("VADD V2.B16, V3.B16, V3.B16")                                       // V3 = sum = init_len + right1
	// right2 = subs_epu8( ext14(prev_carries, sum), 2 )
	b.Raw("VEXT $14, V3.B16, V12.B16, V5.B16")                                 // [prev_carries[14..15], sum[0..13]]
	b.Raw("VUMAX V31.B16, V5.B16, V5.B16").Raw("VSUB V31.B16, V5.B16, V5.B16") // -2 sat
	b.Raw("VADD V5.B16, V3.B16, V3.B16")                                       // V3 = carried_continuations
	// --- checkContinuations ---
	// overunder = cmpeq( sgt(carries, init_len), sgt(init_len, 0) )
	emitSGT(b, "V3.B16", "V2.B16", "V5.B16") // V5 = carries > init_len
	b.Raw("VMOVI $0, V7.B16")
	emitSGT(b, "V2.B16", "V7.B16", "V6.B16") // V6 = init_len > 0
	b.Raw("VCMEQ V6.B16, V5.B16, V5.B16")    // overunder
	b.Raw("VORR V5.B16, V15.B16, V15.B16")
	// --- off1_current_bytes = ext15(prev_rawbytes, rawbytes) (V8) ---
	b.Raw("VEXT $15, V0.B16, V13.B16, V8.B16")
	// --- checkFirstContinuationMax ---
	// badED = (cur > 0x9F) & (off1 == 0xED)
	b.Raw("VCMEQ V25.B16, V8.B16, V9.B16")     // maskED
	emitSGT(b, "V0.B16", "V26.B16", "V10.B16") // cur > 0x9F
	b.Raw("VAND V10.B16, V9.B16, V9.B16")
	b.Raw("VORR V9.B16, V15.B16, V15.B16")
	// badF4 = (cur > 0x8F) & (off1 == 0xF4)
	b.Raw("VCMEQ V24.B16, V8.B16, V9.B16")     // maskF4
	emitSGT(b, "V0.B16", "V27.B16", "V10.B16") // cur > 0x8F
	b.Raw("VAND V10.B16, V9.B16, V9.B16")
	b.Raw("VORR V9.B16, V15.B16, V15.B16")
	// --- checkOverlong ---
	// off1_hibits = ext15(prev_high_nibbles, high_nibbles) (V9)
	b.Raw("VEXT $15, V1.B16, V14.B16, V9.B16")
	// initial_under = sgt( VTBL(initMins, off1_hibits), off1_current_bytes )
	b.Raw("VTBL V9.B16, [V21.B16], V10.B16")
	emitSGT(b, "V10.B16", "V8.B16", "V10.B16") // initial_under
	// second_under = sgt( VTBL(secMins, off1_hibits), current_bytes )
	b.Raw("VTBL V9.B16, [V22.B16], V11.B16")
	emitSGT(b, "V11.B16", "V0.B16", "V11.B16") // second_under
	b.Raw("VAND V11.B16, V10.B16, V10.B16")
	b.Raw("VORR V10.B16, V15.B16, V15.B16")
	// --- update previous = current (VORR self = mov) ---
	b.Raw("VORR V0.B16, V0.B16, V13.B16") // prev rawbytes
	b.Raw("VORR V1.B16, V1.B16, V14.B16") // prev high_nibbles
	b.Raw("VORR V3.B16, V3.B16, V12.B16") // prev carried_continuations
	b.Raw("ADD $16, R1")
	b.Raw("SUB $1, R2")
	b.Raw("CBNZ R2, vloop")
	b.Label("vdone")
	// ret = (has_error == 0) ? 1 : 0
	b.Raw("VMOV V15.D[0], R4")
	b.Raw("VMOV V15.D[1], R5")
	b.Raw("ORR R5, R4, R4")
	b.Raw("CMP $0, R4")
	b.Raw("CSET EQ, R6")
	b.Raw("MOVD R6, ret+32(FP)")
	b.Ret()
	f.Add(b.Func())

	// ====================================================================
	// asciiBlocks: return the number of leading 16-byte blocks that are pure
	// ASCII (every byte < 0x80), stopping at the first block with a high bit.
	// Per block this is a single load + a high-bit test, so a pure-ASCII run is
	// memory-bound — matching the stdlib word-at-a-time ASCII fast path, which a
	// full Lemire–Keiser pass would otherwise regress badly. The caller uses the
	// returned block count to confirm an all-ASCII buffer is valid and
	// rune-count == len in one cheap pass, and to skip the heavy validator over
	// the ASCII prefix.
	// ====================================================================
	a := arm64.NewFunc("asciiBlocks", countSig(), 0)
	a.LoadArg("src_base", "R1").
		LoadArg("n", "R2").
		Raw("MOVD $0, R0"). // ASCII block count = 0
		// Fast lane: process 4 blocks (64 B) at a time, OR-ing them into one
		// accumulator and testing the high bits once per 64 B. This keeps the
		// common all-ASCII run close to memory bandwidth (the per-block
		// horizontal reduction below is only used to pinpoint the exact stop
		// block once a high bit is seen).
		Raw("LSR $2, R2, R6"). // R6 = number of 4-block groups
		Raw("CBZ R6, atail").
		Label("aloop4").
		Raw("VLD1.P 64(R1), [V0.B16, V1.B16, V2.B16, V3.B16]").
		Raw("VORR V1.B16, V0.B16, V0.B16").
		Raw("VORR V3.B16, V2.B16, V2.B16").
		Raw("VORR V2.B16, V0.B16, V0.B16"). // V0 = OR of the 4 blocks
		Raw("VMOV V0.D[0], R4").
		Raw("VMOV V0.D[1], R5").
		Raw("ORR R5, R4, R4").
		Raw("AND $0x8080808080808080, R4, R4").
		Raw("CBNZ R4, aback"). // some high bit in this group: rescan it byte-block-wise
		Raw("ADD $4, R0").
		Raw("SUB $1, R6").
		Raw("CBNZ R6, aloop4").
		Raw("B atail").
		// aback: a high bit appeared in the last 64 B group; step R1 back to the
		// group start and fall through to the per-block tail scan to find the
		// exact first non-ASCII block.
		Label("aback").
		Raw("SUB $64, R1, R1").
		// atail: per-block scan of the remaining blocks (the < 4 leftover, or the
		// group that contained the first high bit), stopping at the first
		// non-ASCII block.
		Label("atail").
		// remaining = original n (still in R2) - R0 (blocks already counted).
		Raw("SUB R0, R2, R7"). // R7 = remaining blocks to scan
		Raw("CBZ R7, adone").
		Label("aloop1").
		Raw("VLD1 (R1), [V0.B16]").
		Raw("VUSHR $7, V0.B16, V0.B16"). // 1 where high bit set
		Raw("VUADDLV V0.B16, V1").       // sum of high bits across the block
		Raw("VMOV V1.H[0], R4").
		Raw("CBNZ R4, adone"). // first non-ASCII block: stop
		Raw("ADD $16, R1").
		Raw("ADD $1, R0").
		Raw("SUB $1, R7").
		Raw("CBNZ R7, aloop1").
		Label("adone").
		Raw("MOVD R0, ret+32(FP)").
		Ret()
	f.Add(a.Func())

	// ====================================================================
	// countCont: count UTF-8 continuation bytes ((b & 0xC0) == 0x80) over n
	// 16-byte blocks. Per block: mark continuations as 0xFF and horizontally
	// sum (VUADDLV gives sum of the 16 bytes; each marked byte contributes
	// 0xFF=255, so block count = sum/255). Accumulate into R0.
	// ====================================================================
	c := arm64.NewFunc("countCont", countSig(), 0)
	c.LoadArg("src_base", "R1").
		LoadArg("n", "R2").
		Raw("MOVD $0, R0").                           // cont count = 0
		Raw("MOVD $0xC0, R3").Raw("VDUP R3, V1.B16"). // 0xC0 mask
		Raw("MOVD $0x80, R3").Raw("VDUP R3, V2.B16"). // 0x80
		Raw("CBZ R2, cdone").
		Label("cloop").
		Raw("VLD1 (R1), [V0.B16]").
		Raw("VAND V1.B16, V0.B16, V0.B16").  // b & 0xC0
		Raw("VCMEQ V2.B16, V0.B16, V0.B16"). // == 0x80 ? 0xFF : 0x00
		Raw("VUADDLV V0.B16, V3").           // V3 = sum of the 16 bytes (in H[0])
		Raw("VMOV V3.H[0], R4").             // R4 = sum (0..16*255)
		Raw("ADD R4, R0, R0").
		Raw("ADD $16, R1").
		Raw("SUB $1, R2").
		Raw("CBNZ R2, cloop").
		Label("cdone").
		// each continuation byte contributed 255; divide the byte-sum by 255.
		Raw("MOVD $255, R5").
		Raw("UDIV R5, R0, R0").
		Raw("MOVD R0, ret+32(FP)").
		Ret()
	f.Add(c.Func())

	if err := os.WriteFile("utf8_arm64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote utf8_arm64.s")
}

// emitSGT emits a signed byte compare-greater (a > b -> 0xFF/0x00) into dst,
// using the sign-flip + unsigned trick (NEON-Go has only VCMEQ). It clobbers
// V16/V17/V18 as scratch and requires V29=0x80 splat and V28=all-ones to be
// loaded. dst may alias a or b. Returns the builder for chaining.
func emitSGT(b *arm64.Builder, a, bb, dst string) *arm64.Builder {
	return b.
		Raw("VEOR V29.B16, %s, V16.B16", a).    // a' = a ^ 0x80
		Raw("VEOR V29.B16, %s, V17.B16", bb).   // b' = b ^ 0x80
		Raw("VUMIN V17.B16, V16.B16, V18.B16"). // umin(a',b')
		Raw("VCMEQ V16.B16, V18.B16, V18.B16"). // a' <= b'  (i.e. a <= b)
		Raw("VEOR V28.B16, V18.B16, %s", dst)   // dst = NOT(a<=b) = a > b
}
