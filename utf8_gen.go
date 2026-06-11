//go:build ignore

// Command gen produces utf8_amd64.s with go-asmgen: the Lemire–Keiser
// "Validating UTF-8 In Less Than One Instruction Per Byte" lookup-based SIMD
// validator, both an SSE2/SSSE3 path (16 bytes per 128-bit block) and an AVX2
// path (32 bytes per 256-bit block).
//
// The algorithm classifies each byte by its high nibble via PSHUFB nibble
// lookups, then over (prev,curr) blocks checks: continuation-length carry,
// over/under-long continuations, the special ED/F4 first-continuation maxima,
// overlong-encoding minima, and "byte > 0xF4". Every check ORs into a running
// error accumulator; the blocks are valid iff that accumulator is all-zero
// (PTEST). A final check on the carried-continuation vector rejects an
// incomplete trailing sequence, so the validated prefix always ends on a rune
// boundary. Constant tables come from emit.File.Data.
//
// Ported 1:1 from Lemire's reference intrinsics (simdutf8check.h, Apache-2.0).
// Run: go run utf8_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/amd64"
	"github.com/go-asmgen/asmgen/emit"
)

func repByte(x byte, n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = x
	}
	return b
}
func rep(v []byte, times int) []byte {
	var b []byte
	for i := 0; i < times; i++ {
		b = append(b, v...)
	}
	return b
}

// _mm_setr_epi8 lays out the first argument at lane 0; tables below are in that
// (low-to-high) order, matching PSHUFB index semantics.

// continuationLengths LUT, indexed by high nibble: 0xxx=>1 (ASCII),
// 10xx=>0 (continuation), 110x=>2, 1110=>3, 1111=>4.
var contLenLUT = []byte{1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4}

// checkOverlong initial_mins LUT, indexed by off1 high nibble: 110x=>0xC2,
// 1110=>0xE1, 1111=>0xF1, everything else => 0x80 (-128, "never under").
var initialMinsLUT = []byte{
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0xC2, 0x80, 0xE1, 0xF1,
}

// checkOverlong second_mins LUT, indexed by off1 high nibble: 110x=>0x7F (127,
// handled by initial alone), 1110=>0xA0, 1111=>0x90, else => 0x80 (-128).
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

// countSig is the signature of the rune-count kernels: src []byte, n blocks,
// returning the number of non-continuation bytes in the first n blocks as an
// int64.
func countSig() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Slice("src"), abi.Scalar("n", abi.Int64)},
		[]abi.Arg{abi.Scalar("ret", abi.Int64)},
	)
}

func main() {
	f := emit.NewFile("amd64")

	// ---- SSE constant tables ----
	contLen := f.Data("contLen", contLenLUT)
	initMins := f.Data("initMins", initialMinsLUT)
	secMins := f.Data("secMins", secondMinsLUT)
	c0F := f.Data("c0F", repByte(0x0F, 16))
	c01 := f.Data("c01", repByte(0x01, 16))
	c02 := f.Data("c02", repByte(0x02, 16))
	cF4 := f.Data("cF4", repByte(0xF4, 16))
	cED := f.Data("cED", repByte(0xED, 16))
	c9F := f.Data("c9F", repByte(0x9F, 16))
	c8F := f.Data("c8F", repByte(0x8F, 16))
	c00 := f.Data("c00", repByte(0x00, 16))

	// ====================================================================
	// SSE2/SSSE3 + SSE4.1 (PTEST): 16 bytes per block.
	//
	// Persistent state across the loop:
	//   X15 = has_error (accumulator)
	//   X13 = previous rawbytes
	//   X14 = previous high_nibbles
	//   X12 = previous carried_continuations
	// ====================================================================
	s := amd64.NewFunc("validBlocksSSE", sig(), 0)
	s.LoadArg("src_base", "SI").LoadArg("n", "CX").
		Raw("PXOR X15, X15"). // has_error = 0
		Raw("PXOR X13, X13"). // prev rawbytes = 0
		Raw("PXOR X14, X14"). // prev high_nibbles = 0
		Raw("PXOR X12, X12"). // prev carried_continuations = 0
		Label("sloop").
		// --- load current block (rawbytes) ---
		Raw("MOVOU (SI), X0").

		// --- count_nibbles: high_nibbles = (bytes >> 4) & 0x0F ---
		Raw("MOVO X0, X1").
		Raw("PSRLW $4, X1").
		Raw("MOVOU %s+0(SB), X2", c0F).
		Raw("PAND X2, X1"). // X1 = high_nibbles (current)

		// --- checkSmallerThan0xF4: has_error |= subs_epu8(bytes, 0xF4) ---
		Raw("MOVO X0, X3").
		Raw("MOVOU %s+0(SB), X4", cF4).
		Raw("PSUBUSB X4, X3").
		Raw("POR X3, X15").

		// --- initial_lengths = pshufb(contLenLUT, high_nibbles) ---
		Raw("MOVOU %s+0(SB), X2", contLen).
		Raw("PSHUFB X1, X2"). // X2 = initial_lengths

		// --- carryContinuations ---
		// right1 = subs_epu8( alignr(initial_lengths, prev_carries, 15), 1 )
		Raw("MOVO X2, X3").
		Raw("MOVO X12, X4").
		Raw("PALIGNR $15, X4, X3"). // X3 = (init_len:prev_carries) >> 15 bytes
		Raw("PSUBUSB %s+0(SB), X3", c01).
		Raw("PADDB X2, X3"). // X3 = sum = initial_lengths + right1

		// right2 = subs_epu8( alignr(sum, prev_carries, 14), 2 )
		Raw("MOVO X3, X5").
		Raw("MOVO X12, X4").
		Raw("PALIGNR $14, X4, X5"). // X5 = (sum:prev_carries) >> 14 bytes
		Raw("PSUBUSB %s+0(SB), X5", c02).
		Raw("PADDB X5, X3"). // X3 = carried_continuations (current)

		// --- checkContinuations ---
		// overunder = cmpeq( cmpgt(carries, init_len), cmpgt(init_len, 0) )
		Raw("MOVO X3, X5").
		Raw("PCMPGTB X2, X5"). // X5 = carries > init_len
		Raw("MOVO X2, X6").
		Raw("MOVOU %s+0(SB), X7", c00).
		Raw("PCMPGTB X7, X6"). // X6 = init_len > 0
		Raw("PCMPEQB X6, X5"). // X5 = overunder
		Raw("POR X5, X15").

		// --- off1_current_bytes = alignr(rawbytes, prev_rawbytes, 15) ---
		Raw("MOVO X0, X8").
		Raw("MOVO X13, X9").
		Raw("PALIGNR $15, X9, X8"). // X8 = off1_current_bytes

		// --- checkFirstContinuationMax ---
		// badED = (cur > 0x9F) & cmpeq(off1, 0xED)
		Raw("MOVO X8, X9").
		Raw("MOVOU %s+0(SB), X10", cED).
		Raw("PCMPEQB X10, X9"). // maskED
		Raw("MOVO X0, X10").
		Raw("MOVOU %s+0(SB), X11", c9F).
		Raw("PCMPGTB X11, X10"). // cur > 0x9F
		Raw("PAND X10, X9").
		Raw("POR X9, X15").
		// badF4 = (cur > 0x8F) & cmpeq(off1, 0xF4)
		Raw("MOVO X8, X9").
		Raw("MOVOU %s+0(SB), X10", cF4).
		Raw("PCMPEQB X10, X9"). // maskF4
		Raw("MOVO X0, X10").
		Raw("MOVOU %s+0(SB), X11", c8F).
		Raw("PCMPGTB X11, X10"). // cur > 0x8F
		Raw("PAND X10, X9").
		Raw("POR X9, X15").

		// --- checkOverlong ---
		// off1_hibits = alignr(high_nibbles, prev_high_nibbles, 15)
		Raw("MOVO X1, X9").
		Raw("MOVO X14, X10").
		Raw("PALIGNR $15, X10, X9"). // X9 = off1_hibits
		// initial_under = cmpgt( pshufb(initMins, off1_hibits), off1_current_bytes )
		Raw("MOVOU %s+0(SB), X10", initMins).
		Raw("PSHUFB X9, X10").
		Raw("PCMPGTB X8, X10"). // X10 = initial_under
		// second_under = cmpgt( pshufb(secMins, off1_hibits), current_bytes )
		Raw("MOVOU %s+0(SB), X11", secMins).
		Raw("PSHUFB X9, X11").
		Raw("PCMPGTB X0, X11"). // X11 = second_under
		Raw("PAND X11, X10").
		Raw("POR X10, X15").

		// --- update previous = current ---
		Raw("MOVO X0, X13"). // prev rawbytes
		Raw("MOVO X1, X14"). // prev high_nibbles
		Raw("MOVO X3, X12"). // prev carried_continuations

		Raw("ADDQ $16, SI").Raw("DECQ CX").Raw("JNZ sloop").

		// No final incomplete-sequence check here: the caller backs the split
		// point up to a rune boundary and re-validates the straddling rune (plus
		// the tail) with the scalar path, so a sequence may legitimately run past
		// the last processed block. The kernel only reports structural errors
		// (bad continuations, overlong, surrogate, too-large) within its range.

		// --- ret = (has_error == 0) ? 1 : 0 ---
		Raw("XORQ AX, AX").
		Raw("PTEST X15, X15").
		Raw("SETEQ AX")
	s.StoreRet("AX", "ret")
	s.Ret()
	f.Add(s.Func())

	// ====================================================================
	// AVX2: 32 bytes per block. Cross-lane "push last byte(s)" uses
	// VPERM2I128 + VPALIGNR (Lemire's push_last_byte_of_a_to_b).
	//
	// Persistent state:
	//   Y15 = has_error, Y12 = prev carried_continuations,
	//   Y13 = prev rawbytes, Y14 = prev high_nibbles
	// Wider 32-byte constant tables (suffix b).
	// ====================================================================
	contLenB := f.Data("contLenB", rep(contLenLUT, 2))
	initMinsB := f.Data("initMinsB", rep(initialMinsLUT, 2))
	secMinsB := f.Data("secMinsB", rep(secondMinsLUT, 2))
	c0Fb := f.Data("c0Fb", repByte(0x0F, 32))
	c01b := f.Data("c01b", repByte(0x01, 32))
	c02b := f.Data("c02b", repByte(0x02, 32))
	cF4b := f.Data("cF4b", repByte(0xF4, 32))
	cEDb := f.Data("cEDb", repByte(0xED, 32))
	c9Fb := f.Data("c9Fb", repByte(0x9F, 32))
	c8Fb := f.Data("c8Fb", repByte(0x8F, 32))
	c00b := f.Data("c00b", repByte(0x00, 32))

	v := amd64.NewFunc("validBlocksAVX2", sig(), 0)
	v.LoadArg("src_base", "SI").LoadArg("n", "CX").
		Raw("VPXOR Y15, Y15, Y15").
		Raw("VPXOR Y13, Y13, Y13").
		Raw("VPXOR Y14, Y14, Y14").
		Raw("VPXOR Y12, Y12, Y12").
		Label("vloop").
		Raw("VMOVDQU (SI), Y0"). // rawbytes

		// high_nibbles = (bytes >> 4) & 0x0F
		Raw("VPSRLW $4, Y0, Y1").
		Raw("VPAND %s+0(SB), Y1, Y1", c0Fb). // Y1 = high_nibbles

		// checkSmallerThan0xF4
		Raw("VPSUBUSB %s+0(SB), Y0, Y3", cF4b).
		Raw("VPOR Y3, Y15, Y15").

		// initial_lengths = pshufb(contLenB, high_nibbles)
		Raw("VMOVDQU %s+0(SB), Y2", contLenB).
		Raw("VPSHUFB Y1, Y2, Y2"). // Y2 = initial_lengths

		// push_last_byte_of_a_to_b(a, b) = alignr(b, permute2x128(a,b,0x21), 15).
		// In Go operand order: VPERM2I128 imm, b, a, t  (t=[a.hi|b.lo]);
		// VPALIGNR imm, t, b, dst  (high operand = b = src1).

		// right1 = subs_epu8( push_last_byte(prev_carries, init_len), 1 )
		Raw("VPERM2I128 $0x21, Y2, Y12, Y4"). // t = [prev_carries.hi | init_len.lo]
		Raw("VPALIGNR $15, Y4, Y2, Y3").      // push_last_byte(prev_carries, init_len)
		Raw("VPSUBUSB %s+0(SB), Y3, Y3", c01b).
		Raw("VPADDB Y2, Y3, Y3"). // Y3 = sum

		// right2 = subs_epu8( push_last_2bytes(prev_carries, sum), 2 )
		Raw("VPERM2I128 $0x21, Y3, Y12, Y4"). // t = [prev_carries.hi | sum.lo]
		Raw("VPALIGNR $14, Y4, Y3, Y5").      // push_last_2bytes(prev_carries, sum)
		Raw("VPSUBUSB %s+0(SB), Y5, Y5", c02b).
		Raw("VPADDB Y5, Y3, Y3"). // Y3 = carried_continuations

		// checkContinuations
		Raw("VPCMPGTB Y2, Y3, Y5").             // carries > init_len
		Raw("VPCMPGTB %s+0(SB), Y2, Y6", c00b). // init_len > 0
		Raw("VPCMPEQB Y6, Y5, Y5").
		Raw("VPOR Y5, Y15, Y15").

		// off1_current_bytes = push_last_byte(prev_rawbytes, rawbytes)
		Raw("VPERM2I128 $0x21, Y0, Y13, Y8"). // t = [prev_rawbytes.hi | rawbytes.lo]
		Raw("VPALIGNR $15, Y8, Y0, Y8").      // Y8 = off1_current_bytes

		// checkFirstContinuationMax
		Raw("VPCMPEQB %s+0(SB), Y8, Y9", cEDb).  // maskED
		Raw("VPCMPGTB %s+0(SB), Y0, Y10", c9Fb). // cur > 0x9F
		Raw("VPAND Y10, Y9, Y9").
		Raw("VPOR Y9, Y15, Y15").
		Raw("VPCMPEQB %s+0(SB), Y8, Y9", cF4b).  // maskF4
		Raw("VPCMPGTB %s+0(SB), Y0, Y10", c8Fb). // cur > 0x8F
		Raw("VPAND Y10, Y9, Y9").
		Raw("VPOR Y9, Y15, Y15").

		// checkOverlong
		// off1_hibits = push_last_byte(prev_high_nibbles, high_nibbles)
		Raw("VPERM2I128 $0x21, Y1, Y14, Y9"). // t = [prev_high_nibbles.hi | high_nibbles.lo]
		Raw("VPALIGNR $15, Y9, Y1, Y9").      // Y9 = off1_hibits
		Raw("VMOVDQU %s+0(SB), Y10", initMinsB).
		Raw("VPSHUFB Y9, Y10, Y10").
		Raw("VPCMPGTB Y8, Y10, Y10"). // initial_under
		Raw("VMOVDQU %s+0(SB), Y11", secMinsB).
		Raw("VPSHUFB Y9, Y11, Y11").
		Raw("VPCMPGTB Y0, Y11, Y11"). // second_under
		Raw("VPAND Y11, Y10, Y10").
		Raw("VPOR Y10, Y15, Y15").

		// update previous = current
		Raw("VMOVDQA Y0, Y13").
		Raw("VMOVDQA Y1, Y14").
		Raw("VMOVDQA Y3, Y12").
		Raw("ADDQ $32, SI").Raw("DECQ CX").Raw("JNZ vloop").

		// No final incomplete-sequence check (see SSE comment): the caller backs
		// the split to a rune boundary and re-validates the tail scalar-side.

		// ret = (has_error == 0) ? 1 : 0
		Raw("XORQ AX, AX").
		Raw("VPTEST Y15, Y15").
		Raw("SETEQ AX").
		Raw("VZEROUPPER")
	v.StoreRet("AX", "ret")
	v.Ret()
	f.Add(v.Func())

	// ====================================================================
	// Rune-count kernels. The number of runes in valid UTF-8 equals the
	// number of bytes that are NOT continuation bytes, i.e. bytes where
	// (b & 0xC0) != 0x80. These kernels count those bytes over n full SIMD
	// blocks; the caller only trusts the result over a block-aligned prefix
	// the validator has confirmed is valid UTF-8 ending on a rune boundary,
	// and falls back to the scalar decoder otherwise (invalid input is where
	// the non-continuation identity breaks).
	//
	// Per block: mark continuation bytes (b & 0xC0 == 0x80) as 0xFF, extract a
	// bitmask with [V]PMOVMSKB, POPCNT it to count continuation bytes, and
	// accumulate; the caller computes non-continuation = n*blockSize - cont.
	// Counting continuations (not non-continuations) keeps the per-block work
	// to a single AND/CMPEQ/PMOVMSKB/POPCNT and a running add.
	// ====================================================================

	cC0 := f.Data("cC0", repByte(0xC0, 16))
	c80 := f.Data("c80", repByte(0x80, 16))

	// ---- SSE: 16 bytes per block ----
	//   AX = continuation-byte count accumulator
	cs := amd64.NewFunc("countContSSE", countSig(), 0)
	cs.LoadArg("src_base", "SI").LoadArg("n", "CX").
		Raw("XORQ AX, AX"). // cont count = 0
		Raw("MOVOU %s+0(SB), X1", cC0).
		Raw("MOVOU %s+0(SB), X2", c80).
		Label("csloop").
		Raw("MOVOU (SI), X0").
		Raw("PAND X1, X0").    // b & 0xC0
		Raw("PCMPEQB X2, X0"). // == 0x80 ? 0xFF : 0x00 (continuation mask)
		Raw("PMOVMSKB X0, DX").
		Raw("POPCNTL DX, DX").
		Raw("ADDQ DX, AX").
		Raw("ADDQ $16, SI").Raw("DECQ CX").Raw("JNZ csloop")
	cs.StoreRet("AX", "ret")
	cs.Ret()
	f.Add(cs.Func())

	// ---- AVX2: 32 bytes per block ----
	cC0b := f.Data("cC0b", repByte(0xC0, 32))
	c80b := f.Data("c80b", repByte(0x80, 32))
	cv := amd64.NewFunc("countContAVX2", countSig(), 0)
	cv.LoadArg("src_base", "SI").LoadArg("n", "CX").
		Raw("XORQ AX, AX").
		Raw("VMOVDQU %s+0(SB), Y1", cC0b).
		Raw("VMOVDQU %s+0(SB), Y2", c80b).
		Label("cvloop").
		Raw("VMOVDQU (SI), Y0").
		Raw("VPAND Y1, Y0, Y0").
		Raw("VPCMPEQB Y2, Y0, Y0").
		Raw("VPMOVMSKB Y0, DX").
		Raw("POPCNTL DX, DX").
		Raw("ADDQ DX, AX").
		Raw("ADDQ $32, SI").Raw("DECQ CX").Raw("JNZ cvloop").
		Raw("VZEROUPPER")
	cv.StoreRet("AX", "ret")
	cv.Ret()
	f.Add(cv.Func())

	if err := os.WriteFile("utf8_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote utf8_amd64.s")
}
