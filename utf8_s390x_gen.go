//go:build ignore

// Command gen produces utf8_s390x.s with go-asmgen: the Lemire–Keiser
// "Validating UTF-8 In Less Than One Instruction Per Byte" lookup-based SIMD
// validator, the continuation-byte counter, and a pure-ASCII pre-scan, for
// s390x (IBM Z, vector facility baseline on z13+, so no runtime dispatch). 16
// bytes per block.
//
// The kernels are a 1:1 port of the amd64 SSE path (utf8_gen.go), with these
// instruction substitutions:
//
//	amd64 (SSE)              s390x (vector facility)
//	MOVOU load              VL  (lane 0 = first memory byte; see below)
//	PSRLW $4 + PAND 0x0F    VESRLB $4 — per-element >>4 yields the high nibble
//	PSUBUSB (subs_epu8)     VMNLB+VSB — s390x has no unsigned saturating sub,
//	                        so subs_epu8(a,b) = a - min(a,b)
//	PADDB                   VAB
//	PSHUFB tbl by idx       VPERM tbl,tbl,idx — 16-entry nibble lookup
//	PALIGNR $n prev,cur     VSLDB $n prev,cur — shift the previous block's
//	                        last n bytes into the current block
//	PCMPGTB (signed)        VCHB (compare high, signed)
//	PCMPEQB                 VCEQB
//	PAND / POR              VN / VO
//	PTEST x,x (all-zero?)   extract both giant lanes (VLGVG) and OR-test
//
// BIG-ENDIAN: s390x is the only big-endian target, but VL places the lowest
// memory address in lane 0 (the leftmost/high-order lane), so lanes are in
// natural memory order — identical to the amd64 memory-order lane semantics.
// The byte-wise classification (AND/compare/shift), the VPERM nibble lookups
// and the VSLDB "shift in the previous block" carry are all lane-order-
// consistent, and the popcount reduction (VPOPCT + VSUMB) is order-invisible.
// No endian fix-up is needed. The position-dependent FuzzValid/FuzzRuneCount
// tests vs unicode/utf8 (on all input, valid and invalid, under qemu) are the
// gate that lane order and every operand order are correct.
//
// Run: GOWORK=off go run utf8_s390x_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/s390x"
)

func repByte(x byte, n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = x
	}
	return b
}

var contLenLUT = []byte{1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4}

var initialMinsLUT = []byte{
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0xC2, 0x80, 0xE1, 0xF1,
}

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
	f := emit.NewFile("s390x")

	contLen := f.Data("contLen_z", contLenLUT)
	initMins := f.Data("initMins_z", initialMinsLUT)
	secMins := f.Data("secMins_z", secondMinsLUT)
	c01 := f.Data("c01_z", repByte(0x01, 16))
	c02 := f.Data("c02_z", repByte(0x02, 16))
	cF4 := f.Data("cF4_z", repByte(0xF4, 16))
	cED := f.Data("cED_z", repByte(0xED, 16))
	c9F := f.Data("c9F_z", repByte(0x9F, 16))
	c8F := f.Data("c8F_z", repByte(0x8F, 16))

	// ====================================================================
	// validBlocksVX: 16 bytes per block.
	//
	// Persistent vector state (V regs):
	//   V20 = has_error, V21 = prev rawbytes, V22 = prev high_nibbles,
	//   V23 = prev carried_continuations
	// Constants: V24=contLen, V25=initMins, V26=secMins, V27=0xF4, V28=0x01,
	//   V29=0x02, V30=zero. Scratch: V0..V13.
	// ====================================================================
	s := s390x.NewFunc("validBlocksVX", sig(), 0)
	s.LoadArg("src_base", "R2").LoadArg("n", "R3").
		Raw("MOVD $%s+0(SB), R5", contLen).Raw("VL (R5), V24").
		Raw("MOVD $%s+0(SB), R5", initMins).Raw("VL (R5), V25").
		Raw("MOVD $%s+0(SB), R5", secMins).Raw("VL (R5), V26").
		Raw("MOVD $%s+0(SB), R5", cF4).Raw("VL (R5), V27").
		Raw("MOVD $%s+0(SB), R5", c01).Raw("VL (R5), V28").
		Raw("MOVD $%s+0(SB), R5", c02).Raw("VL (R5), V29").
		Raw("VZERO V30").
		Raw("VZERO V20"). // has_error
		Raw("VZERO V21"). // prev rawbytes
		Raw("VZERO V22"). // prev high_nibbles
		Raw("VZERO V23"). // prev carried_continuations
		Label("sloop").
		Raw("VL (R2), V0"). // rawbytes

		// high_nibbles = bytes >> 4
		Raw("VESRLB $4, V0, V1"). // V1 = high_nibbles

		// checkSmallerThan0xF4: has_error |= subs_epu8(bytes, 0xF4)
		// subs_epu8(a,b) = a - min(a,b)
		Raw("VMNLB V0, V27, V3"). // V3 = min(bytes, 0xF4)
		Raw("VSB V3, V0, V3").    // V3 = bytes - min = subs_epu8(bytes,0xF4)
		Raw("VO V3, V20, V20").

		// initial_lengths = pshufb(contLenLUT, high_nibbles)
		Raw("VPERM V24, V24, V1, V2"). // V2 = initial_lengths

		// --- carryContinuations ---
		// right1 = subs_epu8( alignr(initial_lengths, prev_carries, 15), 1 )
		Raw("VSLDB $15, V23, V2, V3"). // V3 = [prev_carries[15], init_len[0..14]]
		Raw("VMNLB V3, V28, V4").      // min(., 1)
		Raw("VSB V4, V3, V3").         // subs_epu8(.,1)
		Raw("VAB V2, V3, V3").         // sum = init_len + right1

		// right2 = subs_epu8( alignr(sum, prev_carries, 14), 2 )
		Raw("VSLDB $14, V23, V3, V5"). // V5 = [prev_carries[14..15], sum[0..13]]
		Raw("VMNLB V5, V29, V4").      // min(., 2)
		Raw("VSB V4, V5, V5").         // subs_epu8(.,2)
		Raw("VAB V5, V3, V3").         // carried_continuations

		// --- checkContinuations ---
		// overunder = cmpeq( cmpgt(carries, init_len), cmpgt(init_len, 0) )
		Raw("VCHB V3, V2, V5").   // V5 = carries > init_len (signed)
		Raw("VCHB V2, V30, V6").  // V6 = init_len > 0 (signed)
		Raw("VCEQB V6, V5, V5").  // overunder
		Raw("VO V5, V20, V20").

		// off1_current_bytes = alignr(rawbytes, prev_rawbytes, 15)
		Raw("VSLDB $15, V21, V0, V8"). // V8 = off1_current_bytes

		// --- checkFirstContinuationMax ---
		Raw("MOVD $%s+0(SB), R5", cED).Raw("VL (R5), V9").
		Raw("VCEQB V9, V8, V9"). // maskED (off1 == 0xED)
		Raw("MOVD $%s+0(SB), R5", c9F).Raw("VL (R5), V10").
		Raw("VCHB V0, V10, V10"). // cur > 0x9F
		Raw("VN V10, V9, V9").
		Raw("VO V9, V20, V20").
		Raw("MOVD $%s+0(SB), R5", cF4).Raw("VL (R5), V9").
		Raw("VCEQB V9, V8, V9"). // maskF4 (off1 == 0xF4)
		Raw("MOVD $%s+0(SB), R5", c8F).Raw("VL (R5), V10").
		Raw("VCHB V0, V10, V10"). // cur > 0x8F
		Raw("VN V10, V9, V9").
		Raw("VO V9, V20, V20").

		// --- checkOverlong ---
		// off1_hibits = alignr(high_nibbles, prev_high_nibbles, 15)
		Raw("VSLDB $15, V22, V1, V9"). // off1_hibits
		Raw("VPERM V25, V25, V9, V10").
		Raw("VCHB V10, V8, V10"). // initial_under = min(off1) > off1_cur
		Raw("VPERM V26, V26, V9, V11").
		Raw("VCHB V11, V0, V11"). // second_under = min2 > cur
		Raw("VN V11, V10, V10").
		Raw("VO V10, V20, V20").

		// update previous = current
		Raw("VLR V0, V21").
		Raw("VLR V1, V22").
		Raw("VLR V3, V23").

		Raw("ADD $16, R2").
		Raw("ADD $-1, R3").
		Raw("CMPBNE R3, $0, sloop").

		// ret = (has_error all-zero) ? 1 : 0
		Raw("VLGVG $0, V20, R8").
		Raw("VLGVG $1, V20, R9").
		Raw("OR R9, R8, R8").
		Raw("MOVD $1, R10").
		Raw("CMPBEQ R8, $0, sok").
		Raw("MOVD $0, R10").
		Label("sok")
	s.StoreRet("R10", "ret")
	s.Ret()
	f.Add(s.Func())

	// ====================================================================
	// countContVX: count continuation bytes ((b & 0xC0) == 0x80) over n
	// 16-byte blocks. Per block: VN 0xC0, VCEQB 0x80 (0xFF per cont byte),
	// VN 0x01 (-> 0x01 per cont byte), then reduce with VSUMB (sum bytes ->
	// words) and VSUMQF (sum words -> quadword); add the low 64 bits.
	// ====================================================================
	cC0 := f.Data("cC0_z", repByte(0xC0, 16))
	c80 := f.Data("c80_z", repByte(0x80, 16))
	c01b := f.Data("c01b_z", repByte(0x01, 16))

	cs := s390x.NewFunc("countContVX", countSig(), 0)
	cs.LoadArg("src_base", "R2").LoadArg("n", "R3").
		Raw("MOVD $%s+0(SB), R5", cC0).Raw("VL (R5), V25").
		Raw("MOVD $%s+0(SB), R5", c80).Raw("VL (R5), V26").
		Raw("MOVD $%s+0(SB), R5", c01b).Raw("VL (R5), V27").
		Raw("VZERO V30").
		Raw("MOVD $0, R8"). // accumulator
		Label("csloop").
		Raw("VL (R2), V0").
		Raw("VN V0, V25, V0").     // b & 0xC0
		Raw("VCEQB V0, V26, V0").  // == 0x80 ? 0xFF : 0x00
		Raw("VN V0, V27, V0").     // 0x01 per continuation byte
		Raw("VSUMB V0, V30, V0").  // sum each 4-byte group into a word
		Raw("VSUMQF V0, V30, V0"). // sum the 4 words into one quadword
		Raw("VLGVG $1, V0, R9").   // low 64 bits = block continuation count
		Raw("ADD R9, R8").
		Raw("ADD $16, R2").
		Raw("ADD $-1, R3").
		Raw("CMPBNE R3, $0, csloop")
	cs.StoreRet("R8", "ret")
	cs.Ret()
	f.Add(cs.Func())

	// ====================================================================
	// asciiBlocksVX: return the number of leading 16-byte blocks (out of n)
	// that are pure ASCII (every byte < 0x80), stopping at the first block
	// with a high bit set. Per block this is only a VL + a high-bit test, so a
	// pure-ASCII run is memory-bound — matching the stdlib word-at-a-time ASCII
	// fast path, which the full Lemire–Keiser validator would otherwise regress
	// badly on the (very common) all-ASCII case (measured on real z15: the heavy
	// validator runs the all-ASCII path at ~7.5x below stdlib). The caller uses
	// the returned block count to confirm an all-ASCII buffer is valid and
	// rune-count == len in one cheap pass and to skip the validator over the
	// ASCII prefix. Structurally mirrors the amd64 asciiBlocksSSE (group-of-4
	// fast lane + per-block tail rescan).
	//
	// Fast lane processes 4 blocks (64 B) at a time, VO-ing them into one
	// accumulator and testing the high bits once per group; on a hit it falls
	// into the per-block tail scan to pinpoint the first non-ASCII block. R8 =
	// ASCII block count. The high-bit test is lane-order-invisible, so the
	// big-endian VL lane order does not matter (the count advances in memory
	// order via R2, independent of lanes).
	// ====================================================================
	c80asc := f.Data("c80asc_z", repByte(0x80, 16))

	au := s390x.NewFunc("asciiBlocksVX", countSig(), 0)
	au.LoadArg("src_base", "R2").LoadArg("n", "R3").
		Raw("MOVD $%s+0(SB), R5", c80asc).Raw("VL (R5), V25").
		Raw("MOVD $0, R8").    // ASCII block count
		Raw("SRD $2, R3, R6"). // R6 = n >> 2 = number of 4-block groups
		Raw("CMPBEQ R6, $0, atail").
		Label("aloop4").
		Raw("VL (R2), V0").
		Raw("VL 16(R2), V1").
		Raw("VL 32(R2), V2").
		Raw("VL 48(R2), V3").
		Raw("VO V1, V0, V0").
		Raw("VO V3, V2, V2").
		Raw("VO V2, V0, V0").  // V0 = OR of the 4 blocks
		Raw("VN V0, V25, V0"). // high bits (byte & 0x80)
		Raw("VLGVG $0, V0, R9").
		Raw("VLGVG $1, V0, R10").
		Raw("OR R10, R9, R9").
		Raw("CMPBNE R9, $0, atail"). // some high bit in this group: rescan block-wise
		Raw("ADD $64, R2").
		Raw("ADD $4, R8").
		Raw("ADD $-1, R6").
		Raw("CMPBNE R6, $0, aloop4").
		Label("atail").
		Raw("SUB R8, R3, R7"). // R7 = n - count = blocks left to scan
		Raw("CMPBEQ R7, $0, adone").
		Label("aloop1").
		Raw("VL (R2), V0").
		Raw("VN V0, V25, V0").
		Raw("VLGVG $0, V0, R9").
		Raw("VLGVG $1, V0, R10").
		Raw("OR R10, R9, R9").
		Raw("CMPBNE R9, $0, adone"). // first non-ASCII block: stop
		Raw("ADD $16, R2").
		Raw("ADD $1, R8").
		Raw("ADD $-1, R7").
		Raw("CMPBNE R7, $0, aloop1").
		Label("adone")
	au.StoreRet("R8", "ret")
	au.Ret()
	f.Add(au.Func())

	if err := os.WriteFile("utf8_s390x.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote utf8_s390x.s")
}
