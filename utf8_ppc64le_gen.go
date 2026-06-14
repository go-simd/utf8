//go:build ignore

// Command gen produces utf8_ppc64le.s with go-asmgen: the Lemire–Keiser
// "Validating UTF-8 In Less Than One Instruction Per Byte" lookup-based SIMD
// validator and the continuation-byte counter, for ppc64le (POWER8+, VSX is
// baseline so there is no runtime dispatch). 16 bytes per block.
//
// The kernels are a 1:1 port of the amd64 SSE path (utf8_gen.go), with these
// instruction substitutions:
//
//	amd64 (SSE)              ppc64le (VSX/AltiVec)
//	MOVOU load              LXVB16X  (natural byte order, see below)
//	PSRLW $4 + PAND 0x0F    VSRB v,four — per-byte >>4 yields the high nibble
//	PSUBUSB (subs_epu8)     VSUBUBS  (unsigned saturating byte subtract)
//	PADDB                   VADDUBM
//	PSHUFB tbl by idx       VPERM tbl,tbl,idx — 16-entry nibble lookup
//	PALIGNR $n prev,cur     VSLDOI $n prev,cur — shift the previous block's
//	                        last n bytes into the current block
//	PCMPGTB (signed)        VCMPGTSB
//	PCMPEQB                 VCMPEQUB
//	PAND / POR              VAND / VOR
//	PTEST x,x (all-zero?)   extract both doublewords (MFVSRD) and OR-test
//
// VSX↔AltiVec aliasing: an AltiVec register Vn is the SAME physical register as
// VSX register VS(32+n). LXVB16X writes a VS register, so constants and the
// source block are loaded into VS(32+k) and then operated on as Vk by the
// AltiVec arithmetic.
//
// Endianness / lane order: ppc64le is little-endian, but LXVB16X loads bytes in
// natural memory order (memory byte i -> vector lane i, lane 0 leftmost), so the
// whole algorithm — including VPERM nibble lookups and the VSLDOI "shift in the
// previous block" carry — uses exactly the amd64 memory-order lane semantics. We
// deliberately use LXVB16X, NOT LXVD2X (which swaps the two doublewords on LE).
// The position-dependent FuzzValid/FuzzRuneCount tests vs unicode/utf8 (run on
// all input, valid and invalid, under qemu) are the gate that the lane order and
// every operand order are correct.
//
// Run: GOWORK=off go run utf8_ppc64le_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/ppc64"
)

func repByte(x byte, n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = x
	}
	return b
}

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
	f := emit.NewFile("ppc64le")

	contLen := f.Data("contLen_p", contLenLUT)
	initMins := f.Data("initMins_p", initialMinsLUT)
	secMins := f.Data("secMins_p", secondMinsLUT)
	c04 := f.Data("c04_p", repByte(0x04, 16)) // shift count for >>4
	c01 := f.Data("c01_p", repByte(0x01, 16))
	c02 := f.Data("c02_p", repByte(0x02, 16))
	cF4 := f.Data("cF4_p", repByte(0xF4, 16))
	cED := f.Data("cED_p", repByte(0xED, 16))
	c9F := f.Data("c9F_p", repByte(0x9F, 16))
	c8F := f.Data("c8F_p", repByte(0x8F, 16))
	c00 := f.Data("c00_p", repByte(0x00, 16))

	// ====================================================================
	// validBlocksVSX: 16 bytes per block.
	//
	// Persistent vector state across the loop (AltiVec V regs):
	//   V20 = has_error (accumulator)
	//   V21 = previous rawbytes
	//   V22 = previous high_nibbles
	//   V23 = previous carried_continuations
	// Constants live in V24..V31; the source block and scratch in V0..V13.
	// ====================================================================
	s := ppc64.NewFunc("validBlocksVSX", sig(), 0)
	s.LoadArg("src_base", "R4").LoadArg("n", "R5").
		// load constants
		Raw("MOVD $%s+0(SB), R6", contLen).Raw("LXVB16X (R0)(R6), VS56").  // V24 = contLen LUT
		Raw("MOVD $%s+0(SB), R6", initMins).Raw("LXVB16X (R0)(R6), VS57"). // V25 = initMins LUT
		Raw("MOVD $%s+0(SB), R6", secMins).Raw("LXVB16X (R0)(R6), VS58").  // V26 = secMins LUT
		Raw("MOVD $%s+0(SB), R6", c04).Raw("LXVB16X (R0)(R6), VS59").      // V27 = 0x04
		Raw("MOVD $%s+0(SB), R6", c01).Raw("LXVB16X (R0)(R6), VS60").      // V28 = 0x01
		Raw("MOVD $%s+0(SB), R6", c02).Raw("LXVB16X (R0)(R6), VS61").      // V29 = 0x02
		Raw("MOVD $%s+0(SB), R6", cF4).Raw("LXVB16X (R0)(R6), VS62").      // V30 = 0xF4
		Raw("MOVD $%s+0(SB), R6", c00).Raw("LXVB16X (R0)(R6), VS63").      // V31 = 0x00
		// has_error/prev state := 0
		Raw("VXOR V20, V20, V20").
		Raw("VXOR V21, V21, V21").
		Raw("VXOR V22, V22, V22").
		Raw("VXOR V23, V23, V23").
		Raw("MOVD $0, R7"). // byte offset
		Label("sloop").
		Raw("LXVB16X (R7)(R4), VS32"). // V0 = rawbytes (current block)

		// high_nibbles = bytes >> 4
		Raw("VSRB V0, V27, V1"). // V1 = high_nibbles (current)

		// checkSmallerThan0xF4: has_error |= subs_epu8(bytes, 0xF4)
		Raw("VSUBUBS V0, V30, V3"). // V3 = subs_epu8(bytes, 0xF4) = bytes - 0xF4 (sat)
		Raw("VOR V3, V20, V20").

		// initial_lengths = pshufb(contLenLUT, high_nibbles)
		Raw("VPERM V24, V24, V1, V2"). // V2 = initial_lengths

		// --- carryContinuations ---
		// right1 = subs_epu8( alignr(initial_lengths, prev_carries, 15), 1 )
		Raw("VSLDOI $15, V23, V2, V3"). // V3 = [prev_carries[15], init_len[0..14]]
		Raw("VSUBUBS V3, V28, V3").     // V3 = subs_epu8(., 1) = V3 - 1 (sat)
		Raw("VADDUBM V2, V3, V3").      // V3 = sum = initial_lengths + right1

		// right2 = subs_epu8( alignr(sum, prev_carries, 14), 2 )
		Raw("VSLDOI $14, V23, V3, V5"). // V5 = [prev_carries[14..15], sum[0..13]]
		Raw("VSUBUBS V5, V29, V5").     // V5 = subs_epu8(., 2) = V5 - 2 (sat)
		Raw("VADDUBM V5, V3, V3").      // V3 = carried_continuations (current)

		// --- checkContinuations ---
		// overunder = cmpeq( cmpgt(carries, init_len), cmpgt(init_len, 0) )
		Raw("VCMPGTSB V3, V2, V5").    // V5 = carries > init_len
		Raw("VCMPGTSB V2, V31, V6").   // V6 = init_len > 0
		Raw("VCMPEQUB V6, V5, V5").    // V5 = overunder
		Raw("VOR V5, V20, V20").

		// off1_current_bytes = alignr(rawbytes, prev_rawbytes, 15)
		Raw("VSLDOI $15, V21, V0, V8"). // V8 = off1_current_bytes

		// --- checkFirstContinuationMax ---
		Raw("MOVD $%s+0(SB), R6", cED).Raw("LXVB16X (R0)(R6), VS41"). // V9 = 0xED
		Raw("VCMPEQUB V9, V8, V9").                                   // V9 = maskED (off1 == 0xED)
		Raw("MOVD $%s+0(SB), R6", c9F).Raw("LXVB16X (R0)(R6), VS42"). // V10 = 0x9F
		Raw("VCMPGTSB V0, V10, V10").                                 // V10 = cur > 0x9F
		Raw("VAND V10, V9, V9").
		Raw("VOR V9, V20, V20").
		Raw("MOVD $%s+0(SB), R6", cF4).Raw("LXVB16X (R0)(R6), VS41"). // V9 = 0xF4
		Raw("VCMPEQUB V9, V8, V9").                                   // V9 = maskF4 (off1 == 0xF4)
		Raw("MOVD $%s+0(SB), R6", c8F).Raw("LXVB16X (R0)(R6), VS42"). // V10 = 0x8F
		Raw("VCMPGTSB V0, V10, V10").                                 // V10 = cur > 0x8F
		Raw("VAND V10, V9, V9").
		Raw("VOR V9, V20, V20").

		// --- checkOverlong ---
		// off1_hibits = alignr(high_nibbles, prev_high_nibbles, 15)
		Raw("VSLDOI $15, V22, V1, V9"). // V9 = off1_hibits
		// initial_under = cmpgt( pshufb(initMins, off1_hibits), off1_current_bytes )
		Raw("VPERM V25, V25, V9, V10").
		Raw("VCMPGTSB V10, V8, V10"). // V10 = initial_under = min(off1) > off1_cur
		// second_under = cmpgt( pshufb(secMins, off1_hibits), current_bytes )
		Raw("VPERM V26, V26, V9, V11").
		Raw("VCMPGTSB V11, V0, V11"). // V11 = second_under = min2 > cur
		Raw("VAND V11, V10, V10").
		Raw("VOR V10, V20, V20").

		// update previous = current
		Raw("VOR V0, V0, V21"). // prev rawbytes
		Raw("VOR V1, V1, V22"). // prev high_nibbles
		Raw("VOR V3, V3, V23"). // prev carried_continuations

		Raw("ADD $16, R7").
		Raw("ADD $-1, R5").
		Raw("CMP R5, $0").Raw("BNE sloop").

		// ret = (has_error all-zero) ? 1 : 0
		Raw("MFVSRD VS52, R8").       // R8 = has_error doubleword 0
		Raw("VSLDOI $8, V20, V20, V4").
		Raw("MFVSRD VS36, R9").       // R9 = has_error doubleword 1
		Raw("OR R9, R8, R8").
		Raw("MOVD $1, R10").
		Raw("CMP R8, $0").
		Raw("BEQ sok").
		Raw("MOVD $0, R10").
		Label("sok")
	s.StoreRet("R10", "ret")
	s.Ret()
	f.Add(s.Func())

	// ====================================================================
	// countContVSX: count UTF-8 continuation bytes ((b & 0xC0) == 0x80) over
	// n 16-byte blocks. Per block: VAND 0xC0, VCMPEQUB 0x80 (0xFF per cont
	// byte), VAND 0x01 (-> 0x01 per cont byte), VPOPCNTD (per-doubleword bit
	// count = #cont bytes in that 8-byte half), then add both doublewords.
	// ====================================================================
	cC0 := f.Data("cC0_p", repByte(0xC0, 16))
	c80 := f.Data("c80_p", repByte(0x80, 16))
	c01b := f.Data("c01b_p", repByte(0x01, 16))

	cs := ppc64.NewFunc("countContVSX", countSig(), 0)
	cs.LoadArg("src_base", "R4").LoadArg("n", "R5").
		Raw("MOVD $%s+0(SB), R6", cC0).Raw("LXVB16X (R0)(R6), VS57"). // V25 = 0xC0
		Raw("MOVD $%s+0(SB), R6", c80).Raw("LXVB16X (R0)(R6), VS58"). // V26 = 0x80
		Raw("MOVD $%s+0(SB), R6", c01b).Raw("LXVB16X (R0)(R6), VS59"). // V27 = 0x01
		Raw("MOVD $0, R8").  // cont count accumulator
		Raw("MOVD $0, R7").  // byte offset
		Label("csloop").
		Raw("LXVB16X (R7)(R4), VS32"). // V0 = block
		Raw("VAND V0, V25, V0").       // b & 0xC0
		Raw("VCMPEQUB V0, V26, V0").   // == 0x80 ? 0xFF : 0x00
		Raw("VAND V0, V27, V0").       // 0x01 per continuation byte
		Raw("VPOPCNTD V0, V0").        // per-doubleword popcount = #cont bytes in half
		Raw("MFVSRD VS32, R9").        // doubleword 0 count
		Raw("ADD R9, R8").
		Raw("VSLDOI $8, V0, V0, V1").
		Raw("MFVSRD VS33, R9").        // doubleword 1 count
		Raw("ADD R9, R8").
		Raw("ADD $16, R7").
		Raw("ADD $-1, R5").
		Raw("CMP R5, $0").Raw("BNE csloop")
	cs.StoreRet("R8", "ret")
	cs.Ret()
	f.Add(cs.Func())

	if err := os.WriteFile("utf8_ppc64le.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote utf8_ppc64le.s")
}
