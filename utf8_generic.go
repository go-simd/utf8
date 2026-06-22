//go:build !amd64 && !arm64 && !ppc64le && !s390x

package utf8

import "unicode/utf8"

// valid has no SIMD kernel on this arch; defer entirely to the standard library.
func valid(p []byte) bool { return utf8.Valid(p) }

// runeCount has no SIMD kernel on this arch; defer to the standard library.
func runeCount(p []byte) int { return utf8.RuneCount(p) }
