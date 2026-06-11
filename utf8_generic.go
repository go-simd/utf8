//go:build !amd64

package utf8

import "unicode/utf8"

// valid has no SIMD kernel on this arch; defer entirely to the standard library.
func valid(p []byte) bool { return utf8.Valid(p) }
