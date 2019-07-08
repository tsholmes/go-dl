package asm

import (
	"fmt"
	"testing"
	"time"
)

func BenchmarkLoopReLU(b *testing.B) {
	v := vec(1e8, 0, 1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		LoopReLU(v)
	}
}

var fReLUBig func([]float64)

func BenchmarkCompileReLU(b *testing.B) {
	v := vec(1e8, -1, 1)
	if fReLUBig == nil {
		start := time.Now()
		fReLUBig = CompileReLU(len(v))
		end := time.Now()
		fmt.Printf("Compiled in %s\n", end.Sub(start).String())
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fReLUBig(v)
	}
}

func TestLoopReLU(t *testing.T) {
	v := vec(10, -1, 1)

	LoopReLU(v)
}

func TestCompileReLU(t *testing.T) {
	v := vec(10, -1, 1)

	f := CompileReLU(len(v))

	f(v)
}

func BenchmarkLoopReLUCopy(b *testing.B) {
	v := vec(1e8, 0, 1)
	v2 := make([]float64, len(v))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		LoopReLUCopy(v, v2)
	}
}

var fReLUCopyBig func([]float64, []float64)

func BenchmarkCompileReLUCopy(b *testing.B) {
	v := vec(1e8, -1, 1)
	v2 := make([]float64, len(v))

	if fReLUCopyBig == nil {
		start := time.Now()
		fReLUCopyBig = CompileReLUCopy(len(v))
		end := time.Now()
		fmt.Printf("Compiled in %s\n", end.Sub(start).String())
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		fReLUCopyBig(v, v2)
	}
}

func TestLoopReLUCopy(t *testing.T) {
	v := vec(10, -1, 1)
	v2 := make([]float64, len(v))

	LoopReLUCopy(v, v2)
}

func TestCompileReLUCopy(t *testing.T) {
	v := vec(10, -1, 1)
	v2 := make([]float64, len(v))

	f := CompileReLUCopy(len(v))

	f(v, v2)
}

func BenchmarkLoopReLUMask(b *testing.B) {
	v1 := vec(1e8, 0, 1)
	v2 := vec(1e8, 0, 1)
	v3 := make([]float64, len(v1))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		LoopReLUMask(v1, v2, v3)
	}
}

var fReLUMaskBig func([]float64, []float64, []float64)

func BenchmarkCompileReLUMask(b *testing.B) {
	v1 := vec(1e8, -1, 1)
	v2 := vec(1e8, -1, 1)
	v3 := make([]float64, len(v1))

	if fReLUMaskBig == nil {
		start := time.Now()
		fReLUMaskBig = CompileReLUMask(len(v1))
		end := time.Now()
		fmt.Printf("Compiled in %s\n", end.Sub(start).String())
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		fReLUMaskBig(v1, v2, v3)
	}
}

func TestLoopReLUMask(t *testing.T) {
	v1 := vec(10, -1, 1)
	v2 := vec(10, -1, 1)
	v3 := make([]float64, len(v1))

	LoopReLUMask(v1, v2, v3)
}

func TestCompileReLUMask(t *testing.T) {
	v1 := vec(10, -1, 1)
	v2 := vec(10, -1, 1)
	v3 := make([]float64, len(v1))

	f := CompileReLUMask(len(v1))

	f(v1, v2, v3)
}
