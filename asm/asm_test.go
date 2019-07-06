package asm

import (
	"fmt"
	"math/rand"
	"testing"
	"time"
)

func vec(n int, min float64, max float64) []float64 {
	f := make([]float64, n)
	for i := range f {
		f[i] = min + rand.Float64()*(max-min)
	}
	return f
}

func BenchmarkLoopMulConstant(b *testing.B) {
	v := vec(1e8, 0, 1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		LoopMulConstant(v, 0.99)
	}
}

func BenchmarkScalMulConstant(b *testing.B) {
	v := vec(1e8, 0, 1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ScalMulConstant(v, 0.99)
	}
}

var fMulBig func([]float64, float64)

func BenchmarkCompileMulConstant(b *testing.B) {
	v := vec(1e8, 0, 1)
	if fMulBig == nil {
		start := time.Now()
		fMulBig = CompileMulConstant(len(v))
		end := time.Now()
		fmt.Printf("Compiled in %s\n", end.Sub(start).String())
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fMulBig(v, 0.99)
	}
}

func TestCompileMulConstant(t *testing.T) {
	f := CompileMulConstant(4)
	v := []float64{1.0, 2.0, 3.0, 4.0}

	f(v, 0.99)
}

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
