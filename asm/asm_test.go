package asm

import (
	"fmt"
	"math/rand"
	"testing"
	"time"
)

func vec(n int) []float64 {
	f := make([]float64, n)
	for i := range f {
		f[i] = rand.Float64()
	}
	return f
}

func BenchmarkLoopMulConstant(b *testing.B) {
	v := vec(1e8)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		LoopMulConstant(v, 0.99)
	}
}

func BenchmarkScalMulConstant(b *testing.B) {
	v := vec(1e8)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ScalMulConstant(v, 0.99)
	}
}

var fMulBig func([]float64, float64)

func BenchmarkCompileMulConstant(b *testing.B) {
	v := vec(1e8)
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
