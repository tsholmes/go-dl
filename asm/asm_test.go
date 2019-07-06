package asm

import (
	"math/rand"
)

func vec(n int, min float64, max float64) []float64 {
	f := make([]float64, n)
	for i := range f {
		f[i] = min + rand.Float64()*(max-min)
	}
	return f
}
