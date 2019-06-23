package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/tsholmes/go-dl/calc"
	"github.com/tsholmes/go-dl/tensor"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	x := tensor.Input(5, 1)
	y := tensor.Input(1, 1)
	z := tensor.Input(1, 1)

	t := tensor.Mul(tensor.Sub(x, y), tensor.Sub(x, z))

	out := tensor.Mul(t, t)

	curX := calc.RandomUniform(-1.0, 1.0, 5, 1)
	curY := calc.RandomUniform(-1.0, 1.0, 1, 1)
	curZ := calc.RandomUniform(-1.0, 1.0, 1, 1)

	gradient := tensor.Gradients(out)[x.ID()]

	eval := tensor.MakeEvaluation(gradient)

	for i := 0; i < 100; i++ {
		val := eval.Evaluate(
			tensor.Provide(x, curX),
			tensor.Provide(y, curY),
			tensor.Provide(z, curZ),
		)
		grad := val[0]

		fmt.Println(curX, curY, curZ)

		curX = curX.Add(grad.MulConstant(-0.1))
	}
}
