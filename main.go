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

	x := tensor.Input([]int{2})
	y := tensor.Input([]int{2})

	t := tensor.Sub(x, tensor.Mul(y, tensor.Constant(calc.Constant(0.5, 2))))

	out := tensor.Mul(t, t)

	curX := calc.RandomUniform(-10.0, 10.0, 2)
	curY := calc.RandomUniform(-10.0, 10.0, 2)

	gradient := tensor.Gradients(out)[x.ID()]

	eval := tensor.MakeEvaluation(gradient)

	for i := 0; i < 100; i++ {
		val := eval.Evaluate(tensor.Provide(x, curX), tensor.Provide(y, curY))
		grad := val[0]

		curX = curX.Add(grad.MulConstant(-0.05))

		fmt.Println(curX, curY, grad)
	}
}
