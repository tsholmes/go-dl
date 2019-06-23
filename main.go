package main

import (
	"fmt"

	"github.com/tsholmes/go-dl/calc"
	"github.com/tsholmes/go-dl/tensor"
)

func main() {
	i1 := tensor.Input([]int{1})
	i2 := tensor.Input([]int{1})

	x := tensor.Mul(i1, i1)
	y := tensor.Mul(i2, i2)

	out := tensor.Concat(0, x, y, tensor.Add(x, y))

	eval := tensor.MakeEvaluation(out)

	val := eval.Evaluate(
		tensor.Provide(i1, calc.Ones([]int{1}).MulConstant(3)),
		tensor.Provide(i2, calc.Ones([]int{1}).MulConstant(4)),
	)

	fmt.Println(val[0].String())
}
