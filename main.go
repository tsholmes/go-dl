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

	x := tensor.Input(10)

	targetMean := tensor.Input(1)
	targetStddev := tensor.Input(1)

	mean := tensor.Mean(x, 0)
	stddev := tensor.PowConstant(
		tensor.Mean(
			tensor.PowConstant(
				tensor.Sub(x, mean),
				2,
			),
			0,
		),
		0.5,
	)

	meanErr := tensor.PowConstant(tensor.Sub(mean, targetMean), 2.)
	stdErr := tensor.PowConstant(tensor.Sub(stddev, targetStddev), 2.)

	curX := calc.RandomUniform(-1.0, 1.0, 10)
	curMeanTgt := calc.Zeros(1)
	curStddevTgt := calc.Ones(1)

	gradient := tensor.Gradients(meanErr, stdErr)[x.ID()]

	eval := tensor.MakeEvaluation(mean, stddev, gradient)

	for i := 0; i < 10000; i++ {
		val := eval.Evaluate(
			tensor.Provide(x, curX),
			tensor.Provide(targetMean, curMeanTgt),
			tensor.Provide(targetStddev, curStddevTgt),
		)
		m, std, grad := val[0], val[1], val[2]

		if (i+1)%100 == 0 {
			fmt.Println(curX, m, std)
		}

		curX = curX.Add(grad.MulConstant(-0.1))
	}
}
