package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/tsholmes/go-dl/calc"
	"github.com/tsholmes/go-dl/dataset"
	"github.com/tsholmes/go-dl/model"
	"github.com/tsholmes/go-dl/tensor"
)

func weight(shape ...int) (tensor.Tensor, calc.NDArray) {
	nParams := 1
	for _, i := range shape {
		nParams *= i
	}
	init := math.Sqrt(2.0 / float64(nParams))

	t := tensor.Input(shape...)
	v := calc.RandomUniform(-init, init, shape...)
	return t, v
}

func main() {
	rand.Seed(time.Now().UnixNano())

	dataset.LoadMNIST()
	if true {
		return
	}

	batchSize := 100
	l1Size := 16
	l2Size := 32

	x := tensor.Input(batchSize, 4)
	y := tensor.Input(batchSize, 1)

	m := model.NewModel()

	var t tensor.Tensor = x

	t1, t2 := tensor.Slice(t, 1, 0, 2), tensor.Slice(t, 1, 2, 4)
	t = tensor.Concat(1, t1, t2)

	t = model.Dense(m, t, l1Size)
	t = tensor.ReLU(t)
	t = model.Dense(m, t, l2Size)
	t = tensor.ReLU(t)
	t = model.Dense(m, t, 1)

	pred := tensor.Sigmoid(t)
	loss := tensor.BinaryCrossEntropy(y, pred)

	m.Compile(x, y, pred, loss)

	Yfor := func(x calc.NDArray) calc.NDArray {
		// return x.Add(x.Greater(calc.Zeros(x.Shape()...)).Mul(x)).Add(calc.Constant(10.0, x.Shape()...))
		// return x.PowConstant(2.0)
		p1 := x.Slice(1, 0, 2)
		p2 := x.Slice(1, 2, 4)
		x = p1.Add(p2.MulConstant(-1.))
		return x.PowConstant(2.0).Sum(1).PowConstant(0.5).Greater(calc.Constant(0.5, 1, 1))
	}

	XTest := calc.RandomUniform(0.0, 1.0, batchSize, 4)
	YTest := Yfor(XTest)

	for i := 0; i < 10000; i++ {
		X := calc.RandomUniform(0.0, 1.0, batchSize, 4)
		Y := Yfor(X)

		// lr := math.Pow(10.0, -2-float64(i/500.0))
		lr := 1e-3

		trainLoss := m.Train(X, Y, lr)

		if (i+1)%100 == 0 {
			testLoss := m.Test(XTest, YTest)

			fmt.Println(trainLoss, testLoss)
		}
	}
}
