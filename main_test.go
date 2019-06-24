package main

import (
	"math"
	"testing"

	"github.com/tsholmes/go-dl/calc"
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

func BenchmarkModel(b *testing.B) {

	batchSize := 100
	l1Size := 100
	l2Size := 200

	x := tensor.Input(batchSize, 2)
	y := tensor.Input(batchSize, 1)

	w1, w1v := weight(2, l1Size)
	b1, b1v := weight(1, l1Size)
	w2, w2v := weight(l1Size, l2Size)
	b2, b2v := weight(1, l2Size)
	w3, w3v := weight(l2Size, 1)
	b3, b3v := weight(1, 1)

	var t tensor.Tensor = x

	t = tensor.MatMul(t, w1, 0, 1)
	t = tensor.Add(t, b1)
	t = tensor.ReLU(t)
	t = tensor.MatMul(t, w2, 0, 1)
	t = tensor.Add(t, b2)
	t = tensor.ReLU(t)
	t = tensor.MatMul(t, w3, 0, 1)
	t = tensor.Add(t, b3)

	pred := tensor.Sigmoid(t)
	loss := tensor.BinaryCrossEntropy(y, pred)

	gradients := tensor.Gradients(loss)

	evalTrain := tensor.MakeEvaluation(
		loss,
		gradients[w1.ID()],
		gradients[b1.ID()],
		gradients[w2.ID()],
		gradients[b2.ID()],
		gradients[w3.ID()],
		gradients[b3.ID()],
	)

	Yfor := func(x calc.NDArray) calc.NDArray {
		// return x.Add(x.Greater(calc.Zeros(x.Shape()...)).Mul(x)).Add(calc.Constant(10.0, x.Shape()...))
		// return x.PowConstant(2.0)
		return x.PowConstant(2.0).Sum(1).PowConstant(0.5).Greater(calc.Constant(0.5, 1, 1))
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		X := calc.RandomUniform(0.0, 1.0, batchSize, 2)
		Y := Yfor(X)

		outs := evalTrain.Evaluate(
			tensor.Provide(x, X),
			tensor.Provide(y, Y),
			tensor.Provide(w1, w1v),
			tensor.Provide(b1, b1v),
			tensor.Provide(w2, w2v),
			tensor.Provide(b2, b2v),
			tensor.Provide(w3, w3v),
			tensor.Provide(b3, b3v),
		)

		// lr := -math.Pow(10.0, -2-math.Sqrt(float64(i)/500.0))
		lr := -1e-3
		w1v = w1v.Add(outs[1].MulConstant(lr))
		b1v = b1v.Add(outs[2].MulConstant(lr))
		w2v = w2v.Add(outs[3].MulConstant(lr))
		b2v = b2v.Add(outs[4].MulConstant(lr))
		w3v = w3v.Add(outs[5].MulConstant(lr))
		b3v = b3v.Add(outs[6].MulConstant(lr))
	}
}
