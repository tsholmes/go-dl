package main

import (
	"math"
	"os"
	"testing"

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

var X, Y calc.NDArray

func loadData() {
	if len(X.Shape()) > 0 {
		return
	}
	Xfull, Yfull, _, _ := dataset.LoadMNIST()
	X = Xfull.Split(0, 10)[0]
	Y = Yfull.Split(0, 10)[0]
}

func TestMain(m *testing.M) {
	loadData()
	os.Exit(m.Run())
}

func BenchmarkModel(b *testing.B) {
	batchSize := 10
	l1Size := 16
	l2Size := 32

	x := tensor.Input(batchSize, 28, 28, 1)
	y := tensor.Input(batchSize, 10)

	m := model.NewModel()

	var t tensor.Tensor = tensor.Reshape(x, batchSize, 28, 28, 1)

	t = model.MaxPooling2D(m, t, 2, 2)

	t = model.Conv2D(m, t, 3, 3, l1Size)
	t = tensor.ReLU(t)
	t = model.MaxPooling2D(m, t, 2, 2)

	t = model.Conv2D(m, t, 3, 3, l2Size)
	t = tensor.ReLU(t)

	t = tensor.Flatten(t, 1)

	t = model.Dense(m, t, 10, true)

	pred := tensor.Softmax(t)
	loss := tensor.CategoricalCrossEntropy(y, pred)

	opt := model.SGDOptimizer{LR: 0.001}

	m.Compile(&opt, x, y, pred, loss, tensor.CategoricalAccuracy(y, pred))
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		m.Train(X, Y)
	}
}
