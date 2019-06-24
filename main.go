package main

import (
	"fmt"
	"math/rand"
	"reflect"
	"strings"
	"time"

	"github.com/tsholmes/go-dl/calc"
	"github.com/tsholmes/go-dl/dataset"
	"github.com/tsholmes/go-dl/model"
	"github.com/tsholmes/go-dl/tensor"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	batchSize := 8

	Xfull, Yfull, XTestFull, YTestFull := dataset.LoadMNIST()
	X := Xfull.Split(0, batchSize)
	Y := Yfull.Split(0, batchSize)
	Xtest := XTestFull.Split(0, batchSize)
	Ytest := YTestFull.Split(0, batchSize)

	X, Y, Xtest, Ytest = X[:60], Y[:60], Xtest[:10], Ytest[:10]

	l1Size := 8
	l2Size := 16
	l3Size := 32

	x := tensor.Input(batchSize, 28, 28, 1)
	y := tensor.Input(batchSize, 10)

	m := model.NewModel()

	var t tensor.Tensor = tensor.Reshape(x, batchSize, 28, 28, 1)

	t = model.Conv2D(m, t, 3, 3, l1Size)
	t = tensor.ReLU(t)
	t = model.MaxPooling2D(m, t, 2, 2)

	t = model.Conv2D(m, t, 3, 3, l2Size)
	t = tensor.ReLU(t)
	t = model.MaxPooling2D(m, t, 2, 2)

	t = model.Conv2D(m, t, 3, 3, l3Size)
	t = tensor.ReLU(t)

	t = tensor.Flatten(t, 1)

	t = model.Dense(m, t, 10, true)

	pred := tensor.Softmax(t)
	loss := tensor.CategoricalCrossEntropy(y, pred)

	m.Compile(x, y, pred, loss)
	// m.ClipGradients(1.0)
	// m.L2(0.01)

	const epochs = 10
	const lr = 1e-2

	for epoch := 0; epoch < epochs; epoch++ {
		workingLoss := 0.0
		for i := range X {
			bX, bY := X[i], Y[i]
			start := time.Now()
			workingLoss += m.Train(bX, bY, lr)
			end := time.Now()
			fmt.Printf("epoch %d/%d train batch %d/%d train loss %f duration %s\n", epoch, epochs, i, len(X), workingLoss/float64(i+1), end.Sub(start).String())
		}

		workingLoss = 0.0
		for i := range Xtest {
			bX, bY := Xtest[i], Ytest[i]
			workingLoss += m.Test(bX, bY)
			fmt.Printf("epoch %d/%d test batch %d/%d test loss %f\n", epoch, epochs, i, len(Xtest), workingLoss/float64(i+1))
		}
	}
}

func mag(w calc.NDArray) float64 {
	ax := make([]int, len(w.Shape()))
	for i := range ax {
		ax[i] = i
	}
	return w.PowConstant(2.0).Sum(ax...).PowConstant(0.5).Get(make([]int, len(w.Shape())))
}

func display(t tensor.Tensor) string {
	typ := reflect.TypeOf(t).String()
	var idStrs []string
	for _, it := range t.Inputs() {
		idStrs = append(idStrs, fmt.Sprintf("%d", it.ID()))
	}
	return fmt.Sprintf("%s(%s)%v", typ, strings.Join(idStrs, ","), t.Shape())
}
