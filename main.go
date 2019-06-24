package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/tsholmes/go-dl/dataset"
	"github.com/tsholmes/go-dl/model"
	"github.com/tsholmes/go-dl/tensor"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	batchSize := 100

	Xfull, Yfull, XTestFull, YTestFull := dataset.LoadMNIST()
	X := Xfull.Split(0, batchSize)
	Y := Yfull.Split(0, batchSize)
	Xtest := XTestFull.Split(0, batchSize)
	Ytest := YTestFull.Split(0, batchSize)

	X, Y, Xtest, Ytest = X[:60], Y[:60], Xtest[:10], Ytest[:10]

	l1Size := 4
	l2Size := 8

	x := tensor.Input(batchSize, 28, 28, 1)
	y := tensor.Input(batchSize, 10)

	m := model.NewModel()

	var t tensor.Tensor = tensor.Reshape(x, batchSize, 28, 28, 1)

	t = model.Conv2D(m, t, 3, 3, l1Size) // B * 26 * 26 * l1
	t = tensor.ReLU(t)
	t = model.Conv2D(m, t, 3, 3, l2Size) // B * 24 * 24 * l2
	t = tensor.ReLU(t)

	t = tensor.Flatten(t, 1) // B * (24*24*l2)

	t = model.Dense(m, t, 10)

	pred := tensor.Softmax(t)
	loss := tensor.Mean(tensor.CategoricalCrossEntropy(y, pred), 0)

	m.Compile(x, y, pred, loss)
	m.ClipGradients(0.1)

	const epochs = 100
	const lr = 1e-3

	for epoch := 0; epoch < epochs; epoch++ {
		workingLoss := 0.0
		for i := range X {
			bX, bY := X[i], Y[i]
			workingLoss += m.Train(bX, bY, lr)
			fmt.Printf("epoch %d/%d train batch %d/%d train loss %f\n", epoch, epochs, i, len(X), workingLoss/float64(i+1))
		}

		workingLoss = 0.0
		for i := range Xtest {
			bX, bY := Xtest[i], Ytest[i]
			workingLoss += m.Test(bX, bY)
			fmt.Printf("epoch %d/%d test batch %d/%d test loss %f\n", epoch, epochs, i, len(Xtest), workingLoss/float64(i+1))
		}
	}
}
