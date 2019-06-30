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

	batchSize := 100

	X, Y, Xtest, Ytest := dataset.LoadMNIST()

	trainBatches := 60
	testBatches := 10
	trainSamples := batchSize * trainBatches
	testSamples := batchSize * testBatches

	X, Y, Xtest, Ytest = X.SliceRoot(0, trainSamples), Y.SliceRoot(0, trainSamples), Xtest.SliceRoot(0, testSamples), Ytest.SliceRoot(0, testSamples)

	l1Size := 16
	l2Size := 32
	l3Size := 64

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

	const lr = 1e-3
	opt := model.SGDMomentumOptimizer{LR: lr, Momentum: 0.1, Nesterov: true}

	m.Compile(&opt, x, y, pred, loss, tensor.CategoricalAccuracy(y, pred))

	const epochs = 10

	index := make([]int, trainSamples)
	for i := range index {
		index[i] = i
	}

	for epoch := 0; epoch < epochs; epoch++ {
		rand.Shuffle(len(index), func(i, j int) { index[i], index[j] = index[j], index[i] })
		X = X.ReindexRoot(index)
		Y = Y.ReindexRoot(index)

		workingLoss := 0.0
		workingAcc := 0.0
		for i := 0; i < trainBatches; i++ {
			bX, bY := X.SliceRoot(i*batchSize, batchSize), Y.SliceRoot(i*batchSize, batchSize)
			start := time.Now()
			bloss, bmet := m.Train(bX, bY)
			workingLoss += bloss
			workingAcc += bmet[0]
			end := time.Now()
			fmt.Printf("epoch %d/%d train batch %d/%d train loss %f train acc %f duration %s\n", epoch, epochs, i, trainBatches, workingLoss/float64(i+1), workingAcc/float64(i+1), end.Sub(start).String())
		}

		workingLoss = 0.0
		workingAcc = 0.0
		for i := 0; i < testBatches; i++ {
			bX, bY := Xtest.SliceRoot(i*batchSize, batchSize), Ytest.SliceRoot(i*batchSize, batchSize)
			bloss, bmet := m.Test(bX, bY)
			workingLoss += bloss
			workingAcc += bmet[0]
			fmt.Printf("epoch %d/%d test batch %d/%d test loss %f test acc %f\n", epoch, epochs, i, testBatches, workingLoss/float64(i+1), workingAcc/float64(i+1))
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
