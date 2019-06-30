package model

import (
	"github.com/tsholmes/go-dl/calc"
	"github.com/tsholmes/go-dl/tensor"
)

func NewModel() *Model {
	return &Model{
		weightInitializer: GlorotUniform,
		biasInitializer:   Zeros,
	}
}

type Model struct {
	weights    []tensor.Tensor
	weightVals []calc.NDArray

	weightInitializer Initializer
	biasInitializer   Initializer

	// built on compile
	input tensor.Tensor
	yTrue tensor.Tensor
	yPred tensor.Tensor
	loss  tensor.Tensor

	metrics []tensor.Tensor

	weightGradients []tensor.Tensor

	trainEval   tensor.Evaluation
	testEval    tensor.Evaluation
	predictEval tensor.Evaluation

	opt Optimizer
}

func (m *Model) AddWeight(shape ...int) tensor.Tensor {
	return m.AddWeightWith(m.weightInitializer, shape...)
}

func (m *Model) AddBias(shape ...int) tensor.Tensor {
	return m.AddWeightWith(m.biasInitializer, shape...)
}

func (m *Model) AddWeightWith(init Initializer, shape ...int) tensor.Tensor {
	t := tensor.Input(shape...)
	v := init(shape...)

	m.weights = append(m.weights, t)
	m.weightVals = append(m.weightVals, v)

	return t
}

func (m *Model) Compile(opt Optimizer, input tensor.Tensor, yTrue tensor.Tensor, yPred tensor.Tensor, loss tensor.Tensor, metrics ...tensor.Tensor) {
	m.input = input
	m.yTrue = yTrue
	m.yPred = yPred
	m.loss = loss
	m.metrics = metrics
	for i, mt := range m.metrics {
		m.metrics[i] = tensor.Mean(tensor.Flatten(mt, 0), 0)
	}

	gradients := tensor.Gradients(loss)
	for _, w := range m.weights {
		m.weightGradients = append(m.weightGradients, gradients[w.ID()])
	}

	trainTs := []tensor.Tensor{loss}
	trainTs = append(trainTs, metrics...)
	trainTs = append(trainTs, m.weightGradients...)

	m.trainEval = tensor.MakeEvaluation(trainTs...)
	m.testEval = tensor.MakeEvaluation(append([]tensor.Tensor{loss}, metrics...)...)
	m.predictEval = tensor.MakeEvaluation(yPred)

	m.opt = opt
}

func (m *Model) WeightProvisions() []tensor.ProvidedInput {
	res := make([]tensor.ProvidedInput, len(m.weights))
	for i := range res {
		res[i] = tensor.Provide(m.weights[i], m.weightVals[i])
	}
	return res
}

func (m *Model) Train(X calc.NDArray, Y calc.NDArray) (float64, []float64) {
	provisions := append([]tensor.ProvidedInput{
		tensor.Provide(m.input, X),
		tensor.Provide(m.yTrue, Y),
	}, m.WeightProvisions()...)

	vals := m.trainEval.Evaluate(provisions...)

	loss := vals[0]
	metrics := vals[1 : 1+len(m.metrics)]

	m.opt.UpdateWeights(m.weightVals, vals[1+len(m.metrics):])

	allAxes := make([]int, len(loss.Shape()))
	for i := range allAxes {
		allAxes[i] = i
	}

	mvals := make([]float64, len(metrics))
	for i := range metrics {
		// we flatten all these to 1
		mvals[i] = metrics[i].Get([]int{0})
	}

	return loss.Mean(allAxes...).Get(make([]int, len(loss.Shape()))), mvals
}

func (m *Model) DebugTrain() {
	m.trainEval.DebugDump()
}

func (m *Model) Test(X calc.NDArray, Y calc.NDArray) (float64, []float64) {
	provisions := append([]tensor.ProvidedInput{
		tensor.Provide(m.input, X),
		tensor.Provide(m.yTrue, Y),
	}, m.WeightProvisions()...)
	eval := m.testEval.Evaluate(provisions...)

	loss := eval[0]
	metrics := eval[1:]

	allAxes := make([]int, len(loss.Shape()))
	for i := range allAxes {
		allAxes[i] = i
	}

	mvals := make([]float64, len(metrics))
	for i := range metrics {
		// we flatten all these to 1
		mvals[i] = metrics[i].Get([]int{0})
	}

	return loss.Mean(allAxes...).Get(make([]int, len(loss.Shape()))), mvals
}

func (m *Model) Predict(X calc.NDArray) calc.NDArray {
	provisions := append([]tensor.ProvidedInput{
		tensor.Provide(m.input, X),
	}, m.WeightProvisions()...)
	return m.predictEval.Evaluate(provisions...)[0]
}

func (m *Model) WeightMags() []float64 {
	var res []float64
	for _, w := range m.weightVals {
		res = append(res, mag(w))
	}
	return res
}

func (m *Model) Weights() []calc.NDArray {
	return m.weightVals
}

func mag(w calc.NDArray) float64 {
	ax := make([]int, len(w.Shape()))
	for i := range ax {
		ax[i] = i
	}
	return w.PowConstant(2.0).Sum(ax...).PowConstant(0.5).Get(make([]int, len(w.Shape())))
}

func shapeEq(s1 []int, s2 []int) bool {
	for i := range s1 {
		if s1[i] != s2[i] {
			return false
		}
	}
	return true
}
