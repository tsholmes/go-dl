package model

import (
	"math"

	"github.com/tsholmes/go-dl/calc"
	"github.com/tsholmes/go-dl/tensor"
)

func NewModel() *Model {
	return &Model{}
}

type Model struct {
	weights    []tensor.Tensor
	weightVals []calc.NDArray

	// built on compile
	input tensor.Tensor
	yTrue tensor.Tensor
	yPred tensor.Tensor
	loss  tensor.Tensor

	weightGradients []tensor.Tensor

	trainEval   tensor.Evaluation
	testEval    tensor.Evaluation
	predictEval tensor.Evaluation

	// configurable
	gradientClip float64
}

func (m *Model) AddWeight(shape ...int) tensor.Tensor {
	sz := 1
	for _, s := range shape {
		sz *= s
	}
	init := math.Sqrt(1.0 / float64(sz))

	t := tensor.Input(shape...)
	v := calc.RandomUniform(-init, init, shape...)

	m.weights = append(m.weights, t)
	m.weightVals = append(m.weightVals, v)

	return t
}

func (m *Model) ClipGradients(clip float64) {
	m.gradientClip = clip
}

func (m *Model) Compile(input tensor.Tensor, yTrue tensor.Tensor, yPred tensor.Tensor, loss tensor.Tensor) {
	m.input = input
	m.yTrue = yTrue
	m.yPred = yPred
	m.loss = loss

	gradients := tensor.Gradients(loss)
	for _, w := range m.weights {
		m.weightGradients = append(m.weightGradients, gradients[w.ID()])
	}

	m.trainEval = tensor.MakeEvaluation(append([]tensor.Tensor{loss}, m.weightGradients...)...)
	m.testEval = tensor.MakeEvaluation(loss)
	m.predictEval = tensor.MakeEvaluation(yPred)
}

func (m *Model) weightProvisions() []tensor.ProvidedInput {
	res := make([]tensor.ProvidedInput, len(m.weights))
	for i := range res {
		res[i] = tensor.Provide(m.weights[i], m.weightVals[i])
	}
	return res
}

func (m *Model) updateWeights(grads []calc.NDArray, lr float64) {
	for i, w := range m.weightVals {
		g := grads[i]
		var meanAxes []int
		for j := range w.Shape() {
			if w.Shape()[j] == 1 && g.Shape()[j] > 1 {
				meanAxes = append(meanAxes, j)
			}
		}
		if len(meanAxes) > 0 {
			g = g.Mean(meanAxes...)
		}
		if m.gradientClip > 0 {
			g = g.Clip(-m.gradientClip, m.gradientClip)
		}
		m.weightVals[i] = w.Add(g.MulConstant(-lr))
	}
}

func (m *Model) Train(X calc.NDArray, Y calc.NDArray, lr float64) float64 {
	provisions := append([]tensor.ProvidedInput{
		tensor.Provide(m.input, X),
		tensor.Provide(m.yTrue, Y),
	}, m.weightProvisions()...)

	vals := m.trainEval.Evaluate(provisions...)

	loss := vals[0]

	m.updateWeights(vals[1:], lr)

	allAxes := make([]int, len(loss.Shape()))
	for i := range allAxes {
		allAxes[i] = i
	}

	return loss.Mean(allAxes...).Get(make([]int, len(loss.Shape())))
}

func (m *Model) Test(X calc.NDArray, Y calc.NDArray) float64 {
	provisions := append([]tensor.ProvidedInput{
		tensor.Provide(m.input, X),
		tensor.Provide(m.yTrue, Y),
	}, m.weightProvisions()...)
	loss := m.testEval.Evaluate(provisions...)[0]

	allAxes := make([]int, len(loss.Shape()))
	for i := range allAxes {
		allAxes[i] = i
	}

	return loss.Mean(allAxes...).Get(make([]int, len(loss.Shape())))
}
