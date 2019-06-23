package tensor

import (
	"fmt"

	"github.com/tsholmes/go-dl/calc"
)

func MakeEvaluation(outputs ...Tensor) Evaluation {
	evaluations := collectForward(outputs)
	return Evaluation{
		outputs:     outputs,
		evaluations: evaluations,
	}
}

type Evaluation struct {
	outputs []Tensor

	// Topoligically sorted list of tensors to evalute
	evaluations []Tensor
}

type ProvidedInput struct {
	t Tensor
	v calc.NDArray
}

func Provide(t Tensor, v calc.NDArray) ProvidedInput { return ProvidedInput{t, v} }

func (e *Evaluation) Evaluate(provisions ...ProvidedInput) []calc.NDArray {
	eval := &evaluationVisitor{
		values: map[int64]calc.NDArray{},
	}
	for _, p := range provisions {
		eval.values[p.t.ID()] = p.v
	}

	for _, t := range e.evaluations {
		t.Visit(eval)
	}

	outputs := make([]calc.NDArray, len(e.outputs))
	for i, output := range e.outputs {
		outputs[i] = eval.value(output)
	}
	return outputs
}

var _ TensorVisitor = &evaluationVisitor{}

type evaluationVisitor struct {
	values map[int64]calc.NDArray
}

func (e *evaluationVisitor) value(t Tensor) calc.NDArray {
	v, ok := e.values[t.ID()]
	if !ok {
		panic(fmt.Sprintf("missing value for tensor %d", t.ID()))
	}
	return v
}
