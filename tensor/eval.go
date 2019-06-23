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

func (e *evaluationVisitor) VisitInput(t *InputTensor) {
	// Just assert that it was passed
	e.value(t)
}

func (e *evaluationVisitor) VisitConstant(t *ConstantTensor) {
	// Just assert that it was passed
	e.values[t.ID()] = t.value
}

func (e *evaluationVisitor) VisitAdd(t *AddTensor) {
	v := calc.Zeros(t.Shape()...)
	for _, a := range t.as {
		v = v.Add(e.value(a))
	}
	e.values[t.ID()] = v
}

func (e *evaluationVisitor) VisitMul(t *MulTensor) {
	v := calc.Ones(t.Shape()...)
	for _, a := range t.as {
		v = v.Mul(e.value(a))
	}
	e.values[t.ID()] = v
}

func (e *evaluationVisitor) VisitDiv(t *DivTensor) {
	a := e.value(t.a)
	b := e.value(t.b)
	v := a.Div(b)
	e.values[t.ID()] = v
}

func (e *evaluationVisitor) VisitAbs(t *AbsTensor) {
	v := e.value(t.t)
	v = v.Mul(v.Sign())
	e.values[t.ID()] = v
}

func (e *evaluationVisitor) VisitSign(t *SignTensor) {
	v := e.value(t.t)
	e.values[t.ID()] = v.Sign()
}

func (e *evaluationVisitor) VisitConcat(t *ConcatTensor) {
	v := e.value(t.as[0])
	for _, a := range t.as[1:] {
		v = v.Concat(e.value(a), t.axis)
	}
	e.values[t.ID()] = v
}
