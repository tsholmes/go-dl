package tensor

import (
	"fmt"
	"reflect"
	"strings"
	"time"

	"github.com/tsholmes/go-dl/calc"
)

func MakeEvaluation(outputs ...Tensor) Evaluation {
	evaluations := CollectForward(outputs)
	timings := make([]time.Duration, len(evaluations))
	return Evaluation{
		outputs:     outputs,
		evaluations: evaluations,
		timings:     timings,
	}
}

type Evaluation struct {
	outputs []Tensor

	// Topoligically sorted list of tensors to evalute
	evaluations []Tensor

	timings []time.Duration
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

	for i, t := range e.evaluations {
		start := time.Now()
		t.Visit(eval)
		end := time.Now()
		e.timings[i] += end.Sub(start)
	}

	outputs := make([]calc.NDArray, len(e.outputs))
	for i, output := range e.outputs {
		outputs[i] = eval.value(output)
	}
	return outputs
}

func (e *Evaluation) DebugDump() {
	for i := range e.evaluations {
		t, d := e.evaluations[i], e.timings[i]
		fmt.Println(d, t.ID(), display(t))
	}
}

func display(t Tensor) string {
	typ := reflect.TypeOf(t).String()
	var idStrs []string
	for _, it := range t.Inputs() {
		idStrs = append(idStrs, fmt.Sprintf("(%d %v)", it.ID(), it.Shape()))
	}
	return fmt.Sprintf("%s(%s)%v", typ, strings.Join(idStrs, ","), t.Shape())
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
