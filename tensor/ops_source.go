package tensor

import "github.com/tsholmes/go-dl/calc"

func Input(shape ...int) Tensor {
	return &InputTensor{
		baseTensor: base(shape, 0),
		shape:      shape,
	}
}

type InputTensor struct {
	baseTensor
	shape []int
}

func (t *InputTensor) Visit(v TensorVisitor) { v.VisitInput(t) }

func (e *evaluationVisitor) VisitInput(t *InputTensor) {
	// Just assert that it was passed
	e.value(t)
}

func (g *gradientVisitor) VisitInput(t *InputTensor) {
	g.collect(t)
}

func Constant(value calc.NDArray) Tensor {
	return &ConstantTensor{
		baseTensor: base(value.Shape(), 0),
		value:      value,
	}
}

type ConstantTensor struct {
	baseTensor
	value calc.NDArray
}

func (t *ConstantTensor) Visit(v TensorVisitor) { v.VisitConstant(t) }

func (e *evaluationVisitor) VisitConstant(t *ConstantTensor) {
	// Just assert that it was passed
	e.values[t.ID()] = t.value
}

func (g *gradientVisitor) VisitConstant(t *ConstantTensor) {
	// there's really no point in even bothering with this
	g.collect(t)
}

func Ones(shape ...int) Tensor {
	return Constant(calc.Ones(shape...))
}
