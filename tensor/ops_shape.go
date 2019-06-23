package tensor

func Concat(axis int, as ...Tensor) Tensor {
	return &ConcatTensor{
		baseTensor: base(concat(axis, as...), as...),
		axis:       axis,
		as:         as,
	}
}

type ConcatTensor struct {
	baseTensor
	axis int
	as   []Tensor
}

func (t *ConcatTensor) Visit(v TensorVisitor) { v.VisitConcat(t) }

func (e *evaluationVisitor) VisitConcat(t *ConcatTensor) {
	v := e.value(t.as[0])
	for _, a := range t.as[1:] {
		v = v.Concat(e.value(a), t.axis)
	}
	e.values[t.ID()] = v
}

func (g *gradientVisitor) VisitConcat(t *ConcatTensor) {
	// TODO: needs slice tensor
}
