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

func Transpose(t Tensor, a1 int, a2 int) Tensor {
	return &TransposeTensor{
		baseTensor: base(transpose(t, a1, a2), t),
		t:          t,
		a1:         a1,
		a2:         a2,
	}
}

type TransposeTensor struct {
	baseTensor
	t  Tensor
	a1 int
	a2 int
}

func (t *TransposeTensor) Visit(v TensorVisitor) { v.VisitTranspose(t) }

func (e *evaluationVisitor) VisitTranspose(t *TransposeTensor) {
	v := e.value(t.t)
	e.values[t.ID()] = v.Transpose(t.a1, t.a2)
}

func (g *gradientVisitor) VisitTranspose(t *TransposeTensor) {
	delta := g.collect(t)

	g.push(t.t, Transpose(delta, t.a1, t.a2))
}
