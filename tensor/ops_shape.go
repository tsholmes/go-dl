package tensor

import (
	"github.com/tsholmes/go-dl/calc"
)

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
	delta := g.collect(t)

	offset := 0
	for _, a := range t.as {
		sz := a.Shape()[t.axis]

		g.push(a, Slice(delta, t.axis, offset, offset+sz))
		offset += sz
	}
}

func Slice(t Tensor, axis int, start int, end int) Tensor {
	return &SliceTensor{
		baseTensor: base(resize(t, axis, end-start), t),
		t:          t,
		axis:       axis,
		start:      start,
		end:        end,
	}
}

type SliceTensor struct {
	baseTensor
	t     Tensor
	axis  int
	start int
	end   int
}

func (t *SliceTensor) Visit(v TensorVisitor) { v.VisitSlice(t) }

func (e *evaluationVisitor) VisitSlice(t *SliceTensor) {
	v := e.value(t.t)
	e.values[t.ID()] = v.Slice(t.axis, t.start, t.end)
}

func (g *gradientVisitor) VisitSlice(t *SliceTensor) {
	delta := g.collect(t)

	g.push(t.t, Unslice(delta, t.axis, t.t.Shape()[t.axis], t.start))
}

func Unslice(t Tensor, axis int, size int, offset int) Tensor {
	return &UnsliceTensor{
		baseTensor: base(resize(t, axis, size), t),
		t:          t,
		axis:       axis,
		size:       size,
		offset:     offset,
	}
}

type UnsliceTensor struct {
	baseTensor
	t      Tensor
	axis   int
	size   int
	offset int
}

func (t *UnsliceTensor) Visit(v TensorVisitor) { v.VisitUnslice(t) }

func (e *evaluationVisitor) VisitUnslice(t *UnsliceTensor) {
	v := e.value(t.t)
	v2 := calc.Ones(t.Shape()...)
	v2.SetSlice(v, t.axis, t.offset)
	e.values[t.ID()] = v2
}

func (g *gradientVisitor) VisitUnslice(t *UnsliceTensor) {
	delta := g.collect(t)

	g.push(t.t, Slice(delta, t.axis, t.offset, t.offset+t.Shape()[t.axis]))
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

func Reshape(t Tensor, shape ...int) Tensor {
	return &ReshapeTensor{
		baseTensor: base(shape, t),
		t:          t,
	}
}

type ReshapeTensor struct {
	baseTensor
	t Tensor
}

func (t *ReshapeTensor) Visit(v TensorVisitor) { v.VisitReshape(t) }

func (e *evaluationVisitor) VisitReshape(t *ReshapeTensor) {
	v := e.value(t.t)
	e.values[t.ID()] = v.Reshape(t.Shape()...)
}

func (g *gradientVisitor) VisitReshape(t *ReshapeTensor) {
	delta := g.collect(t)

	g.push(t.t, Reshape(delta, t.t.Shape()...))
}

// flattens all axes after `axis` into 1
func Flatten(t Tensor, axis int) Tensor {
	shape := append([]int{}, t.Shape()...)
	for i := axis + 1; i < len(shape); i++ {
		shape[axis] *= shape[i]
	}
	shape = shape[:axis+1]
	return Reshape(t, shape...)
}
