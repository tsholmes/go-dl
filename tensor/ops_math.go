package tensor

import "github.com/tsholmes/go-dl/calc"

func Add(as ...Tensor) Tensor {
	return &AddTensor{
		baseTensor: base(elementWise(as...), as...),
		as:         as,
	}
}

type AddTensor struct {
	baseTensor
	as []Tensor
}

func (t *AddTensor) Visit(v TensorVisitor) { v.VisitAdd(t) }

func (e *evaluationVisitor) VisitAdd(t *AddTensor) {
	v := calc.Zeros(t.Shape()...)
	for _, a := range t.as {
		v = v.Add(e.value(a))
	}
	e.values[t.ID()] = v
}

func (g *gradientVisitor) VisitAdd(t *AddTensor) {
	delta := g.collect(t)

	for _, a := range t.as {
		g.push(a, delta)
	}
}

func Sub(a Tensor, b Tensor) Tensor {
	return Add(a, Mul(b, Constant(calc.Constant(-1, b.Shape()...))))
}

func Mul(as ...Tensor) Tensor {
	return &MulTensor{
		baseTensor: base(elementWise(as...), as...),
		as:         as,
	}
}

type MulTensor struct {
	baseTensor
	as []Tensor
}

func (t *MulTensor) Visit(v TensorVisitor) { v.VisitMul(t) }

func (e *evaluationVisitor) VisitMul(t *MulTensor) {
	v := calc.Ones(t.Shape()...)
	for _, a := range t.as {
		v = v.Mul(e.value(a))
	}
	e.values[t.ID()] = v
}

func (g *gradientVisitor) VisitMul(t *MulTensor) {
	delta := g.collect(t)

	for i, a := range t.as {
		as := make([]Tensor, len(t.as))
		copy(as, t.as)
		as[i] = delta
		g.push(a, Mul(as...))
	}
}

func Div(a Tensor, b Tensor) Tensor {
	return &DivTensor{
		baseTensor: base(elementWise(a, b), a, b),
		a:          a,
		b:          b,
	}
}

type DivTensor struct {
	baseTensor
	a Tensor
	b Tensor
}

func (t *DivTensor) Visit(v TensorVisitor) { v.VisitDiv(t) }

func (e *evaluationVisitor) VisitDiv(t *DivTensor) {
	a := e.value(t.a)
	b := e.value(t.b)
	v := a.Div(b)
	e.values[t.ID()] = v
}

func (g *gradientVisitor) VisitDiv(t *DivTensor) {
	delta := g.collect(t)

	g.push(t.a, Mul(delta, t.b))
	g.push(t.b, // - d*a / b^2
		Mul(
			Constant(calc.Constant(-1, t.Shape()...)),
			Div(
				Mul(delta, t.a),
				Mul(t.b, t.b),
			),
		),
	)
}

func PowConstant(t Tensor, p float64) Tensor {
	return &PowConstantTensor{
		baseTensor: base(t.Shape(), t),
		t:          t,
		p:          p,
	}
}

type PowConstantTensor struct {
	baseTensor
	t Tensor
	p float64
}

func (t *PowConstantTensor) Visit(v TensorVisitor) { v.VisitPowConstant(t) }

func (e *evaluationVisitor) VisitPowConstant(t *PowConstantTensor) {
	v := e.value(t.t)
	e.values[t.ID()] = v.PowConstant(t.p)
}

func (g *gradientVisitor) VisitPowConstant(t *PowConstantTensor) {
	delta := g.collect(t)

	g.push(t.t, Mul(
		delta,
		Constant(calc.Constant(t.p, t.Shape()...)),
		PowConstant(t.t, t.p-1.),
	))
}
