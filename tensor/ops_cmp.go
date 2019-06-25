package tensor

import "github.com/tsholmes/go-dl/calc"

func Abs(t Tensor) Tensor {
	return &AbsTensor{
		baseTensor: base(t.Shape(), 0, t),
		t:          t,
	}
}

type AbsTensor struct {
	baseTensor
	t Tensor
}

func (t *AbsTensor) Visit(v TensorVisitor) { v.VisitAbs(t) }

func (e *evaluationVisitor) VisitAbs(t *AbsTensor) {
	v := e.value(t.t)
	v = v.Mul(v.Sign())
	e.values[t.ID()] = v
}

func (g *gradientVisitor) VisitAbs(t *AbsTensor) {
	delta := g.collect(t)

	g.push(t.t, Mul(delta, Sign(t.t)))
}

func Sign(t Tensor) Tensor {
	return &SignTensor{
		baseTensor: base(t.Shape(), 0, t),
		t:          t,
	}
}

type SignTensor struct {
	baseTensor
	t Tensor
}

func (t *SignTensor) Visit(v TensorVisitor) { v.VisitSign(t) }

func (e *evaluationVisitor) VisitSign(t *SignTensor) {
	v := e.value(t.t)
	e.values[t.ID()] = v.Sign()
}

func (g *gradientVisitor) VisitSign(t *SignTensor) {
	panic("Sign is not differentiable")
}

func Greater(a Tensor, b Tensor) Tensor {
	return &GreaterTensor{
		baseTensor: base(elementWise(a, b), 0, a, b),
		a:          a,
		b:          b,
	}
}

type GreaterTensor struct {
	baseTensor
	a Tensor
	b Tensor
}

func (t *GreaterTensor) Visit(v TensorVisitor) { v.VisitGreater(t) }

func (e *evaluationVisitor) VisitGreater(t *GreaterTensor) {
	a := e.value(t.a)
	b := e.value(t.b)
	e.values[t.ID()] = a.Greater(b)
}

func (g *gradientVisitor) VisitGreater(t *GreaterTensor) {
	panic("Greater is not differentiable")
}

func Equal(a Tensor, b Tensor) Tensor {
	return &EqualTensor{
		baseTensor: base(elementWise(a, b), 0, a, b),
		a:          a,
		b:          b,
	}
}

type EqualTensor struct {
	baseTensor
	a Tensor
	b Tensor
}

func (t *EqualTensor) Visit(v TensorVisitor) { v.VisitEqual(t) }

func (e *evaluationVisitor) VisitEqual(t *EqualTensor) {
	a := e.value(t.a)
	b := e.value(t.b)
	e.values[t.ID()] = a.Equal(b)
}

func (g *gradientVisitor) VisitEqual(t *EqualTensor) {
	panic("Equal is not differentiable")
}

func ReLU(t Tensor) Tensor {
	return &ReLUTensor{
		baseTensor: base(t.Shape(), 0, t),
		t:          t,
	}
}

type ReLUTensor struct {
	baseTensor
	t Tensor
}

func (t *ReLUTensor) Visit(v TensorVisitor) { v.VisitReLU(t) }

func (e *evaluationVisitor) VisitReLU(t *ReLUTensor) {
	v := e.value(t.t)
	g := v.Greater(calc.Zeros(v.Shape()...))
	e.values[t.ID()] = v.Mul(g)
}

func (g *gradientVisitor) VisitReLU(t *ReLUTensor) {
	delta := g.collect(t)

	g.push(t.t, Mul(delta, Greater(t.t, Constant(calc.Zeros(t.Shape()...)))))
}
