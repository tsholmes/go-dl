package tensor

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

func EqualMask(t Tensor, a Tensor, b Tensor) Tensor {
	return &EqualMaskTensor{
		baseTensor: base(t.Shape(), 0, t, a, b),
		t:          t,
		a:          a,
		b:          b,
	}
}

type EqualMaskTensor struct {
	baseTensor
	t Tensor
	a Tensor
	b Tensor
}

func (t *EqualMaskTensor) Visit(v TensorVisitor) { v.VisitEqualMask(t) }

func (e *evaluationVisitor) VisitEqualMask(t *EqualMaskTensor) {
	v := e.value(t.t)
	a := e.value(t.a)
	b := e.value(t.b)
	e.values[t.ID()] = v.EqualMask(a, b)
}

func (g *gradientVisitor) VisitEqualMask(t *EqualMaskTensor) {
	// at least not until i feel like it
	panic("EqualMask is not differentiable")
}

func ReLU(t Tensor) Tensor {
	return &ReLUTensor{
		baseTensor: base(t.Shape(), 1, t),
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
	o := t.values[0]
	e.values[t.ID()] = v.ReLUInto(o)
}

func (g *gradientVisitor) VisitReLU(t *ReLUTensor) {
	delta := g.collect(t)

	g.push(t.t, ReLUMask(delta, t.t))
}

// Zeroes out all values in t where the corresponding value in m is negative
func ReLUMask(t Tensor, m Tensor) Tensor {
	return &ReLUMaskTensor{
		baseTensor: base(t.Shape(), 1, t, m),
		t:          t,
		m:          m,
	}
}

type ReLUMaskTensor struct {
	baseTensor
	t Tensor
	m Tensor
}

func (t *ReLUMaskTensor) Visit(v TensorVisitor) { v.VisitReLUMask(t) }

func (e *evaluationVisitor) VisitReLUMask(t *ReLUMaskTensor) {
	v := e.value(t.t)
	mv := e.value(t.m)
	o := t.values[0]
	e.values[t.ID()] = v.ReLUMaskInto(mv, o)
}

func (g *gradientVisitor) VisitReLUMask(t *ReLUMaskTensor) {
	// i mean the gradient to t.t is trivial, but its not needed
	panic("ReLUMask is not differentiable")
}
