package tensor

import (
	"github.com/tsholmes/go-dl/calc"
)

func Add(as ...Tensor) Tensor {
	shape := elementWise(as...)
	return &AddTensor{
		baseTensor: base(shape, 2, as...),
		as:         as,
	}
}

type AddTensor struct {
	baseTensor
	as []Tensor
}

func (t *AddTensor) Visit(v TensorVisitor) { v.VisitAdd(t) }

func (e *evaluationVisitor) VisitAdd(t *AddTensor) {
	v, v2 := t.values[0], t.values[1]
	v.Fill(0.0)
	for _, a := range t.as {
		v2 = v.Add(e.value(a))
		v, v2 = v2, v
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
	return Add(a, Negate(b))
}

func Mul(as ...Tensor) Tensor {
	shape := elementWise(as...)
	return &MulTensor{
		baseTensor: base(shape, 2, as...),
		as:         as,
	}
}

type MulTensor struct {
	baseTensor
	as []Tensor
}

func (t *MulTensor) Visit(v TensorVisitor) { v.VisitMul(t) }

func (e *evaluationVisitor) VisitMul(t *MulTensor) {
	v, v2 := t.values[0], t.values[1]
	v.Fill(1.0)
	for _, a := range t.as {
		v2 = v.MulInto(e.value(a), v2)
		v, v2 = v2, v
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

func Negate(t Tensor) Tensor {
	return Mul(t, Constant(calc.Constant(-1., t.Shape()...)))
}

func Div(a Tensor, b Tensor) Tensor {
	return &DivTensor{
		baseTensor: base(elementWise(a, b), 0, a, b),
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

	g.push(t.a, Div(delta, t.b))
	g.push(t.b, // - d*a / b^2
		Negate(
			Div(
				Mul(delta, t.a),
				PowConstant(t.b, 2),
			),
		),
	)
}

func PowConstant(t Tensor, p float64) Tensor {
	return &PowConstantTensor{
		baseTensor: base(t.Shape(), 0, t),
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

func MatMul(a Tensor, b Tensor, a1 int, a2 int) Tensor {
	return &MatMulTensor{
		baseTensor: base(matMul(a, b, a1, a2), 1, a, b),
		a:          a,
		b:          b,
		a1:         a1,
		a2:         a2,
	}
}

type MatMulTensor struct {
	baseTensor
	a  Tensor
	b  Tensor
	a1 int
	a2 int
}

func (t *MatMulTensor) Visit(v TensorVisitor) { v.VisitMatMul(t) }

func (e *evaluationVisitor) VisitMatMul(t *MatMulTensor) {
	a := e.value(t.a)
	b := e.value(t.b)

	e.values[t.ID()] = a.MatMulInto(b, t.a1, t.a2, t.values[0])
}

func (g *gradientVisitor) VisitMatMul(t *MatMulTensor) {
	delta := g.collect(t)

	g.push(t.a, MatMul(delta, Transpose(t.b, t.a1, t.a2), t.a1, t.a2))
	g.push(t.b, MatMul(Transpose(t.a, t.a1, t.a2), delta, t.a1, t.a2))
}

func Log(t Tensor) Tensor {
	return &LogTensor{
		baseTensor: base(t.Shape(), 0, t),
		t:          t,
	}
}

type LogTensor struct {
	baseTensor
	t Tensor
}

func (t *LogTensor) Visit(v TensorVisitor) { v.VisitLog(t) }

func (e *evaluationVisitor) VisitLog(t *LogTensor) {
	v := e.value(t.t)
	e.values[t.ID()] = v.Log()
}

func (g *gradientVisitor) VisitLog(t *LogTensor) {
	delta := g.collect(t)

	g.push(t.t, Div(
		delta,
		t.t,
	))
}

func Exp(t Tensor) Tensor {
	return &ExpTensor{
		baseTensor: base(t.Shape(), 0, t),
		t:          t,
	}
}

type ExpTensor struct {
	baseTensor
	t Tensor
}

func (t *ExpTensor) Visit(v TensorVisitor) { v.VisitExp(t) }

func (e *evaluationVisitor) VisitExp(t *ExpTensor) {
	v := e.value(t.t)
	e.values[t.ID()] = v.Exp()
}

func (g *gradientVisitor) VisitExp(t *ExpTensor) {
	delta := g.collect(t)

	g.push(t.t, Mul(
		delta,
		t,
	))
}

// k must be (h, w, tFilters, outFilters)
func Conv2D(t Tensor, k Tensor, hAxis int, wAxis int, fAxis int) Tensor {
	kh, kw := k.Shape()[0], k.Shape()[1]
	return &Conv2DTensor{
		baseTensor: base(conv2d(t, k, hAxis, wAxis, fAxis), 1, t, k),
		t:          t,
		k:          k,
		hAxis:      hAxis,
		wAxis:      wAxis,
		fAxis:      fAxis,
		padH:       kh - 1,
		padW:       kw - 1,
	}
}

type Conv2DTensor struct {
	baseTensor
	t     Tensor
	k     Tensor
	hAxis int
	wAxis int
	fAxis int
	padH  int
	padW  int
}

func (t *Conv2DTensor) Visit(v TensorVisitor) { v.VisitConv2D(t) }

func (e *evaluationVisitor) VisitConv2D(t *Conv2DTensor) {
	i := e.value(t.t)
	k := e.value(t.k)

	v := i.Conv2DInto(k, t.hAxis, t.wAxis, t.fAxis, t.values[0])

	e.values[t.ID()] = v
}

func (g *gradientVisitor) VisitConv2D(t *Conv2DTensor) {
	delta := g.collect(t)

	kGrad := InverseConv2D(t.t, delta, t.hAxis, t.wAxis, t.fAxis)

	delta = Unslice(delta, t.hAxis, t.Shape()[t.hAxis]+t.padH*2, t.padH)
	delta = Unslice(delta, t.wAxis, t.Shape()[t.wAxis]+t.padW*2, t.padW)

	tGrad := Conv2D(delta, Transpose(Reverse(t.k, 0, 1), 2, 3), t.hAxis, t.wAxis, t.fAxis)

	g.push(t.t, tGrad)
	g.push(t.k, kGrad)
}

func InverseConv2D(t Tensor, g Tensor, hAxis int, wAxis int, fAxis int) Tensor {
	return &InverseConv2DTensor{
		baseTensor: base(inverseConv2d(t, g, hAxis, wAxis, fAxis), 1, t, g),
		t:          t,
		g:          g,
		hAxis:      hAxis,
		wAxis:      wAxis,
		fAxis:      fAxis,
	}
}

type InverseConv2DTensor struct {
	baseTensor
	t     Tensor
	g     Tensor
	hAxis int
	wAxis int
	fAxis int
}

func (t *InverseConv2DTensor) Visit(v TensorVisitor) { v.VisitInverseConv2D(t) }

func (e *evaluationVisitor) VisitInverseConv2D(t *InverseConv2DTensor) {
	i := e.value(t.t)
	g := e.value(t.g)

	v := i.InverseConv2DInto(g, t.hAxis, t.wAxis, t.fAxis, t.values[0])

	e.values[t.ID()] = v
}

func (g *gradientVisitor) VisitInverseConv2D(t *InverseConv2DTensor) {
	panic("InverseConv2D is not differentiable")
}
