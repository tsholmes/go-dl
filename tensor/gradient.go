package tensor

import "github.com/tsholmes/go-dl/calc"

// Get a map of original tensor ID -> gradient tensor given an output tensor
func Gradients(outputs ...Tensor) map[int64]Tensor {
	gv := &gradientVisitor{
		partialGradients: map[int64][]Tensor{},
		gradients:        map[int64]Tensor{},
	}

	for _, t := range outputs {
		gv.partialGradients[t.ID()] = []Tensor{Constant(calc.Ones(t.Shape()...))}
	}

	for _, t := range collectBackward(outputs) {
		t.Visit(gv)
	}

	return gv.gradients
}

var _ TensorVisitor = &gradientVisitor{}

type gradientVisitor struct {
	partialGradients map[int64][]Tensor

	gradients map[int64]Tensor
}

func (g *gradientVisitor) push(tensor Tensor, gradient Tensor) {
	g.partialGradients[tensor.ID()] = append(g.partialGradients[tensor.ID()], gradient)
}

func (g *gradientVisitor) collect(tensor Tensor) Tensor {
	partials := g.partialGradients[tensor.ID()]
	var gradient Tensor
	if len(partials) == 0 {
		gradient = Constant(calc.Zeros(tensor.Shape()...))
	} else if len(partials) == 1 {
		gradient = partials[0]
	} else {
		gradient = Add(partials...)
	}
	g.gradients[tensor.ID()] = gradient
	return gradient
}

func (g *gradientVisitor) VisitInput(t *InputTensor) {
	g.collect(t)
}

func (g *gradientVisitor) VisitConstant(t *ConstantTensor) {
	// there's really no point in even bothering with thi
	g.collect(t)
}

func (g *gradientVisitor) VisitAdd(t *AddTensor) {
	delta := g.collect(t)

	for _, a := range t.as {
		g.push(a, delta)
	}
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

func (g *gradientVisitor) VisitAbs(t *AbsTensor) {
	delta := g.collect(t)

	g.push(t.t, Mul(delta, Sign(t.t)))
}

func (g *gradientVisitor) VisitSign(t *SignTensor) {
	panic("Sign is not differentiable")
}

func (g *gradientVisitor) VisitConcat(t *ConcatTensor) {
	// TODO: needs slice tensor
}
