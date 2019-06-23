package tensor

import "github.com/tsholmes/go-dl/calc"

func Sum(t Tensor, axes ...int) Tensor {
	return &SumTensor{
		baseTensor: base(aggr(t, axes...), t),
		t:          t,
		axes:       axes,
	}
}

type SumTensor struct {
	baseTensor
	t    Tensor
	axes []int
}

func (t *SumTensor) Visit(v TensorVisitor) { v.VisitSum(t) }

func (e *evaluationVisitor) VisitSum(t *SumTensor) {
	v := e.value(t.t)
	e.values[t.ID()] = v.Sum(t.axes...)
}

func (g *gradientVisitor) VisitSum(t *SumTensor) {
	delta := g.collect(t)

	// This will broadcast up to all the alements
	g.push(t.t, delta)
}

func Mean(t Tensor, axes ...int) Tensor {
	div := 1
	for _, i := range axes {
		div *= t.Shape()[i]
	}

	s := Sum(t, axes...)
	return Mul(s, Constant(calc.Constant(1./float64(div), s.Shape()...)))
}
