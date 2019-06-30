package model

import "github.com/tsholmes/go-dl/calc"

type Optimizer interface {
	UpdateWeights(weights []calc.NDArray, grads []calc.NDArray)
}

type SGDOptimizer struct {
	LR float64
}

func (o *SGDOptimizer) UpdateWeights(weights []calc.NDArray, grads []calc.NDArray) {
	for i := range weights {
		w, g := weights[i], grads[i]
		weights[i] = w.AddInto(g.MulConstant(-o.LR), w)
	}
}

type SGDMomentumOptimizer struct {
	LR       float64
	Momentum float64
	Nesterov bool

	moments []calc.NDArray
}

func (o *SGDMomentumOptimizer) UpdateWeights(weights []calc.NDArray, grads []calc.NDArray) {
	if len(o.moments) == 0 {
		o.moments = make([]calc.NDArray, len(weights))
		for i, w := range weights {
			o.moments[i] = calc.Zeros(w.Shape()...)
		}
	}
	for i := range weights {
		w, g := weights[i], grads[i]

		o.moments[i] = o.moments[i].MulConstant(o.Momentum).Add(g.MulConstant(-o.LR))
		if o.Nesterov {
			weights[i] = w.AddInto(o.moments[i].MulConstant(o.Momentum).Add(g.MulConstant(-o.LR)), w)
		} else {
			weights[i] = w.AddInto(o.moments[i], w)
		}
	}
}
