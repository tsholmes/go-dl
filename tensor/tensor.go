package tensor

import "github.com/tsholmes/go-dl/calc"

type Tensor interface {
	ID() int64
	Shape() []int
	Inputs() []Tensor

	Visit(v TensorVisitor)
}

type TensorVisitor interface {
	VisitInput(t *InputTensor)
	VisitConstant(t *ConstantTensor)
	VisitAdd(t *AddTensor)
	VisitMul(t *MulTensor)
	VisitDiv(t *DivTensor)
	VisitAbs(t *AbsTensor)
	VisitSign(t *SignTensor)
	VisitPowConstant(t *PowConstantTensor)
	VisitConcat(t *ConcatTensor)
	VisitSum(t *SumTensor)
}

var nextID int64

type baseTensor struct {
	id     int64
	shape  []int
	inputs []Tensor
}

func (b *baseTensor) ID() int64 {
	return b.id
}

func (b *baseTensor) Shape() []int {
	return b.shape
}

func (b *baseTensor) Inputs() []Tensor {
	return b.inputs
}

func base(shape []int, inputs ...Tensor) baseTensor {
	// TODO: lock around nextID
	id := nextID
	nextID++
	return baseTensor{
		id:     id,
		shape:  shape,
		inputs: inputs,
	}
}

type InputTensor struct {
	baseTensor
	shape []int
}

func (t *InputTensor) Visit(v TensorVisitor) { v.VisitInput(t) }

func Input(shape ...int) Tensor {
	return &InputTensor{
		baseTensor: base(shape),
		shape:      shape,
	}
}

type ConstantTensor struct {
	baseTensor
	value calc.NDArray
}

func (t *ConstantTensor) Visit(v TensorVisitor) { v.VisitConstant(t) }

func Constant(value calc.NDArray) Tensor {
	return &ConstantTensor{
		baseTensor: base(value.Shape()),
		value:      value,
	}
}

type AddTensor struct {
	baseTensor
	as []Tensor
}

func (t *AddTensor) Visit(v TensorVisitor) { v.VisitAdd(t) }

func Add(as ...Tensor) Tensor {
	return &AddTensor{
		baseTensor: base(elementWise(as...), as...),
		as:         as,
	}
}

func Sub(a Tensor, b Tensor) Tensor {
	return Add(a, Mul(b, Constant(calc.Constant(-1, b.Shape()...))))
}

type MulTensor struct {
	baseTensor
	as []Tensor
}

func (t *MulTensor) Visit(v TensorVisitor) { v.VisitMul(t) }

func Mul(as ...Tensor) Tensor {
	return &MulTensor{
		baseTensor: base(elementWise(as...), as...),
		as:         as,
	}
}

type DivTensor struct {
	baseTensor
	a Tensor
	b Tensor
}

func (t *DivTensor) Visit(v TensorVisitor) { v.VisitDiv(t) }

func Div(a Tensor, b Tensor) Tensor {
	return &DivTensor{
		baseTensor: base(elementWise(a, b), a, b),
		a:          a,
		b:          b,
	}
}

type AbsTensor struct {
	baseTensor
	t Tensor
}

func (t *AbsTensor) Visit(v TensorVisitor) { v.VisitAbs(t) }

func Abs(t Tensor) Tensor {
	return &AbsTensor{
		baseTensor: base(t.Shape(), t),
		t:          t,
	}
}

type SignTensor struct {
	baseTensor
	t Tensor
}

func (t *SignTensor) Visit(v TensorVisitor) { v.VisitSign(t) }

func Sign(t Tensor) Tensor {
	return &SignTensor{
		baseTensor: base(t.Shape(), t),
		t:          t,
	}
}

type PowConstantTensor struct {
	baseTensor
	t Tensor
	p float64
}

func (t *PowConstantTensor) Visit(v TensorVisitor) { v.VisitPowConstant(t) }

func PowConstant(t Tensor, p float64) Tensor {
	return &PowConstantTensor{
		baseTensor: base(t.Shape(), t),
		t:          t,
		p:          p,
	}
}

type ConcatTensor struct {
	baseTensor
	axis int
	as   []Tensor
}

func (t *ConcatTensor) Visit(v TensorVisitor) { v.VisitConcat(t) }

func Concat(axis int, as ...Tensor) Tensor {
	return &ConcatTensor{
		baseTensor: base(concat(axis, as...), as...),
		axis:       axis,
		as:         as,
	}
}

type SumTensor struct {
	baseTensor
	t    Tensor
	axes []int
}

func (t *SumTensor) Visit(v TensorVisitor) { v.VisitSum(t) }

func Sum(t Tensor, axes ...int) Tensor {
	return &SumTensor{
		baseTensor: base(aggr(t, axes...), t),
		t:          t,
		axes:       axes,
	}
}

func Mean(t Tensor, axes ...int) Tensor {
	div := 1
	for _, i := range axes {
		div *= t.Shape()[i]
	}

	s := Sum(t, axes...)
	return Mul(s, Constant(calc.Constant(1./float64(div), s.Shape()...)))
}
