package tensor

type Tensor interface {
	ID() int64
	Shape() []int
	Inputs() []Tensor

	Visit(v TensorVisitor)
}

type TensorVisitor interface {
	VisitInput(t *InputTensor)
	VisitAdd(t *AddTensor)
	VisitMulConstant(t *MulConstantTensor)
	VisitMul(t *MulTensor)
	VisitDiv(t *DivTensor)
	VisitConcat(t *ConcatTensor)
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

func Input(shape []int) Tensor {
	return &InputTensor{
		baseTensor: base(shape),
		shape:      shape,
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

type MulConstantTensor struct {
	baseTensor
	tensor Tensor
	mul    float64
}

func (t *MulConstantTensor) Visit(v TensorVisitor) { v.VisitMulConstant(t) }

func MulConstant(tensor Tensor, mul float64) Tensor {
	return &MulConstantTensor{
		baseTensor: base(tensor.Shape(), tensor),
		tensor:     tensor,
		mul:        mul,
	}
}

func Sub(a Tensor, b Tensor) Tensor {
	return Add(a, MulConstant(b, -1))
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
