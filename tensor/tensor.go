package tensor

type Tensor interface {
	ID() int64
	Shape() []int
}

var nextID int64

type baseTensor struct {
	id    int64
	shape []int
}

func (b *baseTensor) ID() int64 {
	return b.id
}

func (b *baseTensor) Shape() []int {
	return b.shape
}

func base(shape []int) baseTensor {
	// TODO: lock around nextID
	id := nextID
	nextID++
	return baseTensor{
		id:    id,
		shape: shape,
	}
}

type InputTensor struct {
	baseTensor
	shape []int
}

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

func Add(as ...Tensor) Tensor {
	return &AddTensor{
		baseTensor: base(elementWise(as...)),
		as:         as,
	}
}

type MulConstantTensor struct {
	baseTensor
	tensor Tensor
	mul    float64
}

func MulConstant(tensor Tensor, mul float64) Tensor {
	return &MulConstantTensor{
		baseTensor: base(tensor.Shape()),
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

func Mul(as ...Tensor) Tensor {
	return &MulTensor{
		baseTensor: base(elementWise(as...)),
		as:         as,
	}
}

type DivTensor struct {
	baseTensor
	a Tensor
	b Tensor
}

func Div(a Tensor, b Tensor) Tensor {
	return &DivTensor{
		baseTensor: base(elementWise(a, b)),
		a:          a,
		b:          b,
	}
}

type ConcatTensor struct {
	baseTensor
	axis int
	as   []Tensor
}

func Concat(axis int, as ...Tensor) Tensor {
	return &ConcatTensor{
		baseTensor: base(concat(axis, as...)),
		axis:       axis,
		as:         as,
	}
}
