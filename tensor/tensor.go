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
	VisitMatMul(t *MatMulTensor)
	VisitLog(t *LogTensor)
	VisitExp(t *ExpTensor)
	VisitNormalize(t *NormalizeTensor)
	VisitInverseNormalize(t *InverseNormalizeTensor)
	VisitConv2D(t *Conv2DTensor)
	VisitInverseConv2D(t *InverseConv2DTensor)
	VisitConcat(t *ConcatTensor)
	VisitSlice(t *SliceTensor)
	VisitUnslice(t *UnsliceTensor)
	VisitTranspose(t *TransposeTensor)
	VisitReshape(t *ReshapeTensor)
	VisitReverse(t *ReverseTensor)
	VisitSum(t *SumTensor)
	VisitMax(t *MaxTensor)
	VisitGreater(t *GreaterTensor)
	VisitEqual(t *EqualTensor)
	VisitReLU(t *ReLUTensor)
	VisitReLUMask(t *ReLUMaskTensor)
	VisitEqualMask(t *EqualMaskTensor)
}

var nextID int64

type baseTensor struct {
	id     int64
	shape  []int
	inputs []Tensor

	values []calc.NDArray
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

func base(shape []int, tempValues int, inputs ...Tensor) baseTensor {
	// TODO: lock around nextID
	id := nextID
	nextID++

	values := make([]calc.NDArray, tempValues)
	for i := range values {
		values[i] = calc.Zeros(shape...)
	}
	return baseTensor{
		id:     id,
		shape:  shape,
		inputs: inputs,
		values: values,
	}
}
