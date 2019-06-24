package tensor

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
	VisitConcat(t *ConcatTensor)
	VisitSlice(t *SliceTensor)
	VisitUnslice(t *UnsliceTensor)
	VisitTranspose(t *TransposeTensor)
	VisitSum(t *SumTensor)
	VisitGreater(t *GreaterTensor)
	VisitReLU(t *ReLUTensor)
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
