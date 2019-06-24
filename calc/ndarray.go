package calc

import (
	"math"
	"math/rand"
	"strconv"
)

func Zeros(shape ...int) NDArray {
	size := 1
	for _, s := range shape {
		size *= s
	}
	return NDArray{
		shape: shape,
		data:  make([]float64, size),
	}
}

func Constant(val float64, shape ...int) NDArray {
	arr := Zeros(shape...)
	for i := range arr.data {
		arr.data[i] = val
	}
	return arr
}

func Ones(shape ...int) NDArray {
	return Constant(1.0, shape...)
}

func RandomUniform(min float64, max float64, shape ...int) NDArray {
	arr := Zeros(shape...)
	for i := range arr.data {
		arr.data[i] = min + rand.Float64()*(max-min)
	}
	return arr
}

func BroadcastShape(aShape []int, bShape []int) []int {
	if len(aShape) > len(bShape) {
		return BroadcastShape(bShape, aShape)
	} else if len(aShape) < len(bShape) {
		// TODO: we don't actually handle adding dims in dataIndexBroadcast
		naShape := make([]int, len(bShape))
		for i := range naShape {
			j := i - len(bShape) + len(aShape)
			if j < 0 {
				naShape[i] = 1
			} else {
				naShape[i] = aShape[j]
			}
		}
		aShape = naShape
	}
	outShape := make([]int, len(aShape))
	for i := range aShape {
		if aShape[i] == bShape[i] {
			outShape[i] = aShape[i]
		} else if aShape[i] == 1 {
			outShape[i] = bShape[i]
		} else if bShape[i] == 1 {
			outShape[i] = aShape[i]
		} else {
			return nil
		}
	}
	return outShape
}

func AggrShape(aShape []int, axes []int) []int {
	outShape := make([]int, len(aShape))
	copy(outShape, aShape)
	for _, i := range axes {
		outShape[i] = 1
	}
	return outShape
}

func MatMulShape(aShape []int, bShape []int, a1 int, a2 int) []int {
	if len(aShape) != len(bShape) {
		return nil
	}
	if aShape[a2] != bShape[a1] {
		return nil
	}
	tAShape := append([]int{}, aShape...)
	tBShape := append([]int{}, bShape...)
	tAShape[a1], tAShape[a2], tBShape[a1], tBShape[a2] = 1, 1, 1, 1

	outShape := BroadcastShape(tAShape, tBShape)
	outShape[a1] = aShape[a1]
	outShape[a2] = bShape[a2]

	return outShape
}

func TransposeShape(shape []int, a1 int, a2 int) []int {
	outShape := append([]int{}, shape...)
	outShape[a1], outShape[a2] = shape[a2], shape[a1]
	return outShape
}

func FromRaw(shape []int, data []float64) NDArray {
	// TODO: validate len(data) = prod(shape)
	return NDArray{
		shape: shape,
		data:  data,
	}
}

type NDArray struct {
	shape []int
	data  []float64
}

func (a NDArray) dataIndex(index []int) int {
	dataIndex := 0
	innerSize := 1

	for i := len(index) - 1; i >= 0; i-- {
		dataIndex += innerSize * index[i]
		innerSize *= a.shape[i]
	}

	return dataIndex
}

func (a NDArray) dataIndexBroadcast(index []int) int {
	dataIndex := 0
	innerSize := 1

	for i := len(index) - 1; i >= 0; i-- {
		idx := index[i]
		if a.shape[i] == 1 {
			idx = 0
		}
		dataIndex += innerSize * idx
		innerSize *= a.shape[i]
	}

	return dataIndex
}

func (a NDArray) index(dataIndex int, reuse ...[]int) []int {
	var index []int
	if len(reuse) > 0 && cap(reuse[0]) >= len(a.shape) {
		index = reuse[0][:len(a.shape)]
	} else {
		index = make([]int, len(a.shape))
	}
	for i := len(a.shape) - 1; i >= 0; i-- {
		index[i] = dataIndex % a.shape[i]
		dataIndex /= a.shape[i]
	}
	return index
}

func (a NDArray) Get(index []int) float64 {
	return a.data[a.dataIndex(index)]
}

func (a NDArray) Set(index []int, value float64) {
	a.data[a.dataIndex(index)] = value
}

func (a NDArray) SetSlice(b NDArray, axis int, offset int) {
	b.ForEach(func(dataIndex int, index []int, value float64) {
		index[axis] += offset
		a.Set(index, value)
	})
}

func (a NDArray) Shape() []int {
	return a.shape
}

func (a NDArray) String() string {
	mods := make([]int, len(a.shape))
	mods[len(a.shape)-1] = a.shape[len(a.shape)-1]
	for i := len(a.shape) - 2; i >= 0; i-- {
		mods[i] = mods[i+1] * a.shape[i]
	}

	out := []byte{}

	for i := range a.data {
		opened := false
		for _, m := range mods {
			if i%m == 0 {
				if !opened && i > 0 {
					out = append(out, ' ')
				}
				opened = true
				out = append(out, '[')
			}
		}
		if !opened {
			out = append(out, ' ')
		}
		out = strconv.AppendFloat(out, a.data[i], 'g', -1, 64)
		for _, m := range mods {
			if (i+1)%m == 0 {
				out = append(out, ']')
			}
		}
	}
	return string(out)
}

func (a NDArray) ForEach(f func(dataIndex int, index []int, value float64)) {
	index := make([]int, len(a.shape))
	passedIndex := make([]int, len(a.shape))
	for i := range a.data {
		f(i, passedIndex, a.data[i])
		a.nextIndex(index)
		copy(passedIndex, index)
	}
}

func (a NDArray) nextIndex(index []int) {
	index[len(index)-1]++
	for i := len(index) - 1; i >= 0; i-- {
		if index[i] == a.shape[i] {
			index[i] = 0
			if i > 0 {
				index[i-1]++
			}
		} else {
			break
		}
	}
}

func (a NDArray) Add(b NDArray) NDArray {
	// TODO: assert size valid
	newShape := BroadcastShape(a.shape, b.shape)

	c := Zeros(newShape...)
	c.ForEach(func(dataIndex int, index []int, value float64) {
		aVal := a.data[a.dataIndexBroadcast(index)]
		bVal := b.data[b.dataIndexBroadcast(index)]
		c.data[dataIndex] = aVal + bVal
	})
	return c
}

func (a NDArray) MulConstant(b float64) NDArray {
	c := Zeros(a.shape...)
	copy(c.data, a.data)
	for i := range c.data {
		c.data[i] *= b
	}
	return c
}

func (a NDArray) Mul(b NDArray) NDArray {
	// TODO: assert size valid
	newShape := BroadcastShape(a.shape, b.shape)

	c := Zeros(newShape...)
	c.ForEach(func(dataIndex int, index []int, value float64) {
		aVal := a.data[a.dataIndexBroadcast(index)]
		bVal := b.data[b.dataIndexBroadcast(index)]
		c.data[dataIndex] = aVal * bVal
	})
	return c
}

func (a NDArray) Div(b NDArray) NDArray {
	// TODO: assert size valid
	newShape := BroadcastShape(a.shape, b.shape)

	c := Zeros(newShape...)
	c.ForEach(func(dataIndex int, index []int, value float64) {
		aVal := a.data[a.dataIndexBroadcast(index)]
		bVal := b.data[b.dataIndexBroadcast(index)]
		c.data[dataIndex] = aVal / bVal
	})
	return c
}

func (a NDArray) Concat(b NDArray, axis int) NDArray {
	// TODO: assert all sizes other than axis are equal
	shape := make([]int, len(a.shape))
	for i := range shape {
		if i == axis {
			shape[i] = a.shape[i] + b.shape[i]
		} else {
			shape[i] = a.shape[i]
		}
	}

	c := Zeros(shape...)
	c.ForEach(func(dataIndex int, index []int, value float64) {
		if index[axis] >= a.shape[axis] {
			index[axis] -= a.shape[axis]
			c.data[dataIndex] = b.Get(index)
		} else {
			c.data[dataIndex] = a.Get(index)
		}
	})
	return c
}

func (a NDArray) Slice(axis int, start int, end int) NDArray {
	outShape := append([]int{}, a.shape...)
	outShape[axis] = end - start

	arr := Zeros(outShape...)
	arr.ForEach(func(dataIndex int, index []int, value float64) {
		index[axis] += start
		arr.data[dataIndex] = a.Get(index)
	})
	return arr
}

func (a NDArray) Sign() NDArray {
	arr := Zeros(a.Shape()...)
	for i, v := range a.data {
		if v < 0 {
			arr.data[i] = -1.
		} else if v > 0 {
			arr.data[i] = 1.
		}
	}
	return arr
}

func (a NDArray) PowConstant(e float64) NDArray {
	arr := Zeros(a.Shape()...)
	for i := range arr.data {
		arr.data[i] = math.Pow(a.data[i], e)
	}
	return arr
}

func (a NDArray) Sum(axes ...int) NDArray {
	arr := Zeros(AggrShape(a.shape, axes)...)
	a.ForEach(func(dataIndex int, index []int, value float64) {
		arr.data[arr.dataIndexBroadcast(index)] += value
	})

	return arr
}

func (a NDArray) Mean(axes ...int) NDArray {
	arr := Zeros(AggrShape(a.shape, axes)...)
	div := 1
	for _, ax := range axes {
		div *= a.shape[ax]
	}
	a.ForEach(func(dataIndex int, index []int, value float64) {
		arr.data[arr.dataIndexBroadcast(index)] += value / float64(div)
	})

	return arr
}

func (a NDArray) Greater(b NDArray) NDArray {
	arr := Zeros(BroadcastShape(a.shape, b.shape)...)

	arr.ForEach(func(dataIndex int, index []int, value float64) {
		av := a.data[a.dataIndexBroadcast(index)]
		bv := b.data[b.dataIndexBroadcast(index)]
		if bv > av {
			arr.data[dataIndex] = 1.
		}
	})

	return arr
}

func (a NDArray) MatMul(b NDArray, a1 int, a2 int) NDArray {
	arr := Zeros(MatMulShape(a.shape, b.shape, a1, a2)...)

	aOff := 1
	bOff := 1
	for i := a2 + 1; i < len(a.shape); i++ {
		aOff *= a.shape[i]
	}
	for i := a1 + 1; i < len(b.shape); i++ {
		bOff *= b.shape[i]
	}

	var aIndex []int
	var bIndex []int

	arr.ForEach(func(dataIndex int, index []int, value float64) {
		aIndex = append(aIndex[:0], index...)
		bIndex = append(bIndex[:0], index...)

		aIndex[a2], bIndex[a1] = 0, 0

		aDIndex := a.dataIndex(aIndex)
		bDIndex := b.dataIndex(bIndex)

		for j := 0; j < a.shape[a2]; j++ {
			arr.data[dataIndex] += a.data[aDIndex] * b.data[bDIndex]
			aDIndex += aOff
			bDIndex += bOff
		}
	})

	return arr
}

func (a NDArray) Transpose(a1 int, a2 int) NDArray {
	arr := Zeros(TransposeShape(a.shape, a1, a2)...)

	arr.ForEach(func(dataIndex int, index []int, value float64) {
		index[a1], index[a2] = index[a2], index[a1]
		arr.data[dataIndex] = a.Get(index)
	})

	return arr
}

func (a NDArray) Log() NDArray {
	arr := Zeros(a.shape...)

	for i, v := range a.data {
		arr.data[i] = math.Log(v)
	}

	return arr
}

func (a NDArray) Exp() NDArray {
	arr := Zeros(a.shape...)

	for i, v := range a.data {
		arr.data[i] = math.Exp(v)
	}

	return arr
}
