package calc

import (
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

func (a NDArray) index(dataIndex int) []int {
	index := make([]int, len(a.shape))
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

func (a NDArray) Add(b NDArray) NDArray {
	// TODO: assert size valid
	newShape := BroadcastShape(a.shape, b.shape)

	c := Zeros(newShape...)
	for i := range c.data {
		aVal := a.data[a.dataIndexBroadcast(c.index(i))]
		bVal := b.data[b.dataIndexBroadcast(c.index(i))]
		c.data[i] = aVal + bVal
	}
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
	for i := range c.data {
		aVal := a.data[a.dataIndexBroadcast(c.index(i))]
		bVal := b.data[b.dataIndexBroadcast(c.index(i))]
		c.data[i] = aVal * bVal
	}
	return c
}

func (a NDArray) Div(b NDArray) NDArray {
	// TODO: assert size valid
	newShape := BroadcastShape(a.shape, b.shape)

	c := Zeros(newShape...)
	for i := range c.data {
		aVal := a.data[a.dataIndexBroadcast(c.index(i))]
		bVal := b.data[b.dataIndexBroadcast(c.index(i))]
		c.data[i] = aVal / bVal
	}
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
	for i := range c.data {
		index := c.index(i)
		if index[axis] >= a.shape[axis] {
			index[axis] -= a.shape[axis]
			c.data[i] = b.Get(index)
		} else {
			c.data[i] = a.Get(index)
		}
	}
	return c
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
