package calc

import (
	"fmt"
	"math"
	"math/rand"
	"strconv"
)

const Epsilon = 1e-9

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

func RandomNormal(mean float64, stddev float64, shape ...int) NDArray {
	arr := Zeros(shape...)
	for i := range arr.data {
		arr.data[i] = mean + rand.NormFloat64()*stddev
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

func Conv2DShape(shape []int, hAxis int, wAxis int, fAxis int, kernelH int, kernelW int, kernelF int) []int {
	outShape := append([]int{}, shape...)
	outShape[hAxis] -= kernelH - 1
	outShape[wAxis] -= kernelW - 1
	outShape[fAxis] = kernelF
	return outShape
}

func InverseConv2DShape(shape []int, kShape []int, hAxis int, wAxis int, fAxis int) []int {
	kernelH := shape[hAxis] - kShape[hAxis] + 1
	kernelW := shape[wAxis] - kShape[wAxis] + 1
	return []int{kernelH, kernelW, shape[fAxis], kShape[fAxis]}
}

func FromRaw(shape []int, data []float64) NDArray {
	// TODO: validate len(data) = prod(shape)
	return NDArray{
		shape: shape,
		data:  data,
	}
}

func ShapeEqual(s1 []int, s2 []int) bool {
	if len(s1) != len(s2) {
		return false
	}
	for i := range s1 {
		if s1[i] != s2[i] {
			return false
		}
	}
	return true
}

type NDArray struct {
	shape []int
	data  []float64

	// cached offsets for dataIndex*
	broadcastSizes []int
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

func (a *NDArray) dataIndexBroadcast(index []int) int {
	if len(a.broadcastSizes) == 0 {
		a.broadcastSizes = make([]int, len(a.shape))
		innerSize := 1
		for i := len(index) - 1; i >= 0; i-- {
			if a.shape[i] != 1 {
				a.broadcastSizes[i] = innerSize
			}
			innerSize *= a.shape[i]
		}
	}
	dataIndex := 0

	for i := len(index) - 1; i >= 0; i-- {
		dataIndex += a.broadcastSizes[i] * index[i]
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

func (a NDArray) Fill(value float64) {
	for i := range a.data {
		a.data[i] = value
	}
}

func (a NDArray) SetSlice(b NDArray, axis int, offset int) {
	walkSlice(a.shape, b.shape, axis, offset, func(inIndex int, outIndex int) {
		a.data[inIndex] = b.data[outIndex]
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
	return a.AddInto(b, c)
}

func (a NDArray) AddInto(b NDArray, c NDArray) NDArray {
	// TODO: assert size valid
	if ShapeEqual(a.shape, b.shape) {
		for i := range c.data {
			c.data[i] = a.data[i] + b.data[i]
		}
	} else {
		walkBroadcast(a.shape, b.shape, c.shape, func(aIndex int, bIndex int, outIndex int) {
			c.data[outIndex] = a.data[aIndex] + b.data[bIndex]
		})
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
	return a.MulInto(b, c)
}

func (a NDArray) MulInto(b NDArray, c NDArray) NDArray {
	// TODO: assert size valid
	if ShapeEqual(a.Shape(), b.Shape()) {
		for i := range c.data {
			c.data[i] = a.data[i] * b.data[i]
		}
	} else {
		walkBroadcast(a.shape, b.shape, c.shape, func(aIndex int, bIndex int, outIndex int) {
			c.data[outIndex] = a.data[aIndex] * b.data[bIndex]
		})
	}
	return c
}

func (a NDArray) Div(b NDArray) NDArray {
	// TODO: assert size valid
	newShape := BroadcastShape(a.shape, b.shape)

	c := Zeros(newShape...)
	walkBroadcast(a.shape, b.shape, c.shape, func(aIndex int, bIndex int, outIndex int) {
		c.data[outIndex] = a.data[aIndex] / b.data[bIndex]
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
	walkSlice(a.shape, outShape, axis, start, func(inIndex int, outIndex int) {
		arr.data[outIndex] = a.data[inIndex]
	})
	return arr
}

func (a NDArray) Split(axis int, batch int) []NDArray {
	if a.shape[axis]%batch != 0 {
		panic(fmt.Sprintf("Cannot split %v on axis %d into batches of %d", a.shape, axis, batch))
	}
	batchCount := a.shape[axis] / batch
	batchShape := append([]int{}, a.shape...)
	batchShape[axis] = batch

	arrs := make([]NDArray, batchCount)
	for i := range arrs {
		arrs[i] = Zeros(batchShape...)
	}

	a.ForEach(func(dataIndex int, index []int, value float64) {
		batchI := index[axis] / batch
		index[axis] -= batchI * batch
		arrs[batchI].Set(index, value)
	})

	return arrs
}

func (a NDArray) Reshape(shape ...int) NDArray {
	// TODO: validate prod(a.shape) == prod(shape)
	return NDArray{
		shape: shape,
		data:  a.data,
	}
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
	f := func(v float64) float64 { return math.Pow(v, e) }
	if e == 2.0 {
		f = func(v float64) float64 { return v * v }
	} else if e == 0.5 {
		f = math.Sqrt
	}
	arr := Zeros(a.Shape()...)
	for i := range arr.data {
		arr.data[i] = f(a.data[i])
	}
	return arr
}

func (a NDArray) Sum(axes ...int) NDArray {
	arr := Zeros(AggrShape(a.shape, axes)...)
	walkAggr(a.shape, arr.shape, func(inIndex int, outIndex int) {
		arr.data[outIndex] += a.data[inIndex]
	})
	return arr
}

func (a NDArray) Mean(axes ...int) NDArray {
	arr := Zeros(AggrShape(a.shape, axes)...)
	div := 1
	for _, ax := range axes {
		div *= a.shape[ax]
	}
	walkAggr(a.shape, arr.shape, func(inIndex int, outIndex int) {
		arr.data[outIndex] += a.data[inIndex]
	})
	for i := range arr.data {
		arr.data[i] /= float64(div)
	}

	return arr
}

func (a NDArray) Max(axes ...int) NDArray {
	arr := Constant(math.Inf(-1), AggrShape(a.shape, axes)...)
	walkAggr(a.shape, arr.shape, func(inIndex int, outIndex int) {
		if a.data[inIndex] > arr.data[outIndex] {
			arr.data[outIndex] = a.data[inIndex]
		}
	})

	return arr
}

func (a NDArray) Greater(b NDArray) NDArray {
	arr := Zeros(BroadcastShape(a.shape, b.shape)...)

	arr.ForEach(func(dataIndex int, index []int, value float64) {
		av := a.data[a.dataIndexBroadcast(index)]
		bv := b.data[b.dataIndexBroadcast(index)]
		if av > bv {
			arr.data[dataIndex] = 1.
		}
	})

	return arr
}

func (a NDArray) Equal(b NDArray) NDArray {
	arr := Zeros(BroadcastShape(a.shape, b.shape)...)

	arr.ForEach(func(dataIndex int, index []int, value float64) {
		av := a.data[a.dataIndexBroadcast(index)]
		bv := b.data[b.dataIndexBroadcast(index)]
		if math.Abs(av-bv) < Epsilon {
			arr.data[dataIndex] = 1.
		}
	})

	return arr
}

func (a NDArray) EqualMask(e1 NDArray, e2 NDArray) NDArray {
	arr := Zeros(a.shape...)

	if ShapeEqual(a.shape, e1.shape) && ShapeEqual(a.shape, e2.shape) {
		for i, v := range a.data {
			if math.Abs(e1.data[i]-e2.data[i]) < Epsilon {
				arr.data[i] = v
			}
		}
	} else {
		arr.ForEach(func(dataIndex int, index []int, value float64) {
			v1 := e1.data[e1.dataIndexBroadcast(index)]
			v2 := e2.data[e2.dataIndexBroadcast(index)]
			if math.Abs(v1-v2) < Epsilon {
				arr.data[dataIndex] = a.data[dataIndex]
			}
		})
	}

	return arr
}

func (a NDArray) MatMul(b NDArray, a1 int, a2 int) NDArray {
	arr := Zeros(MatMulShape(a.shape, b.shape, a1, a2)...)
	return a.MatMulInto(b, a1, a2, arr)
}

func (a NDArray) MatMulInto(b NDArray, a1 int, a2 int, arr NDArray) NDArray {
	// We rely on the data being zeros
	arr.Fill(0.)

	aOff := 1
	bOff := 1
	for i := a2 + 1; i < len(a.shape); i++ {
		aOff *= a.shape[i]
	}
	for i := a1 + 1; i < len(b.shape); i++ {
		bOff *= b.shape[i]
	}

	arr.ForEach(func(dataIndex int, index []int, value float64) {
		ia2 := index[a2]

		index[a2] = 0
		aDIndex := a.dataIndexBroadcast(index)
		index[a1], index[a2] = 0, ia2
		bDIndex := b.dataIndexBroadcast(index)

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

func (a NDArray) Clip(min float64, max float64) NDArray {
	arr := Zeros(a.shape...)

	for i, v := range a.data {
		if v < min {
			v = min
		} else if v > max {
			v = max
		}
		arr.data[i] = v
	}
	return arr
}

func (a NDArray) Reverse(axes ...int) NDArray {
	arr := Zeros(a.shape...)
	a.ForEach(func(dataIndex int, index []int, value float64) {
		for _, ax := range axes {
			index[ax] = a.Shape()[ax] - index[ax] - 1
		}
		arr.Set(index, value)
	})
	return arr
}

func (a NDArray) Conv2D(k NDArray, hAxis int, wAxis int, fAxis int) NDArray {
	kShape := k.Shape()
	kh, kw, kf := kShape[0], kShape[1], kShape[3]
	arr := Zeros(Conv2DShape(a.Shape(), hAxis, wAxis, fAxis, kh, kw, kf)...)
	return a.Conv2DInto(k, hAxis, wAxis, fAxis, arr)
}

func (a NDArray) Conv2DInto(k NDArray, hAxis int, wAxis int, fAxis int, arr NDArray) NDArray {
	kShape := k.Shape()
	kh, kw, inf, kf := kShape[0], kShape[1], kShape[2], kShape[3]

	arr.Fill(0.)

	adim := len(a.shape)
	if fAxis == adim-1 && wAxis == adim-2 && hAxis == adim-3 {
		blasConv2D(a, k, arr)
		return arr
	}

	var iHOff, iWOff, iFOff int
	iSize := 1
	for i := len(a.shape) - 1; i >= 0; i-- {
		if i == fAxis {
			iFOff = iSize
		} else if i == wAxis {
			iWOff = iSize
		} else if i == hAxis {
			iHOff = iSize
		}
		iSize *= a.shape[i]
	}

	kFOff := 1
	kIFOff := kFOff * kf
	kWOff := kIFOff * inf
	kHOff := kWOff * kw

	arr.ForEach(func(dataIndex int, index []int, value float64) {
		fIndex := index[fAxis]
		index[fAxis] = 0
		iDataIndex := a.dataIndex(index)

		ai1 := iDataIndex
		ki1 := fIndex * kFOff
		for h := 0; h < kh; h++ {
			ai2 := ai1
			ki2 := ki1
			for w := 0; w < kw; w++ {
				ai := ai2
				ki := ki2
				for f := 0; f < inf; f++ {
					arr.data[dataIndex] += a.data[ai] * k.data[ki]
					ai += iFOff
					ki += kIFOff
				}
				ai2 += iWOff
				ki2 += kWOff
			}
			ai1 += iHOff
			ki1 += kHOff
		}
	})
	return arr
}

func (a NDArray) InverseConv2D(g NDArray, hAxis int, wAxis int, fAxis int) NDArray {
	kShape := InverseConv2DShape(a.shape, g.shape, hAxis, wAxis, fAxis)
	arr := Zeros(kShape...)
	return a.InverseConv2DInto(g, hAxis, wAxis, fAxis, arr)
}

func (a NDArray) InverseConv2DInto(g NDArray, hAxis int, wAxis int, fAxis int, arr NDArray) NDArray {
	kShape := arr.Shape()
	kh, kw, inf, kf := kShape[0], kShape[1], kShape[2], kShape[3]
	arr.Fill(0.)

	adim := len(a.shape)
	if fAxis == adim-1 && wAxis == adim-2 && hAxis == adim-3 {
		blasInverseConv2D(a, g, arr)
		return arr
	}

	var iHOff, iWOff, iFOff int
	iSize := 1
	for i := len(a.shape) - 1; i >= 0; i-- {
		if i == fAxis {
			iFOff = iSize
		} else if i == wAxis {
			iWOff = iSize
		} else if i == hAxis {
			iHOff = iSize
		}
		iSize *= a.shape[i]
	}

	kFOff := 1
	kIFOff := kFOff * kf
	kWOff := kIFOff * inf
	kHOff := kWOff * kw

	g.ForEach(func(dataIndex int, index []int, value float64) {
		fIndex := index[fAxis]
		index[fAxis] = 0
		iDataIndex := a.dataIndex(index)

		ai1 := iDataIndex
		ki1 := fIndex * kFOff
		for h := 0; h < kh; h++ {
			ai2 := ai1
			ki2 := ki1
			for w := 0; w < kw; w++ {
				ai := ai2
				ki := ki2
				for f := 0; f < inf; f++ {
					arr.data[ki] += a.data[ai] * value
					ai += iFOff
					ki += kIFOff
				}
				ai2 += iWOff
				ki2 += kWOff
			}
			ai1 += h * iHOff
			ki1 += h * kHOff
		}
	})

	return arr
}

func (a NDArray) ReLU() NDArray {
	arr := Zeros(a.shape...)

	for i, v := range a.data {
		if v > 0. {
			arr.data[i] = v
		}
	}

	return arr
}

func (a NDArray) ReLUMask(m NDArray) NDArray {
	arr := Zeros(BroadcastShape(a.shape, m.shape)...)

	if ShapeEqual(a.shape, m.shape) {
		for i := range arr.data {
			if m.data[i] > 0 {
				arr.data[i] = a.data[i]
			}
		}
	} else {
		arr.ForEach(func(dataIndex int, index []int, value float64) {
			aIndex := a.dataIndexBroadcast(index)
			mIndex := m.dataIndexBroadcast(index)

			if m.data[mIndex] > 0 {
				arr.data[dataIndex] = a.data[aIndex]
			}
		})
	}

	return arr
}

func (a NDArray) ReindexRoot(indices []int) NDArray {
	if len(indices) != a.shape[0] {
		// TODO: PANIC
	}

	arr := Zeros(a.shape...)

	topSz := 1
	for _, sz := range a.shape[1:] {
		topSz *= sz
	}

	for i, idx := range indices {
		inIndex := i * topSz
		outIndex := idx * topSz
		for j, v := range a.data[inIndex : inIndex+topSz] {
			arr.data[outIndex+j] = v
		}
	}

	return arr
}

func (a NDArray) SliceRoot(start int, length int) NDArray {
	topSz := 1
	for _, sz := range a.shape[1:] {
		topSz *= sz
	}

	newShape := make([]int, len(a.shape))
	copy(newShape, a.shape)
	newShape[0] = length

	return NDArray{
		data:  a.data[start*topSz : start*topSz+length*topSz],
		shape: newShape,
	}
}
