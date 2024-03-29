package calc

func walkBroadcast(aShape []int, bShape []int, outShape []int, f func(aIndex int, bIndex int, outIndex int)) {
	asz, bsz := 1, 1
	for i := range aShape {
		asz *= aShape[i]
		bsz *= bShape[i]
	}
	if canFastBroadcast(aShape, bShape) {
		fastWalkAggr(asz, bsz, func(inIndex int, outIndex int) {
			f(inIndex, outIndex, inIndex)
		})
		return
	} else if canFastBroadcast(bShape, aShape) {
		fastWalkAggr(bsz, asz, func(inIndex int, outIndex int) {
			f(outIndex, inIndex, inIndex)
		})
		return
	}

	aSize := make([]int, len(aShape))
	bSize := make([]int, len(aShape))
	outSize := make([]int, len(aShape))

	for i := len(aShape) - 1; i >= 0; i-- {
		if i == len(aShape)-1 {
			aSize[i] = 1
			bSize[i] = 1
			outSize[i] = 1
		} else {
			aSize[i] = aSize[i+1] * aShape[i+1]
			bSize[i] = bSize[i+1] * bShape[i+1]
			outSize[i] = outSize[i+1] * outShape[i+1]
		}
	}

	for i := range aShape {
		if aShape[i] == 1 {
			aSize[i] = 0
		}
		if bShape[i] == 1 {
			bSize[i] = 0
		}
	}

	var walk func(int, int, int, int)
	walk = func(aIndex int, bIndex int, outIndex int, axis int) {
		if axis == len(aShape) {
			f(aIndex, bIndex, outIndex)
			return
		}

		outInc := outSize[axis]
		aInc := aSize[axis]
		bInc := bSize[axis]

		for i := 0; i < outShape[axis]; i++ {
			walk(aIndex, bIndex, outIndex, axis+1)

			outIndex += outInc
			aIndex += aInc
			bIndex += bInc
		}
	}

	walk(0, 0, 0, 0)
}

func walkAggr(inShape []int, outShape []int, f func(inIndex int, outIndex int)) {
	if canFastAggr(outShape) {
		inSz := 1
		for i := range inShape {
			inSz *= inShape[i]
		}
		fastWalkAggr(inSz, outShape[len(outShape)-1], f)
		return
	}
	inSize := make([]int, len(inShape))
	outSize := make([]int, len(inShape))

	for i := len(inShape) - 1; i >= 0; i-- {
		if i == len(inShape)-1 {
			inSize[i] = 1
			outSize[i] = 1
		} else {
			inSize[i] = inSize[i+1] * inShape[i+1]
			outSize[i] = outSize[i+1] * outShape[i+1]
		}
	}

	for i := range outShape {
		if outShape[i] == 1 {
			outSize[i] = 0
		}
	}

	var walk func(int, int, int)
	walk = func(inIndex int, outIndex int, axis int) {
		if axis == len(inShape) {
			f(inIndex, outIndex)
			return
		}

		for i := 0; i < inShape[axis]; i++ {
			walk(inIndex, outIndex, axis+1)

			outIndex += outSize[axis]
			inIndex += inSize[axis]
		}
	}

	walk(0, 0, 0)
}

func walkSlice(inShape []int, outShape []int, sliceAxis int, offset int, f func(inIndex int, outIndex int)) {
	inSize := make([]int, len(inShape))
	outSize := make([]int, len(inShape))

	for i := len(inShape) - 1; i >= 0; i-- {
		if i == len(inShape)-1 {
			inSize[i] = 1
			outSize[i] = 1
		} else {
			inSize[i] = inSize[i+1] * inShape[i+1]
			outSize[i] = outSize[i+1] * outShape[i+1]
		}
	}

	var walk func(int, int, int)
	walk = func(inIndex int, outIndex int, axis int) {
		if axis == len(inShape) {
			f(inIndex, outIndex)
			return
		}

		if axis == sliceAxis {
			inIndex += offset * inSize[axis]
		}

		for i := 0; i < outShape[axis]; i++ {
			walk(inIndex, outIndex, axis+1)

			outIndex += outSize[axis]
			inIndex += inSize[axis]
		}
	}

	walk(0, 0, 0)
}

func canFastAggr(outShape []int) bool {
	for i := len(outShape) - 2; i >= 0; i-- {
		if outShape[i] > 1 {
			return false
		}
	}
	return true
}

func fastWalkAggr(inSize int, outSize int, f func(inIndex int, outIndex int)) {
	for inIndex := 0; inIndex < inSize; inIndex += outSize {
		for outIndex := 0; outIndex < outSize; outIndex++ {
			f(inIndex+outIndex, outIndex)
		}
	}
}

func canFastBroadcast(aShape []int, bShape []int) bool {
	lastAxis := len(aShape) - 1
	if aShape[lastAxis] != bShape[lastAxis] {
		return false
	}
	for i := lastAxis - 1; i >= 0; i-- {
		if bShape[i] > 1 {
			return false
		}
	}
	return true
}
