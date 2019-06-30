package calc

func walkBroadcast(aShape []int, bShape []int, outShape []int, f func(aIndex int, bIndex int, outIndex int)) {
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

		for i := 0; i < outShape[axis]; i++ {
			walk(aIndex, bIndex, outIndex, axis+1)

			outIndex += outSize[axis]
			aIndex += aSize[axis]
			bIndex += bSize[axis]
		}
	}

	walk(0, 0, 0, 0)
}
