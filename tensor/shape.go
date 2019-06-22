package tensor

func elementWise(as ...Tensor) []int {
	// TODO: validate equal
	return as[0].Shape()
}

func concat(axis int, as ...Tensor) []int {
	// TODO validate match
	shape := make([]int, len(as[0].Shape()))
	copy(shape, as[0].Shape())
	for i := 1; i < len(as); i++ {
		shape[axis] += as[i].Shape()[axis]
	}
	return shape
}
