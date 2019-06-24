package tensor

func Sigmoid(t Tensor) Tensor {
	// 1 / (1 + e^-x)
	return PowConstant(
		Add(
			Ones(t.Shape()...),
			Exp(Negate(t)),
		),
		-1,
	)
}

func Softmax(t Tensor) Tensor {
	axis := len(t.Shape()) - 1

	e := Exp(t)
	s := Sum(e, axis)

	return Div(e, s)
}
