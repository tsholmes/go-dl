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
