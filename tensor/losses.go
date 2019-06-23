package tensor

func BinaryCrossEntropy(yTrue Tensor, yPred Tensor) Tensor {
	// -mean(y log(yp) + (1-y)log(1-yp)))
	return Negate(Mean(
		Add(
			Mul(yTrue, Log(yPred)),
			Mul(
				Sub(Ones(yTrue.Shape()...), yTrue),
				Log(Sub(Ones(yPred.Shape()...), yPred)),
			),
		),
		len(yTrue.Shape())-1,
	))
}