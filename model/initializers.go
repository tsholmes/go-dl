package model

import (
	"math"

	"github.com/tsholmes/go-dl/calc"
)

type Initializer func(shape ...int) calc.NDArray

func GlorotUniform(shape ...int) calc.NDArray {
	fan_in := shape[len(shape)-2]
	fan_out := shape[len(shape)-1]
	init := math.Sqrt(6.0 / float64(fan_in+fan_out))
	return calc.RandomUniform(-init, init, shape...)
}

func GlorotNormal(shape ...int) calc.NDArray {
	fan_in := shape[len(shape)-2]
	fan_out := shape[len(shape)-1]
	init := math.Sqrt(2.0 / float64(fan_in+fan_out))
	return calc.RandomNormal(0, init, shape...)
}

func LecunNormal(shape ...int) calc.NDArray {
	fan_in := shape[len(shape)-2]
	init := math.Sqrt(1.0 / float64(fan_in))
	return calc.RandomNormal(0, init, shape...)
}

func Zeros(shape ...int) calc.NDArray {
	return calc.Zeros(shape...)
}
