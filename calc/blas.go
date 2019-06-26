package calc

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

func blasConv2D(a NDArray, k NDArray, arr NDArray) {
	kShape := k.Shape()
	kh, kw, inf, kf := kShape[0], kShape[1], kShape[2], kShape[3]
	adim := len(a.shape)

	iWOff := a.shape[adim-1]
	iHOff := iWOff * a.shape[adim-2]
	iBOff := iHOff * a.shape[adim-3]

	aWOff := arr.shape[adim-1]
	aHOff := aWOff * arr.shape[adim-2]
	aBOff := aHOff * arr.shape[adim-3]

	kFOff := 1
	kIFOff := kFOff * kf
	kWOff := kIFOff * inf
	kHOff := kWOff * kw

	// loop over batches
	iBIndex := 0
	for aBIndex := 0; aBIndex < len(arr.data); aBIndex += aBOff {
		for r := 0; r < arr.shape[adim-3]; r++ {
			for c := 0; c < arr.shape[adim-2]; c++ {
				dataIndex := aBIndex + r*aHOff + c*aWOff
				iDataIndex := iBIndex + r*iHOff + c*iWOff

				y := blas64.Vector{
					N:    kf,
					Data: arr.data[dataIndex : dataIndex+kf],
					Inc:  1,
				}

				for h := 0; h < kh; h++ {
					for w := 0; w < kw; w++ {
						pIndex := iDataIndex + h*iHOff + w*iWOff
						kIndex := h*kHOff + w*kWOff
						x := blas64.Vector{
							N:    inf,
							Data: a.data[pIndex : pIndex+inf],
							Inc:  1,
						}
						a := blas64.General{
							Rows:   inf,
							Cols:   kf,
							Data:   k.data[kIndex : kIndex+inf*kf],
							Stride: kf,
						}
						blas64.Gemv(blas.Trans, 1.0, a, x, 1.0, y)
					}
				}
			}
		}
		iBIndex += iBOff
	}
}

func blasInverseConv2D(a NDArray, g NDArray, arr NDArray) {
	kShape := arr.Shape()
	kh, kw, inf, kf := kShape[0], kShape[1], kShape[2], kShape[3]

	adim := len(a.shape)

	iWOff := a.shape[adim-1]
	iHOff := iWOff * a.shape[adim-2]

	kFOff := 1
	kIFOff := kFOff * kf
	kWOff := kIFOff * inf
	kHOff := kWOff * kw

	g.ForEach(func(dataIndex int, index []int, value float64) {
		if index[adim-1] != 0 {
			return
		}
		iDataIndex := a.dataIndex(index)

		y := blas64.Vector{
			N:    kf,
			Data: g.data[dataIndex : dataIndex+kf],
			Inc:  1,
		}

		for h := 0; h < kh; h++ {
			for w := 0; w < kw; w++ {
				pIndex := iDataIndex + h*iHOff + w*iWOff
				kIndex := h*kHOff + w*kWOff
				x := blas64.Vector{
					N:    inf,
					Data: a.data[pIndex : pIndex+inf],
					Inc:  1,
				}
				a := blas64.General{
					Rows:   inf,
					Cols:   kf,
					Data:   arr.data[kIndex : kIndex+inf*kf],
					Stride: kf,
				}
				blas64.Ger(1.0, x, y, a)
			}
		}
	})
}
