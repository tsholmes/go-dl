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
					pIndex := iDataIndex + h*iHOff
					kIndex := h * kHOff
					x := blas64.Vector{
						N:    inf * kw,
						Data: a.data[pIndex : pIndex+inf*kw],
						Inc:  1,
					}
					a := blas64.General{
						Rows:   inf * kw,
						Cols:   kf,
						Data:   k.data[kIndex : kIndex+inf*kf*kw],
						Stride: kf,
					}
					blas64.Gemv(blas.Trans, 1.0, a, x, 1.0, y)
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
	iBOff := iHOff * a.shape[adim-3]

	gWOff := g.shape[adim-1]
	gHOff := gWOff * g.shape[adim-2]
	gBOff := gHOff * g.shape[adim-3]

	kFOff := 1
	kIFOff := kFOff * kf
	kWOff := kIFOff * inf
	kHOff := kWOff * kw

	for iBIndex, gBIndex := 0, 0; iBIndex < len(a.data); iBIndex, gBIndex = iBIndex+iBOff, gBIndex+gBOff {
		for r := 0; r < g.shape[adim-3]; r++ {
			for c := 0; c < g.shape[adim-2]; c++ {
				gDataIndex := gBIndex + r*gHOff + c*gWOff
				y := blas64.Vector{
					N:    kf,
					Data: g.data[gDataIndex : gDataIndex+kf],
					Inc:  1,
				}
				for h := 0; h < kh; h++ {
					iDataIndex := iBIndex + (r+h)*iHOff + c*iWOff
					kIndex := h * kHOff
					x := blas64.Vector{
						N:    inf,
						Data: a.data[iDataIndex : iDataIndex+inf*kw],
						Inc:  1,
					}
					a := blas64.General{
						Rows:   inf * kw,
						Cols:   kf,
						Data:   arr.data[kIndex : kIndex+inf*kf*kw],
						Stride: kf,
					}
					blas64.Ger(1.0, x, y, a)
				}
			}
		}
	}
}
