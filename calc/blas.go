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

	for aBIndex, iBIndex := 0, 0; aBIndex < len(arr.data); aBIndex, iBIndex = aBIndex+aBOff, iBIndex+iBOff {
		for r := 0; r < arr.shape[adim-3]; r++ {
			aDataIndex := aBIndex + r*aHOff
			c := blas64.General{
				Rows:   arr.shape[adim-2],
				Cols:   kf,
				Data:   arr.data[aDataIndex:],
				Stride: kf,
			}
			for h := 0; h < kh; h++ {
				for w := 0; w < kw; w++ {
					iDataIndex := iBIndex + (h+r)*iHOff + w*iWOff
					kIndex := kHOff*h + kWOff*w
					a := blas64.General{
						Rows:   arr.shape[adim-2],
						Cols:   inf,
						Data:   a.data[iDataIndex:],
						Stride: inf,
					}
					b := blas64.General{
						Rows:   inf,
						Cols:   kf,
						Data:   k.data[kIndex:],
						Stride: kf,
					}
					blas64.Gemm(blas.NoTrans, blas.NoTrans, 1.0, a, b, 1.0, c)
				}
			}
		}
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

func blasStddev(data []float64, stddev []float64) {
	sz := len(stddev)
	sz2 := len(data) / sz
	for i := 0; i < sz; i++ {
		stddev[i] = blas64.Nrm2(blas64.Vector{
			N:    sz2,
			Data: data[i:],
			Inc:  sz,
		}) / float64(sz2)
	}
}

func blasDVariance(a []float64, g []float64, stddev []float64, dVariance []float64) {
	sz := len(stddev)
	sz2 := len(a) / sz

	for i := 0; i < sz; i++ {
		s := stddev[i]
		dVariance[i] = blas64.Dot(blas64.Vector{
			N:    sz2,
			Data: a[i:],
			Inc:  sz,
		}, blas64.Vector{
			N:    sz2,
			Data: g[i:],
			Inc:  sz,
		}) * -0.5 / (s * s * s)
	}
}
