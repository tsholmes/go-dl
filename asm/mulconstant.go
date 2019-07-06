package asm

import (
	asm "github.com/twitchyliquid64/golang-asm"
	"github.com/twitchyliquid64/golang-asm/obj"
	"github.com/twitchyliquid64/golang-asm/obj/x86"
	"gonum.org/v1/gonum/blas/blas64"
)

func LoopMulConstant(arr []float64, c float64) {
	for i := range arr {
		arr[i] = arr[i] * c
	}
}

func ScalMulConstant(arr []float64, c float64) {
	blas64.Scal(c, blas64.Vector{
		N:    len(arr),
		Data: arr,
		Inc:  1,
	})
}

// Sized to stay inside 1 memory page
const mulChunkSize = 32 * 10

func CompileMulConstant(N int) func([]float64, float64) {
	bigChunks := N / mulChunkSize
	extra := N % mulChunkSize

	b, err := asm.NewBuilder("amd64", mulChunkSize*3+10)
	if err != nil {
		panic(err)
	}
	movq(b, aregOff(x86.REG_SP, 0x8), reg(x86.REG_SI)) // SI == arr.data.ptr
	if N%4 != 0 {
		movddup(b, aregOff(x86.REG_SP, 0x20), reg(x86.REG_X0)) // X0 = c, c
	}
	if N >= 4 {
		vbroadcastsd(b, aregOff(x86.REG_SP, 0x20), reg(x86.REG_Y0)) // Y0 = c, c, c, c
	}

	// If we have more than one chunk, keep a iteration counter
	if bigChunks > 1 {
		movq(b, constant(bigChunks), reg(x86.REG_CX))
	}

	// if we have at least mulChunkSize, write the big loop
	var bigStart *obj.Prog
	if bigChunks > 0 {
		bigStart = compileMulChunk(mulChunkSize, b)

		if bigChunks > 1 || extra > 0 {
			// if we are looping or there is an extra chunk, move the pointer
			addq(b, constant(8*mulChunkSize), reg(x86.REG_SI))
		}
		if bigChunks > 1 {
			_ = bigStart
			decq(b, reg(x86.REG_CX))
			jnz(b, bigStart)
		}
	}
	if extra > 0 {
		compileMulChunk(extra, b)
	}
	ret(b)

	var f func([]float64, float64)

	makeFunc(&f, b.Assemble())

	return f
}

func compileMulChunk(N int, b *asm.Builder) *obj.Prog {
	blockSize := 32
	var first *obj.Prog
	// Do 4s in chunks
	for i := 0; i+blockSize <= N; i += blockSize {
		p := fload1(b, blockSize, x86.REG_SI, i, 1)
		if i == 0 {
			first = p
		}
		fmul1(b, blockSize, x86.REG_SI, i, 1)
		fstore1(b, blockSize, x86.REG_SI, i, 1)
	}
	if n := N % blockSize; n != 0 {
		p := fload1(b, n, x86.REG_SI, N-n, 1)
		if n == N {
			first = p
		}
		fmul1(b, n, x86.REG_SI, N-n, 1)
		fstore1(b, n, x86.REG_SI, N-n, 1)
	}
	return first
}
