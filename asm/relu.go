package asm

import (
	asm "github.com/twitchyliquid64/golang-asm"
	"github.com/twitchyliquid64/golang-asm/obj"
	"github.com/twitchyliquid64/golang-asm/obj/x86"
)

func LoopReLU(arr []float64) {
	for i := range arr {
		if arr[i] < 0 {
			arr[i] = 0
		}
	}
}

const reluChunkSize = 2 * 4

func CompileReLU(N int) func([]float64) {
	bigChunks := N / reluChunkSize
	extra := N % reluChunkSize

	b, err := asm.NewBuilder("amd64", reluChunkSize*3+10)
	if err != nil {
		panic(err)
	}
	movq(b, aregOff(x86.REG_SP, 0x8), reg(x86.REG_SI))           // SI == arr.data.ptr
	vxorps(b, reg(x86.REG_X0), reg(x86.REG_X0), reg(x86.REG_X0)) // X0 (and Y0) == 0

	// If we have more than one chunk, keep a iteration counter
	if bigChunks > 1 {
		movq(b, constant(bigChunks), reg(x86.REG_CX))
	}

	// if we have at least maxChunkSize, write the big loop
	var bigStart *obj.Prog
	if bigChunks > 0 {
		bigStart = compileReLUChunk(reluChunkSize, b)

		if bigChunks > 1 || extra > 0 {
			// if we are looping or there is an extra chunk, move the pointer
			addq(b, constant(8*reluChunkSize), reg(x86.REG_SI))
		}
		if bigChunks > 1 {
			decq(b, reg(x86.REG_CX))
			jnz(b, bigStart)
		}
	}
	if extra > 0 {
		compileReLUChunk(extra, b)
	}
	ret(b)

	var f func([]float64)

	makeFunc(&f, b.Assemble())

	return f
}

func compileReLUChunk(N int, b *asm.Builder) *obj.Prog {
	blockSize := 4
	var first *obj.Prog
	for i := 0; i+blockSize <= N; i += blockSize {
		p := frelumask(b, blockSize, x86.REG_SI, i, 1)
		if i == 0 {
			first = p
		}
		fmaskstore(b, blockSize, x86.REG_SI, i, 1)
	}
	if n := N % blockSize; n != 0 {
		p := frelumask(b, n, x86.REG_SI, N-n, 1)
		if n == N {
			first = p
		}
		fmaskstore(b, n, x86.REG_SI, N-n, 1)
	}
	return first
}

func LoopReLUCopy(src []float64, dst []float64) {
	for i := range src {
		if src[i] > 0 {
			dst[i] = src[i]
		} else {
			dst[i] = 0
		}
	}
}

func CompileReLUCopy(N int) func([]float64, []float64) {
	bigChunks := N / reluChunkSize
	extra := N % reluChunkSize

	b, err := asm.NewBuilder("amd64", reluChunkSize*3+10)
	if err != nil {
		panic(err)
	}
	movq(b, aregOff(x86.REG_SP, 0x8), reg(x86.REG_SI))           // SI == src.data.ptr
	movq(b, aregOff(x86.REG_SP, 0x20), reg(x86.REG_DI))          // DI == dst.data.ptr
	vxorps(b, reg(x86.REG_X0), reg(x86.REG_X0), reg(x86.REG_X0)) // X0 (and Y0) == 0

	// If we have more than one chunk, keep a iteration counter
	if bigChunks > 1 {
		movq(b, constant(bigChunks), reg(x86.REG_CX))
	}

	// if we have at least maxChunkSize, write the big loop
	var bigStart *obj.Prog
	if bigChunks > 0 {
		bigStart = compileReLUCopyChunk(reluChunkSize, b)

		if bigChunks > 1 || extra > 0 {
			// if we are looping or there is an extra chunk, move the pointer
			addq(b, constant(8*reluChunkSize), reg(x86.REG_SI))
			addq(b, constant(8*reluChunkSize), reg(x86.REG_DI))
		}
		if bigChunks > 1 {
			decq(b, reg(x86.REG_CX))
			jnz(b, bigStart)
		}
	}
	if extra > 0 {
		compileReLUCopyChunk(extra, b)
	}
	ret(b)

	var f func([]float64, []float64)

	makeFunc(&f, b.Assemble())

	return f
}

func compileReLUCopyChunk(N int, b *asm.Builder) *obj.Prog {
	blockSize := 4
	var first *obj.Prog
	for i := 0; i+blockSize <= N; i += blockSize {
		p := freluload(b, blockSize, x86.REG_SI, i, 1)
		if i == 0 {
			first = p
		}
		fstore(b, blockSize, x86.REG_DI, i, 1, f256)
	}
	if n := N % blockSize; n != 0 {
		p := freluload(b, n, x86.REG_SI, N-n, 1)
		if n == N {
			first = p
		}
		fstore(b, n, x86.REG_DI, N-n, 1, f256)
	}
	return first
}

func LoopReLUMask(msrc []float64, dsrc []float64, dst []float64) {
	for i := range msrc {
		if msrc[i] > 0 {
			dst[i] = dsrc[i]
		} else {
			dst[i] = 0
		}
	}
}

const (
	reluMaskBlockSize = 8
	reluMaskChunkSize = reluMaskBlockSize * 128
)

func CompileReLUMask(N int) func([]float64, []float64, []float64) {
	bigChunks := N / reluMaskChunkSize
	extra := N % reluMaskChunkSize

	b, err := asm.NewBuilder("amd64", reluMaskChunkSize*3+10)
	if err != nil {
		panic(err)
	}
	movq(b, aregOff(x86.REG_SP, 0x8), reg(x86.REG_SI))           // SI == msrc.data.ptr
	movq(b, aregOff(x86.REG_SP, 0x20), reg(x86.REG_DI))          // DI == dsrc.data.ptr
	movq(b, aregOff(x86.REG_SP, 0x38), reg(x86.REG_AX))          // AX == dst.data.ptr
	vxorps(b, reg(x86.REG_X0), reg(x86.REG_X0), reg(x86.REG_X0)) // X0 (and Y0) == 0

	// If we have more than one chunk, keep a iteration counter
	if bigChunks > 1 {
		movq(b, constant(bigChunks), reg(x86.REG_CX))
	}

	// if we have at least maxChunkSize, write the big loop
	var bigStart *obj.Prog
	if bigChunks > 0 {
		bigStart = compileReLUMaskChunk(reluMaskChunkSize, b)

		if bigChunks > 1 || extra > 0 {
			// if we are looping or there is an extra chunk, move the pointer
			addq(b, constant(8*reluMaskChunkSize), reg(x86.REG_SI))
			addq(b, constant(8*reluMaskChunkSize), reg(x86.REG_DI))
			addq(b, constant(8*reluMaskChunkSize), reg(x86.REG_AX))
		}
		if bigChunks > 1 {
			decq(b, reg(x86.REG_CX))
			jnz(b, bigStart)
		}
	}
	if extra > 0 {
		compileReLUMaskChunk(extra, b)
	}
	ret(b)

	var f func([]float64, []float64, []float64)

	makeFunc(&f, b.Assemble())

	return f
}

func compileReLUMaskChunk(N int, b *asm.Builder) *obj.Prog {
	var first *obj.Prog
	for i := 0; i+reluMaskBlockSize <= N; i += reluMaskBlockSize {
		p := frelumaskload(b, reluMaskBlockSize, x86.REG_SI, x86.REG_DI, i, 1, f128)
		if i == 0 {
			first = p
		}
		fstore(b, reluMaskBlockSize, x86.REG_AX, i, 1, f128)
	}
	if n := N % reluMaskBlockSize; n != 0 {
		p := frelumaskload(b, n, x86.REG_SI, x86.REG_DI, N-n, 1, f128)
		if n == N {
			first = p
		}
		fstore(b, n, x86.REG_AX, N-n, 1, f128)
	}
	return first
}
