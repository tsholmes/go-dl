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

const reluChunkSize = 32 * 11

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
	blockSize := 32
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
