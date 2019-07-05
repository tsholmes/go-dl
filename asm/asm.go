package asm

import (
	"reflect"
	"syscall"
	"unsafe"

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
const maxChunkSize = 30 * 13

func CompileMulConstant(N int) func([]float64, float64) {
	bigChunks := N / maxChunkSize
	extra := N % maxChunkSize

	b, err := asm.NewBuilder("amd64", N*3+10)
	if err != nil {
		panic(err)
	}
	movq(b, aregOff(x86.REG_SP, 0x8), reg(x86.REG_SI))     // SI == arr.data.ptr
	movddup(b, aregOff(x86.REG_SP, 0x20), reg(x86.REG_X0)) // X0 = c

	// If we have more than one chunk, keep a iteration counter
	if bigChunks > 1 {
		movq(b, constant(bigChunks), reg(x86.REG_CX))
	}

	// if we have at least maxChunkSize, write the big loop
	var bigStart *obj.Prog
	if bigChunks > 0 {
		bigStart = compileMulChunk(maxChunkSize, b)

		if bigChunks > 1 || extra > 0 {
			// if we are looping or there is an extra chunk, move the pointer
			addq(b, constant(8*maxChunkSize), reg(x86.REG_SI))
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
	blockSize := 30
	var first *obj.Prog
	// Do 4s in chunks
	for i := 0; i+blockSize <= N; i += blockSize {
		p := fload1(b, blockSize, x86.REG_SI, i)
		if i == 0 {
			first = p
		}
		fmul1(b, blockSize, x86.REG_SI, i)
		fstore1(b, blockSize, x86.REG_SI, i)
	}
	if n := N % blockSize; n != 0 {
		p := fload1(b, n, x86.REG_SI, N-n)
		if n == N {
			first = p
		}
		fmul1(b, n, x86.REG_SI, N-n)
		fstore1(b, n, x86.REG_SI, N-n)
	}
	return first
}

const pageSize = 4096

func makeFunc(f interface{}, data []byte) {
	pageCount := (len(data) / pageSize) + 1
	emem, err := syscall.Mmap(
		-1,
		0,
		pageCount*pageSize,
		syscall.PROT_READ|syscall.PROT_WRITE|syscall.PROT_EXEC,
		syscall.MAP_PRIVATE|syscall.MAP_ANON,
	)

	if err != nil {
		panic(err)
	}
	copy(emem, data)

	eslice := *(*reflect.SliceHeader)(unsafe.Pointer(&emem))

	fptr := (**uintptr)(unsafe.Pointer(reflect.ValueOf(f).Pointer()))
	*fptr = &eslice.Data
}
