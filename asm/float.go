package asm

import (
	asm "github.com/twitchyliquid64/golang-asm"
	"github.com/twitchyliquid64/golang-asm/obj"
	"github.com/twitchyliquid64/golang-asm/obj/x86"
)

var fRegs = [...]int16{
	x86.REG_X1,
	x86.REG_X2, x86.REG_X3,
	x86.REG_X4, x86.REG_X5,
	x86.REG_X6, x86.REG_X7,
	x86.REG_X8, x86.REG_X9,
	x86.REG_X10, x86.REG_X11,
	x86.REG_X12, x86.REG_X13,
	x86.REG_X14, x86.REG_X15,
}

func fload1(b *asm.Builder, N int, addr int16, idx int) (p *obj.Prog) {
	if N < 1 || N > 30 {
		panic(N)
	}
	for i := 0; i < N; i += 2 {
		f := movups
		// odd case
		if i+1 == N {
			f = movsd
		}
		ip := f(b, aregOff(addr, (idx+i)*8), reg(fRegs[i/2]))
		if i == 0 {
			p = ip
		}
	}
	return p
}

func fmul1(b *asm.Builder, N int, addr int16, idx int) {
	// TODO: return if we need to later
	if N < 1 || N > 30 {
		panic(N)
	}
	for i := 0; i < N; i += 2 {
		f := mulpd
		// odd case
		if i+1 == N {
			f = mulsd
		}
		f(b, reg(x86.REG_X0), reg(fRegs[i/2]))
	}
}

func fstore1(b *asm.Builder, N int, addr int16, idx int) {
	// TODO: return if we need to later
	if N < 1 || N > 30 {
		panic(N)
	}
	for i := 0; i < N; i += 2 {
		f := movups
		// odd case
		if i+1 == N {
			f = movsd
		}
		f(b, reg(fRegs[i/2]), aregOff(addr, (idx+i)*8))
	}
}
