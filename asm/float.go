package asm

import (
	asm "github.com/twitchyliquid64/golang-asm"
	"github.com/twitchyliquid64/golang-asm/obj"
	"github.com/twitchyliquid64/golang-asm/obj/x86"
)

var fRegs128 = [...]int16{
	x86.REG_X0, x86.REG_X1,
	x86.REG_X2, x86.REG_X3,
	x86.REG_X4, x86.REG_X5,
	x86.REG_X6, x86.REG_X7,
	x86.REG_X8, x86.REG_X9,
	x86.REG_X10, x86.REG_X11,
	x86.REG_X12, x86.REG_X13,
	x86.REG_X14, x86.REG_X15,
}

var fRegs256 = [...]int16{
	x86.REG_Y0, x86.REG_Y1,
	x86.REG_Y2, x86.REG_Y3,
	x86.REG_Y4, x86.REG_Y5,
	x86.REG_Y6, x86.REG_Y7,
	x86.REG_Y8, x86.REG_Y9,
	x86.REG_Y10, x86.REG_Y11,
	x86.REG_Y12, x86.REG_Y13,
	x86.REG_Y14, x86.REG_Y15,
}

func fload1(b *asm.Builder, N int, addr int16, idx int, res int) (p *obj.Prog) {
	if N < 1 || N+res*4 > 64 {
		panic(N)
	}
	for i := 0; i < N; i += 4 {
		var ip *obj.Prog
		switch N - i {
		case 1:
			// single in X
			ip = movsd(b, aregOff(addr, (idx+i)*8), reg(fRegs128[i/4+res]))
		case 2:
			// double in X
			ip = movups(b, aregOff(addr, (idx+i)*8), reg(fRegs128[i/4+res]))
		case 3:
			// double+single in X
			ip = movups(b, aregOff(addr, (idx+i)*8), reg(fRegs128[i/4+res]))
			movsd(b, aregOff(addr, (idx+i+2)*8), reg(fRegs128[i/4+1+res]))
		default:
			// normal quad in Y
			ip = vmovups(b, aregOff(addr, (idx+i)*8), reg(fRegs256[i/4+res]))
		}
		if i == 0 {
			p = ip
		}
	}
	return p
}

func fstore1(b *asm.Builder, N int, addr int16, idx int, res int) {
	// TODO: return if we need to later
	if N < 1 || N+res*4 > 64 {
		panic(N)
	}
	for i := 0; i < N; i += 4 {
		switch N - i {
		case 1:
			// single in X
			movsd(b, reg(fRegs128[i/4+res]), aregOff(addr, (idx+i)*8))
		case 2:
			// double in X
			movups(b, reg(fRegs128[i/4+res]), aregOff(addr, (idx+i)*8))
		case 3:
			// double+single in X
			movups(b, reg(fRegs128[i/4+res]), aregOff(addr, (idx+i)*8))
			movsd(b, reg(fRegs128[i/4+1+res]), aregOff(addr, (idx+i+2)*8))
		default:
			// normal quad in Y
			vmovups(b, reg(fRegs256[i/4+res]), aregOff(addr, (idx+i)*8))
		}
	}
}

func fmul1(b *asm.Builder, N int, addr int16, idx int, res int) {
	// TODO: return if we need to later
	if N < 1 || N+res*4 > 64 {
		panic(N)
	}
	for i := 0; i < N; i += 4 {
		switch N - i {
		case 1:
			// single in X
			mulsd(b, reg(x86.REG_X0), reg(fRegs128[i/4+res]))
		case 2:
			// double in X
			mulpd(b, reg(x86.REG_X0), reg(fRegs128[i/4+res]))
		case 3:
			// double+single in X
			mulpd(b, reg(x86.REG_X0), reg(fRegs128[i/4+res]))
			mulsd(b, reg(x86.REG_X0), reg(fRegs128[i/4+1+res]))
		default:
			// normal quad in Y
			vmulpd(b, reg(x86.REG_Y0), reg(fRegs256[i/4+res]))
		}
	}
}

func frelumask(b *asm.Builder, N int, addr int16, idx int, res int) (p *obj.Prog) {
	if N < 1 || N+res*4 > 64 {
		panic(N)
	}
	for i := 0; i < N; i += 4 {
		var ip *obj.Prog
		switch N - i {
		case 1:
			// single in X
			ip = vcmpsd(b, aregOff(addr, (idx+i)*8), reg(x86.REG_X0), reg(fRegs128[i/4+res]), cmpGT)
		case 2:
			// double in X
			ip = vcmppd(b, aregOff(addr, (idx+i)*8), reg(x86.REG_X0), reg(fRegs128[i/4+res]), cmpGT)
		case 3:
			// double+single in X
			ip = vcmppd(b, aregOff(addr, (idx+i)*8), reg(x86.REG_X0), reg(fRegs128[i/4+res]), cmpGT)
			vcmpsd(b, aregOff(addr, (idx+i+2)*8), reg(x86.REG_X0), reg(fRegs128[i/4+res+1]), cmpGT)
		default:
			// normal quad in Y
			ip = vcmppd(b, aregOff(addr, (idx+i)*8), reg(x86.REG_Y0), reg(fRegs256[i/4+res]), cmpGT)
		}
		if i == 0 {
			p = ip
		}
	}
	return p
}

func fmaskstore(b *asm.Builder, N int, addr int16, idx int, res int) {
	if N < 1 || N+res*4 > 64 {
		panic(N)
	}
	for i := 0; i < N; i += 4 {
		switch N - i {
		case 1:
			// single in X
			vmaskmovps(b, reg(x86.REG_X0), reg(fRegs128[i/4+res]), aregOff(addr, (idx+i)*8))
		case 2:
			// double in X
			vmaskmovpd(b, reg(x86.REG_X0), reg(fRegs128[i/4+res]), aregOff(addr, (idx+i)*8))
		case 3:
			// double+single in X
			vmaskmovpd(b, reg(x86.REG_X0), reg(fRegs128[i/4+res]), aregOff(addr, (idx+i)*8))
			vmaskmovps(b, reg(x86.REG_X0), reg(fRegs128[i/4+res+1]), aregOff(addr, (idx+i+2)*8))
		default:
			// normal quad in Y
			vmaskmovpd(b, reg(x86.REG_Y0), reg(fRegs256[i/4+res]), aregOff(addr, (idx+i)*8))
		}
	}
}
