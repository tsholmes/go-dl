package asm

import (
	asm "github.com/twitchyliquid64/golang-asm"
	"github.com/twitchyliquid64/golang-asm/obj"
	"github.com/twitchyliquid64/golang-asm/obj/x86"
)

type regSize int

const (
	f128 regSize = iota
	f256
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

func fN(sz regSize) int {
	switch sz {
	case f128:
		return 2
	case f256:
		return 4
	default:
		panic(sz)
	}
}

func valFN(N int, res int, sz regSize) {
	if N < 1 || N > (16-res)*fN(sz) {
		panic(N)
	}
}

func fReg(i int, res int, sz regSize) obj.Addr {
	switch sz {
	case f128:
		return obj.Addr{Type: obj.TYPE_REG, Reg: fRegs128[i/2+res]}
	case f256:
		return obj.Addr{Type: obj.TYPE_REG, Reg: fRegs256[i/4+res]}
	default:
		panic(sz)
	}
}

func fRegI(i int, res int, sz regSize) int {
	return res + i/fN(sz)
}

func fCount(i int, N int, sz regSize) int {
	if N-i < fN(sz) {
		return N - i
	}
	return fN(sz)
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

func fstore(b *asm.Builder, N int, addr int16, idx int, res int, sz regSize) {
	// TODO: return if we need to later
	valFN(N, res, sz)
	for i := 0; i < N; i += fN(sz) {
		switch fCount(i, N, sz) {
		case 1:
			// single in X
			movsd(b, reg(fRegs128[fRegI(i, res, sz)]), aregOff(addr, (idx+i)*8))
		case 2:
			// double in X
			movups(b, reg(fRegs128[fRegI(i, res, sz)]), aregOff(addr, (idx+i)*8))
		case 3:
			// double+single in X
			movups(b, reg(fRegs128[fRegI(i, res, sz)]), aregOff(addr, (idx+i)*8))
			movsd(b, reg(fRegs128[fRegI(i, res, sz)+1]), aregOff(addr, (idx+i+2)*8))
		case 4:
			// normal quad in Y
			vmovups(b, reg(fRegs256[fRegI(i, res, sz)]), aregOff(addr, (idx+i)*8))
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

func freluload(b *asm.Builder, N int, addr int16, idx int, res int) (p *obj.Prog) {
	if N < 1 || N+res*4 > 64 {
		panic(N)
	}
	for i := 0; i < N; i += 4 {
		var ip *obj.Prog
		switch N - i {
		case 1:
			// single in X
			ip = vcmpsd(b, aregOff(addr, (idx+i)*8), reg(x86.REG_X0), reg(fRegs128[i/4+res]), cmpLT)
			vmaskmovps(b, aregOff(addr, (idx+i)*8), reg(fRegs128[i/4+res]), reg(fRegs128[i/4+res]))
		case 2:
			// double in X
			ip = vcmppd(b, aregOff(addr, (idx+i)*8), reg(x86.REG_X0), reg(fRegs128[i/4+res]), cmpLT)
			vmaskmovpd(b, aregOff(addr, (idx+i)*8), reg(fRegs128[i/4+res]), reg(fRegs128[i/4+res]))
		case 3:
			// double+single in X
			ip = vcmppd(b, aregOff(addr, (idx+i)*8), reg(x86.REG_X0), reg(fRegs128[i/4+res]), cmpLT)
			vcmpsd(b, aregOff(addr, (idx+i+2)*8), reg(x86.REG_X0), reg(fRegs128[i/4+res+1]), cmpLT)
			vmaskmovpd(b, aregOff(addr, (idx+i)*8), reg(fRegs128[i/4+res]), reg(fRegs128[i/4+res]))
			vmaskmovps(b, aregOff(addr, (idx+i+2)*8), reg(fRegs128[i/4+res+1]), reg(fRegs128[i/4+res+1]))
		default:
			// normal quad in Y
			ip = vcmppd(b, aregOff(addr, (idx+i)*8), reg(x86.REG_Y0), reg(fRegs256[i/4+res]), cmpLT)
			vmaskmovpd(b, aregOff(addr, (idx+i)*8), reg(fRegs256[i/4+res]), reg(fRegs256[i/4+res]))
		}
		if i == 0 {
			p = ip
		}
	}
	return p
}

func frelumaskload(b *asm.Builder, N int, addr1 int16, addr2 int16, idx int, res int, sz regSize) (p *obj.Prog) {
	valFN(N, res, sz)
	for i := 0; i < N; i += fN(sz) {
		var ip *obj.Prog
		switch fCount(i, N, sz) {
		case 1:
			// single in X
			ip = vcmpsd(b, aregOff(addr1, (idx+i)*8), reg(x86.REG_X0), reg(fRegs128[fRegI(i, res, sz)]), cmpLT)
		case 2:
			// double in X
			ip = vcmppd(b, aregOff(addr1, (idx+i)*8), reg(x86.REG_X0), reg(fRegs128[fRegI(i, res, sz)]), cmpLT)
		case 3:
			// double+single in X
			ip = vcmppd(b, aregOff(addr1, (idx+i)*8), reg(x86.REG_X0), reg(fRegs128[fRegI(i, res, sz)]), cmpLT)
			vcmpsd(b, aregOff(addr1, (idx+i+2)*8), reg(x86.REG_X0), reg(fRegs128[fRegI(i, res, sz)+1]), cmpLT)
		case 4:
			// normal quad in Y
			ip = vcmppd(b, aregOff(addr1, (idx+i)*8), reg(x86.REG_Y0), reg(fRegs256[fRegI(i, res, sz)]), cmpLT)
		}
		if i == 0 {
			p = ip
		}
	}
	for i := 0; i < N; i += fN(sz) {
		switch fCount(i, N, sz) {
		case 1:
			// single in X
			vmaskmovps(b, aregOff(addr2, (idx+i)*8), reg(fRegs128[fRegI(i, res, sz)]), reg(fRegs128[fRegI(i, res, sz)]))
		case 2:
			// double in X
			vmaskmovpd(b, aregOff(addr2, (idx+i)*8), reg(fRegs128[fRegI(i, res, sz)]), reg(fRegs128[fRegI(i, res, sz)]))
		case 3:
			// double+single in X
			vmaskmovpd(b, aregOff(addr2, (idx+i)*8), reg(fRegs128[fRegI(i, res, sz)]), reg(fRegs128[fRegI(i, res, sz)]))
			vmaskmovps(b, aregOff(addr2, (idx+i+2)*8), reg(fRegs128[fRegI(i, res, sz)+1]), reg(fRegs128[fRegI(i, res, sz)+1]))
		case 4:
			// normal quad in Y
			vmaskmovpd(b, aregOff(addr2, (idx+i)*8), reg(fRegs256[fRegI(i, res, sz)]), reg(fRegs256[fRegI(i, res, sz)]))
		}
	}
	return p
}
