package asm

import (
	asm "github.com/twitchyliquid64/golang-asm"
	"github.com/twitchyliquid64/golang-asm/obj"
	"github.com/twitchyliquid64/golang-asm/obj/x86"
)

func reg(r int16) obj.Addr {
	return obj.Addr{
		Type: obj.TYPE_REG,
		Reg:  r,
	}
}

func areg(r int16) obj.Addr {
	return obj.Addr{
		Type: obj.TYPE_MEM,
		Reg:  r,
	}
}

func aregOff(r int16, off int) obj.Addr {
	return obj.Addr{
		Type:   obj.TYPE_MEM,
		Reg:    r,
		Offset: int64(off),
	}
}

func aregIndexOff(r int16, i int16, off int) obj.Addr {
	return obj.Addr{
		Type:   obj.TYPE_MEM,
		Reg:    r,
		Index:  i,
		Offset: int64(off),
	}
}

func constant(v int) obj.Addr {
	return obj.Addr{
		Type:   obj.TYPE_CONST,
		Offset: int64(v),
	}
}

func _one(b *asm.Builder, op obj.As, to obj.Addr) *obj.Prog {
	p := b.NewProg()
	p.As = op
	p.To = to
	b.AddInstruction(p)
	return p
}

func _two(b *asm.Builder, op obj.As, from obj.Addr, to obj.Addr) *obj.Prog {
	p := b.NewProg()
	p.As = op
	p.From = from
	p.To = to
	b.AddInstruction(p)
	return p
}

func _three(b *asm.Builder, op obj.As, from obj.Addr, from2 obj.Addr, to obj.Addr) *obj.Prog {
	p := b.NewProg()
	p.As = op
	p.From = from
	p.RestArgs = []obj.Addr{from2}
	p.To = to
	b.AddInstruction(p)
	return p
}

func _four(b *asm.Builder, op obj.As, from1 obj.Addr, from2 obj.Addr, from3 obj.Addr, to obj.Addr) *obj.Prog {
	p := b.NewProg()
	p.As = op
	p.From = from1
	p.RestArgs = []obj.Addr{from2, from3}
	p.To = to
	b.AddInstruction(p)
	return p
}

func _jump(b *asm.Builder, op obj.As, to *obj.Prog) *obj.Prog {
	p := b.NewProg()
	p.As = op
	p.To.Type = obj.TYPE_BRANCH
	p.Pcond = to
	b.AddInstruction(p)
	return p
}

func movq(b *asm.Builder, from obj.Addr, to obj.Addr) *obj.Prog {
	return _two(b, x86.AMOVQ, from, to)
}

func movsd(b *asm.Builder, from obj.Addr, to obj.Addr) *obj.Prog {
	return _two(b, x86.AMOVSD, from, to)
}

func movddup(b *asm.Builder, from obj.Addr, to obj.Addr) *obj.Prog {
	return _two(b, x86.AMOVDDUP, from, to)
}

func vbroadcastsd(b *asm.Builder, from obj.Addr, to obj.Addr) *obj.Prog {
	return _two(b, x86.AVBROADCASTSD, from, to)
}

func movups(b *asm.Builder, from obj.Addr, to obj.Addr) *obj.Prog {
	return _two(b, x86.AMOVUPS, from, to)
}

func vmovups(b *asm.Builder, from obj.Addr, to obj.Addr) *obj.Prog {
	return _two(b, x86.AVMOVUPS, from, to)
}

func xorl(b *asm.Builder, from obj.Addr, to obj.Addr) *obj.Prog {
	return _two(b, x86.AXORL, from, to)
}

func addq(b *asm.Builder, from obj.Addr, to obj.Addr) *obj.Prog {
	return _two(b, x86.AADDQ, from, to)
}

func mulsd(b *asm.Builder, from obj.Addr, to obj.Addr) *obj.Prog {
	return _two(b, x86.AMULSD, from, to)
}

func mulpd(b *asm.Builder, from obj.Addr, to obj.Addr) *obj.Prog {
	return _two(b, x86.AMULPD, from, to)
}

func vmulpd(b *asm.Builder, from obj.Addr, to obj.Addr) *obj.Prog {
	return _three(b, x86.AVMULPD, from, to, to)
}

func vxorps(b *asm.Builder, from1 obj.Addr, from2 obj.Addr, to obj.Addr) *obj.Prog {
	return _three(b, x86.AVXORPS, from1, from2, to)
}

func vmaskmovps(b *asm.Builder, from obj.Addr, from2 obj.Addr, to obj.Addr) *obj.Prog {
	return _three(b, x86.AVMASKMOVPS, from, from2, to)
}

func vmaskmovpd(b *asm.Builder, from obj.Addr, from2 obj.Addr, to obj.Addr) *obj.Prog {
	return _three(b, x86.AVMASKMOVPD, from, from2, to)
}

func vcmpsd(b *asm.Builder, from obj.Addr, from2 obj.Addr, to obj.Addr, op byte) *obj.Prog {
	return _four(b, x86.AVCMPSD, from, from2, obj.Addr{Type: obj.TYPE_CONST, Offset: int64(op)}, to)
}

func vcmppd(b *asm.Builder, from obj.Addr, from2 obj.Addr, to obj.Addr, op byte) *obj.Prog {
	return _four(b, x86.AVCMPPD, obj.Addr{Type: obj.TYPE_CONST, Offset: int64(op)}, from, from2, to)
}

func incq(b *asm.Builder, to obj.Addr) *obj.Prog {
	return _one(b, x86.AINCQ, to)
}

func decq(b *asm.Builder, to obj.Addr) *obj.Prog {
	return _one(b, x86.ADECQ, to)
}

func jnz(b *asm.Builder, to *obj.Prog) *obj.Prog {
	return _jump(b, x86.AJNE, to)
}

func ret(b *asm.Builder) {
	p := b.NewProg()
	p.As = obj.ARET
	b.AddInstruction(p)
}

const (
	cmpLT = 0x01
	cmpGT = 0x0E
)
