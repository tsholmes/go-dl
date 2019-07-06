package asm

import (
	"reflect"
	"syscall"
	"unsafe"
)

const pageSize = 4096

func makeFunc(f interface{}, data []byte) {
	pageCount := (len(data) / pageSize) + 1
	emem, err := syscall.Mmap(
		-1,
		0,
		pageCount*pageSize,
		syscall.PROT_READ|syscall.PROT_WRITE,
		syscall.MAP_PRIVATE|syscall.MAP_ANON,
	)

	if err != nil {
		panic(err)
	}
	copy(emem, data)

	// It's WAYYY faster to write without exec, and then swap to ro/exec
	if err := syscall.Mprotect(emem, syscall.PROT_READ|syscall.PROT_EXEC); err != nil {
		panic(err)
	}

	eslice := *(*reflect.SliceHeader)(unsafe.Pointer(&emem))

	fptr := (**uintptr)(unsafe.Pointer(reflect.ValueOf(f).Pointer()))
	*fptr = &eslice.Data
}
