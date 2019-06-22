package main

import (
	"fmt"

	"github.com/tsholmes/go-dl/tensor"
)

func main() {
	a := tensor.Input([]int{2, 2, 3})
	b := tensor.Input([]int{2, 2, 3})
	c := tensor.Input([]int{2, 2, 1})

	x := tensor.Add(a, b)
	x = tensor.Concat(2, x, c)

	fmt.Println(x.Shape())
}
