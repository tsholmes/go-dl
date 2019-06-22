package main

import (
	"fmt"

	"github.com/tsholmes/go-dl/calc"
)

func main() {
	a := calc.Ones([]int{2, 2, 3})
	b := calc.Ones([]int{2, 2, 3})
	c := a.Add(b)
	d := c.Add(b)
	e := c.Div(d)
	fmt.Printf("%s\n", e.String())
}
