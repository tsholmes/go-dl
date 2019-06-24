package tensor

// outputs all dependent tensors in evaluation order
func CollectForward(outputs []Tensor) []Tensor {
	graph := make(map[int64]Tensor)
	forward := make(map[int64][]int64)

	fwork := []int64{}

	// build the graph backwards to collect all tensors and forward links
	work := append([]Tensor{}, outputs...)
	for len(work) > 0 {
		var t Tensor
		t, work = work[0], work[1:]

		if _, ok := graph[t.ID()]; ok {
			continue
		}
		graph[t.ID()] = t

		// Tensors without inputs are starting points
		if len(t.Inputs()) == 0 {
			fwork = append(fwork, t.ID())
		}

		for _, input := range t.Inputs() {
			forward[input.ID()] = append(forward[input.ID()], t.ID())
		}

		work = append(work, t.Inputs()...)
	}

	eval := []Tensor{}
	seen := map[int64]bool{}

	// walk forwards through the graph to get evaluation order
	for len(fwork) > 0 {
		var id int64
		id, fwork = fwork[0], fwork[1:]

		if seen[id] {
			continue
		}
		// make sure we have already seen the inputs (otherwise we'll catch it later)
		ready := true
		for _, t := range graph[id].Inputs() {
			if !seen[t.ID()] {
				ready = false
			}
		}
		if !ready {
			continue
		}

		seen[id] = true

		eval = append(eval, graph[id])
		fwork = append(fwork, forward[id]...)
	}

	return eval
}

// outputs all dependent tensors in reverse order
func CollectBackward(outputs []Tensor) []Tensor {
	graph := make(map[int64]Tensor)
	forward := make(map[int64][]int64)

	// build the graph backwards to collect all tensors and forward links
	work := append([]Tensor{}, outputs...)
	for len(work) > 0 {
		var t Tensor
		t, work = work[0], work[1:]

		if _, ok := graph[t.ID()]; ok {
			continue
		}
		graph[t.ID()] = t

		for _, input := range t.Inputs() {
			forward[input.ID()] = append(forward[input.ID()], t.ID())
		}

		work = append(work, t.Inputs()...)
	}

	eval := []Tensor{}
	seen := map[int64]bool{}

	// walk backwards through the graph so that all upstream tensors are after their outputs
	work = append(work, outputs...)
	for len(work) > 0 {
		var t Tensor
		t, work = work[0], work[1:]

		if seen[t.ID()] {
			continue
		}
		// make sure we have already seen all the outputs (otherwise we'll catch it later)
		ready := true
		for _, t2 := range forward[t.ID()] {
			if !seen[t2] {
				ready = false
			}
		}
		if !ready {
			continue
		}

		seen[t.ID()] = true

		eval = append(eval, t)
		work = append(work, t.Inputs()...)
	}

	return eval
}
