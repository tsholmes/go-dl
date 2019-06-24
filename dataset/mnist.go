package dataset

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"os"

	"github.com/tsholmes/go-dl/calc"
)

func loadIDX(fname string) ([]int, []byte) {
	f, err := os.Open(fname)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		panic(err)
	}

	var magic int32
	if err := binary.Read(gz, binary.BigEndian, &magic); err != nil {
		panic(err)
	}

	dataType := (magic >> 8) & 0xFF
	if dataType != 8 {
		panic(fmt.Sprintf("Unknown data type %d", dataType))
	}

	axes := int(magic & 0xFF)
	shape := make([]int, axes)
	totalLen := 1
	for i := range shape {
		var axis int32
		if err := binary.Read(gz, binary.BigEndian, &axis); err != nil {
			panic(err)
		}
		shape[i] = int(axis)
		totalLen *= int(axis)
	}

	data := make([]byte, totalLen)
	if _, err := io.ReadFull(gz, data); err != nil {
		panic(err)
	}

	return shape, data
}

func bwFloat(shape []int, data []byte) calc.NDArray {
	fdata := make([]float64, len(data))
	for i, v := range data {
		fdata[i] = float64(v) / 255.0
	}
	return calc.FromRaw(shape, fdata)
}

func categorical10(shape []int, data []byte) calc.NDArray {
	shape = []int{shape[0], 10}
	fData := make([]float64, len(data)*10)
	labels := make([][]float64, 10)
	for i := range labels {
		labels[i] = make([]float64, 10)
		labels[i][i] = 1.0
	}
	for i, v := range data {
		copy(fData[i*10:], labels[v])
	}
	return calc.FromRaw(shape, fData)
}

func LoadMNIST() (XTrain, YTrain, XTest, YTest calc.NDArray) {
	return bwFloat(loadIDX("dataset/train-images-idx3-ubyte.gz")),
		categorical10(loadIDX("dataset/train-labels-idx1-ubyte.gz")),
		bwFloat(loadIDX("dataset/t10k-images-idx3-ubyte.gz")),
		categorical10(loadIDX("dataset/t10k-labels-idx1-ubyte.gz"))
}
