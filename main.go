package main

import (
	"bufio"
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"

	"gonum.org/v1/gonum/mat"
)

func ridgeRegression(X *mat.Dense, y *mat.VecDense, lambda float64) *mat.VecDense {
	r, c := X.Dims()

	// Create Xt (transpose of X)
	Xt := mat.NewDense(c, r, nil)
	Xt.Copy(X.T())

	// Compute Xt * X
	XtX := mat.NewDense(c, c, nil)
	XtX.Mul(Xt, X)

	// Add lambda * I to XtX for regularization
	for i := 0; i < c; i++ {
		XtX.Set(i, i, XtX.At(i, i)+lambda)
	}

	// Compute Xt * y
	Xty := mat.NewVecDense(c, nil)
	Xty.MulVec(Xt, y)

	// Solve (XtX + lambda * I) * beta = Xty for beta
	var beta mat.VecDense
	err := beta.SolveVec(XtX, Xty)
	if err != nil {
		fmt.Printf("Error solving linear system: %v\n", err)
		return nil
	}

	return &beta
}

func main() {
	reader := bufio.NewReader(os.Stdin)
	fmt.Print("Enter lambda value: ")
	input, _ := reader.ReadString('\n')
	input = strings.TrimSpace(input)
	lambda, err := strconv.ParseFloat(input, 64)
	if err != nil {
		fmt.Println("Invalid lambda value. Please provide a valid number.")
		return
	}

	// Measure memory usage before
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	fmt.Printf("Memory Usage Before: %v KB\n", m.Alloc/1024)

	// Measure processing time
	start := time.Now()

	// Define the same dataset From R (First 4 rows from dataset )
	X := mat.NewDense(4, 5, []float64{
		6, 160, 110, 3.9, 2.62,
		6, 160, 110, 3.9, 2.875,
		4, 108, 93, 3.85, 2.32,
		6, 258, 110, 3.08, 3.215,
	})
	y := mat.NewVecDense(4, []float64{21, 21, 22.8, 21.4})

	beta := ridgeRegression(X, y, lambda)

	duration := time.Since(start)
	fmt.Printf("Processing Time: %v\n", duration)

	// Measure memory usage after
	runtime.ReadMemStats(&m)
	fmt.Printf("Memory Usage After: %v KB\n", m.Alloc/1024)

	if beta != nil {
		fmt.Println("Ridge Regression Coefficients:")
		fmt.Println(mat.Formatted(beta, mat.Prefix(" "), mat.Squeeze()))
	}

	// Print memory usage
	outputFile, err := os.Create("ridge_regression_output.txt")
	if err != nil {
		fmt.Println("Error creating output file:", err)
		return
	}
	defer outputFile.Close()

	fmt.Fprintf(outputFile, "Lambda Value: %v\n", lambda)
	fmt.Fprintf(outputFile, "Processing Time: %v\n", duration)
	fmt.Fprintf(outputFile, "Memory Usage Before: %v KB\n", m.Alloc/1024)
	fmt.Fprintf(outputFile, "Memory Usage After: %v KB\n", m.Alloc/1024)

	if beta != nil {
		fmt.Fprintf(outputFile, "Ridge Regression Coefficients:\n")
		fmt.Fprintf(outputFile, "%v\n", mat.Formatted(beta, mat.Prefix(" "), mat.Squeeze()))
	}

	fmt.Println("Results have been saved to ridge_regression_output.txt")

	fmt.Println("Press 'Enter' to exit...")
	bufio.NewReader(os.Stdin).ReadString('\n')
}
