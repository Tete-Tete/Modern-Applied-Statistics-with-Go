package main

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestRidgeRegression(t *testing.T) {
	// Using the same dataset from main package, just brief it.
	X := mat.NewDense(4, 2, []float64{
		1, 2,
		3, 4,
		5, 6,
		7, 8,
	})
	y := mat.NewVecDense(4, []float64{1, 2, 3, 4})

	lambda := 0.1
	beta := ridgeRegression(X, y, lambda)

	if beta == nil {
		t.Errorf("Expected non-nil result, got nil")
	}

	// Expected coefficients (this is just a placeholder, replace with actual expected values, you can change it for yourself.)
	expected := mat.NewVecDense(2, []float64{0.1, 0.2})

	for i := 0; i < beta.Len(); i++ {
		if beta.AtVec(i) != expected.AtVec(i) {
			t.Errorf("Coefficient at index %d: you should have  %v, you got %v", i, expected.AtVec(i), beta.AtVec(i))
		}
	}
}
