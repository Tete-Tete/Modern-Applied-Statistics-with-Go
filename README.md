# Modern Applied Statistics with Go
This repository contains a Go implementation for Ridge Regression analysis, comparing performance with R. The main goal of this project is to evaluate the effectiveness of Go for regularization, specifically Ridge Regression, and its potential cloud computing cost savings.

## Overview
This project demonstrates the implementation of Ridge Regression using both R and Go programming languages. It aims to evaluate the performance differences between R and Go in applying regularization, particularly Ridge Regression, and to assess potential savings in cloud computing costs by switching from R to Go.

## Features
- **Ridge Regression Implementation in R**: Uses the `glmnet` package, which is commonly used for regularization tasks such as LASSO and Ridge Regression. The link is https://cran.r-project.org/web/packages/glmnet/.
- **Ridge Regression Implementation in Go**: Utilizes the `gonum` package to perform Ridge Regression.
- **Comparison between R and Go**: Analyze the performance differences between R and Go, focusing on execution speed, resource efficiency, and cost savings.v
- **Unit Testing**: Includes testing of the Go implementation to verify the accuracy of the regression results.

## Requirements
- **R**: Install R and the `glmnet` package for Ridge Regression.
- **Go**:Go programming language installed (version 1.16 or higher recommended).
- **Go Dependencies**: Install the required package, `gonum`:
  ```sh
  go get gonum.org/v1/gonum/mat
  ```

## Installation From Git and Setup
### Step 1: Clone the Repository
Clone this repository to your local machine:
```sh
git clone <https://github.com/Tete-Tete/Modern-Applied-Statistics-with-Go.git>
```

### Step 2: Run the Application
To build and run the Go application, run the following commands in your terminal:
```sh
go build -o ridge_regression.exe main.go
./regression
```
This will create an executable file named `ridge_regression` in your current directory.

## Running the Project
To run the project and see the regression analysis results:
```sh
go run main.go
```
The program will prompt you to enter the lambda value for the Ridge Regression model.

## Testing
### Running Tests
The Go implementation includes a unit test (regression_test.go) to verify the accuracy of the regression calculations. To run the tests, use:
```sh
go test
```
This command will execute all the test cases in the project, ensuring that the Ridge Regression calculations are correct.

## Dataset
The dataset used for this Ridge Regression example is a simplified version of a real dataset, defined directly in the code. You can modify `main.go` to read from a file or a database if needed.

## Output
The output includes the Ridge Regression coefficients (slope and intercept), displayed with appropriate formatting to understand the relationships between the variables.

For example:
```
Ridge Regression Coefficients:
[0.12345, 0.67890, ...]
...
```
## Efforts in Finding R and Go Packages
- **R Package Selection**:For the R implementation, the `glmnet` package was chosen because it is a well-established package for regularization techniques such as Ridge Regression and LASSO. It provides a comprehensive set of tools for fitting regularized linear models, and its integration with R makes it straightforward to use for both modeling and extracting coefficients.
- **Go Package Selection**:For the Go implementation, the `gonum` package was selected. `gonum`is a powerful numerical computation library for Go, providing robust support for matrix operations, which are essential for implementing Ridge Regression. The decision to use `gonum` was based on its performance, community support, and ability to handle the necessary linear algebra computations efficiently.

## Performance Considerations
- **Use of Go**:For large datasets and cloud environments, Go offers significant performance benefits over R, especially when utilizing multi-core processors.
- **Cloud Cost Savings**:By using Go, cloud computing costs can be reduced due to its efficient use of resources.

## Compare Result
- **R**： output show below. Time unit is microseconds
```
Unit: microseconds
        expr   min       lq     mean  median       uq      max neval
 ridge_model 666.2 710.2515 798.4949 756.301 873.7505 1170.302   100
```
```
"Memory usage after fitting model: 33024 bytes"
```
- **Go**: There will be a txt file if you run the project. And it will show the result.