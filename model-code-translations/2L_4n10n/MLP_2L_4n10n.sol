// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import { SD59x18, convert, sd } from "../lib/prb-math/src/SD59x18.sol";

contract GPT_MLP_2L_4n10n_one_image_view {


    // Storage for model parameters
    int256[][] public weights1; // Weights for the first layer (input to hidden)
    int256[][] public weights2; // Weights for the second layer (hidden to output)
    int256[] public biases1;    // Biases for the first layer
    int256[] public biases2;    // Biases for the second layer

    // Function to add weights in batches
    function addWeights1(int256[][] memory _weights1) public {
        for (uint i = 0; i < _weights1.length; i++) {
            weights1.push(_weights1[i]);
        }
    }

    function addWeights2(int256[][] memory _weights2) public {
        for (uint i = 0; i < _weights2.length; i++) {
            weights2.push(_weights2[i]);
        }
    }

    // Function to add biases in batches
    function addBiases1(int256[] memory _biases1) public {
        for (uint i = 0; i < _biases1.length; i++) {
            biases1.push(_biases1[i]);
        }
    }

    function addBiases2(int256[] memory _biases2) public {
        for (uint i = 0; i < _biases2.length; i++) {
            biases2.push(_biases2[i]);
        }
    }

    // Helper function to perform matrix multiplication
    function matMul(SD59x18[][] memory A, SD59x18[] memory B) internal pure returns (SD59x18[] memory) {
        SD59x18[] memory C = new SD59x18[](A.length);
        for (uint i = 0; i < A.length; i++) {
            SD59x18 sum = sd(0);  // Initialize the sum as SD59x18
            for (uint j = 0; j < B.length; j++) {
                sum = sum + A[i][j].mul(B[j]);  // Multiply wrapped values
            }
            C[i] = sum;
        }
        return C;
    }

    // Helper function to apply ReLU activation
    function relu(SD59x18[] memory x) internal pure returns (SD59x18[] memory) {
        SD59x18[] memory result = new SD59x18[](x.length);
        for (uint i = 0; i < x.length; i++) {
            result[i] = x[i] > sd(0) ? x[i] : sd(0);  // ReLU activation
        }
        return result;
    }

    // Function to classify a single image and return the predicted label
    function classifySingleImage(int256[] calldata inputInt) public view returns (int256) {

        // Convert input data to SD59x18
        SD59x18[] memory input = new SD59x18[](inputInt.length);
        for (uint k = 0; k < inputInt.length; k++) {
            input[k] = sd(inputInt[k]);
        }

        // Convert weights1 and biases1 to SD59x18
        SD59x18[][] memory weights1Fixed = new SD59x18[][](weights1.length);
        for (uint j = 0; j < weights1.length; j++) {
            weights1Fixed[j] = new SD59x18[](weights1[j].length);
            for (uint k = 0; k < weights1[j].length; k++) {
                weights1Fixed[j][k] = sd(weights1[j][k]);
            }
        }

        SD59x18[] memory biases1Fixed = new SD59x18[](biases1.length);
        for (uint j = 0; j < biases1.length; j++) {
            biases1Fixed[j] = sd(biases1[j]);
        }

        // Layer 1 (Input to Hidden)
        SD59x18[] memory hidden = relu(matMul(weights1Fixed, input));
        for (uint j = 0; j < hidden.length; j++) {
            hidden[j] = hidden[j] + biases1Fixed[j];  // Add biases
        }

        // Convert weights2 and biases2 to SD59x18
        SD59x18[][] memory weights2Fixed = new SD59x18[][](weights2.length);
        for (uint j = 0; j < weights2.length; j++) {
            weights2Fixed[j] = new SD59x18[](weights2[j].length);
            for (uint k = 0; k < weights2[j].length; k++) {
                weights2Fixed[j][k] = sd(weights2[j][k]);
            }
        }

        SD59x18[] memory biases2Fixed = new SD59x18[](biases2.length);
        for (uint j = 0; j < biases2.length; j++) {
            biases2Fixed[j] = sd(biases2[j]);
        }

        // Layer 2 (Hidden to Output)
        SD59x18[] memory output = matMul(weights2Fixed, hidden);
        for (uint j = 0; j < output.length; j++) {
            output[j] = output[j] + biases2Fixed[j];  // Add biases
        }

        // Find the index with the maximum value in the output (predicted label)
        SD59x18 maxVal = output[0];
        int256 predictedLabel = 0;

        for (uint j = 1; j < output.length; j++) {
            if (output[j] > maxVal) {
                maxVal = output[j];
                predictedLabel = int256(j);
            }
        }
        return predictedLabel;
    }

}
