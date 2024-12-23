// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import { SD59x18, convert, sd } from "../lib/prb-math/src/SD59x18.sol";

contract GPT_MLP_3L_4n5n10n_one_image_view {

    // Storage for model parameters
    int256[][] public weights1; // Weights for the first layer (input to first hidden)
    int256[][] public weights2; // Weights for the second layer (first hidden to second hidden)
    int256[][] public weights3; // Weights for the third layer (second hidden to output)
    int256[] public biases1;    // Biases for the first layer
    int256[] public biases2;    // Biases for the second layer
    int256[] public biases3;    // Biases for the third layer

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

    function addWeights3(int256[][] memory _weights3) public {
        for (uint i = 0; i < _weights3.length; i++) {
            weights3.push(_weights3[i]);
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

    function addBiases3(int256[] memory _biases3) public {
        for (uint i = 0; i < _biases3.length; i++) {
            biases3.push(_biases3[i]);
        }
    }

    // Helper function to perform matrix multiplication
    function matMul(SD59x18[][] memory A, SD59x18[] memory B) internal pure returns (SD59x18[] memory) {
        SD59x18[] memory C = new SD59x18[](A.length);
        for (uint i = 0; i < A.length; i++) {
            SD59x18 sum = sd(0);
            for (uint j = 0; j < B.length; j++) {
                sum = sum + A[i][j].mul(B[j]);
            }
            C[i] = sum;
        }
        return C;
    }

    // Helper function to apply ReLU activation
    function relu(SD59x18[] memory x) internal pure returns (SD59x18[] memory) {
        SD59x18[] memory result = new SD59x18[](x.length);
        for (uint i = 0; i < x.length; i++) {
            result[i] = x[i] > sd(0) ? x[i] : sd(0);
        }
        return result;
    }

    // Function to classify a single image and return the predicted label
    function classifySingleImage(int256[] memory inputInt) public view returns (int256) {

        // Convert input data to SD59x18
        SD59x18[] memory input = new SD59x18[](inputInt.length);
        for (uint k = 0; k < inputInt.length; k++) {
            input[k] = sd(inputInt[k]);
        }

        // Layer 1: Input to First Hidden
        SD59x18[] memory hidden1 = matMul(convertWeights(weights1), input);
        for (uint j = 0; j < hidden1.length; j++) {
            hidden1[j] = hidden1[j] + sd(biases1[j]);
        }
        hidden1 = relu(hidden1);

        // Layer 2: First Hidden to Second Hidden
        SD59x18[] memory hidden2 = matMul(convertWeights(weights2), hidden1);
        for (uint j = 0; j < hidden2.length; j++) {
            hidden2[j] = hidden2[j] + sd(biases2[j]);
        }
        hidden2 = relu(hidden2);

        // Layer 3: Second Hidden to Output
        SD59x18[] memory output = matMul(convertWeights(weights3), hidden2);
        for (uint j = 0; j < output.length; j++) {
            output[j] = output[j] + sd(biases3[j]);
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

    // Helper function to convert weights to SD59x18 format
    function convertWeights(int256[][] storage weights) internal view returns (SD59x18[][] memory) {
        SD59x18[][] memory weightsFixed = new SD59x18[][](weights.length);
        for (uint i = 0; i < weights.length; i++) {
            weightsFixed[i] = new SD59x18[](weights[i].length);
            for (uint j = 0; j < weights[i].length; j++) {
                weightsFixed[i][j] = sd(weights[i][j]);
            }
        }
        return weightsFixed;
    }
}
