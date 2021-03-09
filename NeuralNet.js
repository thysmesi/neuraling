const _ = require('lodash')

class NeuralNet {
    constructor(inputs, outputs, sizeOfHiddenLayer, numberOfHiddenLayers) {
        this.fitness = 0
        this.weights = {}
        this.bias = {}
        this.inputs = inputs
        this.outputs = outputs
        this.numberOfHiddenLayers = numberOfHiddenLayers
        this.sizeOfHiddenLayer = sizeOfHiddenLayer

        // ----- Setup Net Values With Random ----- //
        for (let index = 0; index < inputs.length; index++) {
            this.weights[inputs[index]] = {}
            for (let neuron = 0; neuron < sizeOfHiddenLayer; neuron++) {
                this.weights[inputs[index]][`.h${0}_${neuron}`] = Math.random()
            }
        }
        for (let layer = 0; layer < numberOfHiddenLayers; layer++) {
            for (let neuron = 0; neuron < sizeOfHiddenLayer; neuron++) {
                this.weights[`.h${layer}_${neuron}`] = {}
                if (layer === numberOfHiddenLayers - 1) {
                    for (let index = 0; index < outputs.length; index++) {
                        this.weights[`.h${layer}_${neuron}`][outputs[index]] = Math.random()
                    }
                } else {
                    for (let index = 0; index < sizeOfHiddenLayer; index++) {
                        this.weights[`.h${layer}_${neuron}`][`.h${layer+1}_${index}`] = Math.random()
                    }
                }
            }
            this.bias[`.b${layer}`] = Math.random()
        }
        this.bias[`.bo`] = Math.random()
    }

    // ----- Maps The Nets Values ----- //
    mapValues(action) {
        for (const neuron in this.weights) {
            for (const weight in this.weights[neuron]) {
                this.weights[neuron][weight] = action(this.weights[neuron][weight])
            }
        }
        for (const bias in this.bias) {
            this.bias[bias] = action(this.bias[bias])
        }
    }

    // ----- Sigmoid Activation ----- //
    sigmoid(pass) {
        return 1 / (1 + Math.pow(Math.E, -pass));
    }

    // ----- Passthrough Given Input Data ----- //
    passthrough(data) {
        var calculated = {}
        var outputs = {}

        // ----- Calculate First Hidden Layer Values ----- //
        for (let neuron = 0; neuron < this.sizeOfHiddenLayer; neuron++) {
            let neuronOut = 0
            for (const inputName in data) {
                neuronOut += data[inputName] * this.weights[inputName][`.h${0}_${neuron}`]
            }
            calculated[`.h${0}_${neuron}`] = this.sigmoid(neuronOut + this.bias['.b0'])
        }

        // ----- Calculate the rest of the hidden values ----- //
        for (let layer = 1; layer < this.numberOfHiddenLayers; layer++) {
            for (let neuron = 0; neuron < this.sizeOfHiddenLayer; neuron++) {
                let neuronOut = 0
                for (let prevNeuron = 0; prevNeuron < this.sizeOfHiddenLayer; prevNeuron++) {
                    neuronOut += calculated[`.h${layer-1}_${neuron}`] * this.weights[`.h${layer-1}_${neuron}`][`.h${layer}_${neuron}`]
                }
                calculated[`.h${layer}_${neuron}`] = this.sigmoid(neuronOut + this.bias[`.b${layer}`])
            }
        }

        // ----- Calculate Outputs ----- //
        for (let index = 0; index < this.outputs.length; index++) {
            let neuronOut = 0
            for (let neuron = 0; neuron < this.sizeOfHiddenLayer; neuron++) {
                neuronOut += calculated[`.h${this.numberOfHiddenLayers-1}_${neuron}`] * this.weights[`.h${this.numberOfHiddenLayers-1}_${neuron}`][this.outputs[index]]
            }
            outputs[this.outputs[index]] = this.sigmoid(neuronOut + this.bias['.bo'])
        }

        return outputs
    }

    // ----- Mutate Function ----- //
    mutate(mutationRate) {
        var tempNet = _.cloneDeep(this)

        tempNet.mapWeights(function (weight) {
            return weight += (Math.floor(Math.random() * Math.floor(mutationRate * 2)) - mutationRate) / 100
        })

        return tempNet
    }

    // ----- Creates a Json with Nets Values ----- //
    exportNet() {
        return (JSON.stringify({
            weights: this.weights,
            bias: this.bias,
            inputs: this.inputs,
            outputs: this.outputs,
            numberOfHiddenLayers: this.numberOfHiddenLayers,
            sizeOfHiddenLayer: this.sizeOfHiddenLayer
        }))
    }

    // ----- Import Net Values From Json ----- //
    uploadNet(json) {
        var obj = JSON.parse(json)
        this.weights = obj.weights,
            this.bias = obj.bias,
            this.inputs = obj.inputs,
            this.outputs = obj.outputs,
            this.numberOfHiddenLayers = obj.numberOfHiddenLayers,
            this.sizeOfHiddenLayer = obj.sizeOfHiddenLayer
    }
}

exports.NeuralNet = NeuralNet

var a = new NeuralNet()