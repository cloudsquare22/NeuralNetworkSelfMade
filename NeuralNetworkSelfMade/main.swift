//
//  main.swift
//  NeuralNetworkSelfMade
//
//  Created by Shin Inaba on 2019/01/09.
//  Copyright © 2019 shi-n. All rights reserved.
//

import Foundation

// 入力層、隠れ層、出力層のノード数
let input_nodes = 3
let hidden_nodes = 3
let output_nodes = 3

// 学習率 = 0.3
let learning_rate = 0.3

let n = neuralNetwork(inputnodes: input_nodes, hidedennodes: hidden_nodes, outputnodes: output_nodes, learningrate: learning_rate)

let result = n.query(inputs_list: [1.0, 0.5, -1.5])
print(result.description())
