//
//  neuralNetwork.swift
//  NeuralNetworkSelfMade
//
//  Created by Shin Inaba on 2019/01/09.
//  Copyright © 2019 shi-n. All rights reserved.
//

import Foundation

class neuralNetwork {
    var inodes:Int
    var hnodes:Int
    var onodes:Int
    var lr:Double
    var wih:Matrix
    var who:Matrix
    
    init(inputnodes:Int, hidedennodes:Int, outputnodes:Int, learningrate:Double) {
        // 入力層、隠れ層、出力層のノード数の設定
        inodes = inputnodes
        hnodes = hidedennodes
        onodes = outputnodes

        // リンクの重み行列 wihとwho
        // 行列内の重み w_i_j,ノードIから次の層のノードjへのリンクの重み
        // w11 w21
        // w12 w22 など
        wih = Matrix.rand(rows: hnodes, colums: inodes) - 0.5
        who = Matrix.rand(rows: onodes, colums: hnodes) - 0.5
        
        // 学習率の設定
        lr = learningrate
    }
    
    // ニューラルネットワークの学習
    func train(inputs_list:[Double], targets_list:[Double]) {
        // 入力リストを行列に変換
        var inputs = Matrix()
        inputs.matrix = [[Double]](repeating: [Double](repeating: 0.0, count: 1), count: inputs_list.count)
        for r in 0..<inputs_list.count - 1{
            inputs.matrix[r][0] = inputs_list[r]
        }
        var targets = Matrix()
        targets.matrix = [[Double]](repeating: [Double](repeating: 0.0, count: 1), count: targets_list.count)
        for r in 0..<targets_list.count - 1{
            targets.matrix[r][0] = targets_list[r]
        }
        
        // 隠れ層に入ってくる信号の計算
        let hidden_inputs = Matrix.dot(left: wih, right: inputs)
        // 隠れ層で結合された信号を活性化関数により出力
        let hidden_outputs = activation_function(input: hidden_inputs)

        // 出力層に入ってくる信号の計算
        let final_inputs = Matrix.dot(left: who, right: hidden_outputs)
        // 出力層で結合された信号を活性化関数により出力
        let final_outputs = activation_function(input: final_inputs)

        // 出力層の誤差 = (目標出力 - 最終出力)
        let output_errors = targets - final_outputs
        // 隠れ層の誤差おは出力層の誤差をリンクの重みの配合で分配
        let hidden_errors = Matrix.dot(left: who, right: output_errors)
        
        // 隠れ層と出力層の間のリンクの重みを更新
        who = who + lr * Matrix.dot(left: (output_errors * final_outputs) * (1.0 - final_outputs), right: Matrix.transpose(input: hidden_outputs))
        
        // 入力層と隠れ層の間のリンクの重みを更新
        wih = wih + lr * Matrix.dot(left: (hidden_errors * hidden_outputs) * (1.0 - hidden_outputs), right: Matrix.transpose(input: inputs))
    }
    
    // ニューラルネットワークへの照会
    func query(inputs_list:[Double]) -> Matrix {
        var inputs = Matrix()
        inputs.matrix = [[Double]](repeating: [Double](repeating: 0.0, count: 1), count: inputs_list.count)
        for r in 0..<inputs_list.count - 1{
            inputs.matrix[r][0] = inputs_list[r]
        }
        // 隠れ層に入ってくる信号の計算
        let hidden_inputs = Matrix.dot(left: wih, right: inputs)
        // 隠れ層で結合された信号を活性化関数により出力
        let hidden_outputs = activation_function(input: hidden_inputs)

        // 出力層に入ってくる信号の計算
        let final_inputs = Matrix.dot(left: who, right: hidden_outputs)
        // 出力層で結合された信号を活性化関数により出力
        let final_outputs = activation_function(input: final_inputs)
        
        return final_outputs
    }
    
    func activation_function(input:Matrix) -> Matrix {
        var result = Matrix()
        result.matrix = [[Double]](repeating: [Double](repeating: 0.0, count: input.matrix[0].count), count: input.matrix.count)
        for r in 0..<input.matrix.count {
            for c in 0..<input.matrix[r].count {
                // シグモイド関数 1 / (1 + exp(-(x)))
                result.matrix[r][c] = 1 / (1 + exp(-(input.matrix[r][c])))
            }
        }
        return result
    }
    
}
