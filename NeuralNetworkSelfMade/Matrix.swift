//
//  Matrix.swift
//  NeuralNetworkSelfMade
//
//  Created by Shin Inaba on 2019/01/12.
//  Copyright © 2019 shi-n. All rights reserved.
//

import Foundation

/// 行列
struct Matrix {
    var matrix:[[Double]] = []

    func description() {
        for r in 0..<matrix.count {
            print("\(r) : \(matrix[r])")
        }
    }
    
    static func rand(rows:Int, colums:Int) -> Matrix {
        var createMatrix = Matrix()
        createMatrix.matrix = [[Double]](repeating: [Double](repeating: 0.0, count: rows), count: colums)
        for r in 0..<rows {
            for c in 0..<colums {
                createMatrix.matrix[r][c] = Double.random(in: 0...1)
            }
        }
        return createMatrix
    }
    
    /// 行列の内積
    /// - parameter left: 左辺(行列)
    /// - parameter right: 右辺(行列)
    /// - returns: 結果(行列)
    static func dot(left:Matrix, right:Matrix) -> Matrix {
        var result = Matrix()
        result.matrix = [[Double]](repeating: [Double](repeating: 0.0, count: right.matrix[0].count), count: right.matrix.count)
        for rresult in 0..<result.matrix.count {
            for cresult in 0..<result.matrix[rresult].count {
                var sum = 0.0
                for rright in 0..<right.matrix.count {
                    sum = sum + (right.matrix[rright][cresult] * left.matrix[rresult][rright])
                }
                result.matrix[rresult][cresult] = sum
            }
        }
        return result
    }
    
    /// 行列の転置
    /// - parameter input: 行列
    /// - returns: 結果(行列)
    static func transpose(input:Matrix) -> Matrix {
        var result = Matrix()
        result.matrix = [[Double]](repeating: [Double](repeating: 0.0, count: input.matrix.count), count: input.matrix[0].count)
        for r in 0..<input.matrix.count {
            for c in 0..<input.matrix[r].count {
                result.matrix[c][r] = input.matrix[r][c]
            }
        }
        return result
    }
    
}

func - (left: Matrix, right: Double) -> Matrix {
    var result = Matrix()
    result.matrix = left.matrix
    for r in 0..<result.matrix.count {
        for c in 0..<result.matrix[r].count {
            result.matrix[r][c] = result.matrix[r][c] - right
        }
    }
    return result
}

func - (left: Double, right: Matrix) -> Matrix {
    var result = Matrix()
    result.matrix = right.matrix
    for r in 0..<result.matrix.count {
        for c in 0..<result.matrix[r].count {
            result.matrix[r][c] = left - result.matrix[r][c]
        }
    }
    return result
}

func - (left: Matrix, right: Matrix) -> Matrix {
    var result = Matrix()
    result.matrix = left.matrix
    for r in 0..<result.matrix.count {
        for c in 0..<result.matrix[r].count {
            result.matrix[r][c] = result.matrix[r][c] - right.matrix[r][c]
        }
    }
    return result
}

func + (left: Matrix, right: Matrix) -> Matrix {
    var result = Matrix()
    result.matrix = left.matrix
    for r in 0..<result.matrix.count {
        for c in 0..<result.matrix[r].count {
            result.matrix[r][c] = result.matrix[r][c] + right.matrix[r][c]
        }
    }
    return result
}

/// 行列の要素同士の掛け算
/// - parameter left: 左辺(行列)
/// - parameter right: 右辺(行列)
/// - returns: 結果(行列)
func * (left: Matrix, right: Matrix) -> Matrix {
    var result = Matrix()
    result.matrix = left.matrix
    for r in 0..<result.matrix.count {
        for c in 0..<result.matrix[r].count {
            result.matrix[r][c] = result.matrix[r][c] * right.matrix[r][c]
        }
    }
    return result
}

/// 行列の要素同士の掛け算
/// - parameter left: 左辺(数値)
/// - parameter right: 右辺(行列)
/// - returns: 結果(行列)
func * (left: Double, right: Matrix) -> Matrix {
    var result = Matrix()
    result.matrix = right.matrix
    for r in 0..<result.matrix.count {
        for c in 0..<result.matrix[r].count {
            result.matrix[r][c] = left * result.matrix[r][c]
        }
    }
    return result
}
