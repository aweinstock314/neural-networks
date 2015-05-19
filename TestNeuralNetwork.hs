{-# LANGUAGE NoMonomorphismRestriction #-}
import NeuralNetwork
import Numeric.LinearAlgebra.HMatrix

xorDataSet = zip xs ys where
    xs = [vector [0, 0],
          vector [0, 1],
          vector [1, 0],
          vector [1, 1]]
    ys = map (vector . (:[])) [0, 1, 1, 0]

initialNN = randomlyWeightedNetwork 0 [2, 2, 1] tanhActivation

nns = iterate (stochasticGradientDescent xorDataSet 0.2) initialNN

outputs x = map (last . flip forwardPropagate (vector x)) nns
