{-# LANGUAGE NoMonomorphismRestriction #-}
import NeuralNetwork
import Numeric.LinearAlgebra.HMatrix

xs = [vector [0, 0],
      vector [0, 1],
      vector [1, 0],
      vector [1, 1]]
ys = map (vector . (:[])) [0, 1, 1, 0]

xorDataSet = zip xs ys

initialNN = randomlyWeightedNetwork 0 [2, 3, 1] logisticAF

nns = iterate (stochasticGradientDescent xorDataSet 0.02) initialNN

outputs = map (\nn -> map (applyNN nn) xs) nns

errors = map (map norm_2 . zipWith (-) ys) outputs
