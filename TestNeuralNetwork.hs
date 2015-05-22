{-# LANGUAGE NoMonomorphismRestriction #-}
import NeuralNetwork
import Numeric.LinearAlgebra.HMatrix
import VisualizeFunction

xs = [vector [0, 0],
      vector [0, 1],
      vector [1, 0],
      vector [1, 1]]
ys = map (vector . (:[])) [0, 1, 1, 0]

xorDataSet = zip xs ys

nn = randomlyWeightedNetwork 0 [2, 3, 1] logisticAF

nns = iterate (stochasticGradientDescent xorDataSet 0.02) nn

outputs = map (\nn -> map (applyNN nn) xs) nns

errors = map (map norm_2 . zipWith (-) ys) outputs

images = map (visualizeFunctionBMP (-1,-1) (1,1) (1024,1024) (-1,1) . ((!0).) . applyNN) nns

main = flip mapM_ [0,1,2,5,10,100] $ \i -> saveImage (concat ["xornet", show i, ".bmp"]) (images !! i)
