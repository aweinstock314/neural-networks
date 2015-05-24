{-# LANGUAGE NoMonomorphismRestriction, QuasiQuotes #-}
import Control.Arrow
import Control.Monad
import NeuralNetwork
import Numeric.LinearAlgebra.HMatrix
import Text.Printf.TH
import VisualizeFunction

xorDataSet = map (vector *** vector) [
    ([-1,-1], [-1]),
    ([-1, 1], [ 1]),
    ([ 1,-1], [ 1]),
    ([ 1, 1], [-1])
    ]

andDataSet = map (vector *** vector) [
    ([-1,-1], [-1]),
    ([-1, 1], [ 1]),
    ([ 1,-1], [ 1]),
    ([ 1, 1], [ 1])
    ]

nn = randomlyWeightedNetwork 0 [2, 9, 1] tanhAF
trainOn dataset = iterate (stochasticGradientDescent dataset 0.02) nn

andNNs = trainOn andDataSet
xorNNs = trainOn xorDataSet

outputs dataset nns = map (\nn -> map (applyNN nn) (map fst dataset)) nns
andOutputs = outputs andDataSet andNNs
xorOutputs = outputs xorDataSet xorNNs

image = visualizeFunctionBMP (-1,-1) (1,1) (1024,1024) (-1,1) . ((!0).) . applyNN

main = forM_ [0..10] $ \i -> do
    saveImage ([s|neuralnetwork_and_%04d.bmp|] (2^i)) (image (andNNs !! (2^i)))
    saveImage ([s|neuralnetwork_xor_%04d.bmp|] (2^i)) (image (xorNNs !! (2^i)))
