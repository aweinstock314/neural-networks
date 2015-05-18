{-# LANGUAGE NoMonomorphismRestriction #-}
module NeuralNetwork where
import Control.Arrow (first)
import Control.Monad
import Control.Monad.Trans.State.Lazy
import Numeric.LinearAlgebra.HMatrix
import System.Random

data NeuralNetwork a = NeuralNetwork { getWeightMatrices :: [Matrix a] } deriving Show

-- (forwardPropagate nn x) returns the values at all the intermediate layers
forwardPropagate :: Numeric a => NeuralNetwork a -> Vector a -> [Vector a]
forwardPropagate (NeuralNetwork mats) x = scanl (flip app) x mats

makeRandomMatrix gen (m, n) range = runState (replicateM (m*n) (state (randomR range)) >>= return . matrix m) gen

randomlyWeightedNetwork seed dims = NeuralNetwork . fst . foldr step ([], mkStdGen seed) $ zip dims (tail dims) where
    step (m, n) (mats, gen) = first (:mats) (makeRandomMatrix gen (m, n) (0, 1))
