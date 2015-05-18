{-# LANGUAGE NoMonomorphismRestriction #-}
module NeuralNetwork where
import Control.Arrow (first)
import Control.Monad
import Control.Monad.Trans.State.Lazy
import Numeric.LinearAlgebra.HMatrix
import System.Random
import qualified Data.Vector.Storable as V

-- ActivationFunction wraps a function with its derivative
data ActivationFunction a = ActivationFunction (a -> a) (a -> a)

tanhActivation = ActivationFunction tanh ((**2) . sech) where sech = (1/) . cosh

data NeuralNetwork a = NeuralNetwork {
    getWeightMatrices :: [Matrix a],
    getActivationFunction :: ActivationFunction a
    }

instance (Element a, Show a) => Show (NeuralNetwork a) where
    show (NeuralNetwork {getWeightMatrices = mats}) = "NeuralNetwork " ++ show mats

-- (forwardPropagate nn x) returns the values at all the intermediate layers
forwardPropagate :: Numeric a => NeuralNetwork a -> Vector a -> [Vector a]
forwardPropagate (NeuralNetwork mats (ActivationFunction theta _)) x = scanl ((V.map theta .) . flip app) x mats

makeRandomMatrix gen (m, n) range = runState (replicateM (m*n) (state (randomR range)) >>= return . matrix m) gen

randomlyWeightedNetwork seed dims af = NeuralNetwork (fst . foldr step ([], mkStdGen seed) $ zip dims (tail dims)) af where
    step (m, n) (mats, gen) = first (:mats) (makeRandomMatrix gen (m, n) (0, 1))
