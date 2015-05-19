{-# LANGUAGE FlexibleContexts, NoMonomorphismRestriction #-}
module NeuralNetwork where
import Control.Monad
import Control.Monad.Trans.State.Lazy
import Numeric.LinearAlgebra.HMatrix
import System.Random
import qualified Data.Vector.Storable as V

(<#>) = flip fmap


-- ActivationFunction wraps a function with its derivative, along with a name (for printing)
data ActivationFunction a = AF (a -> a) (a -> a) String
instance Show (ActivationFunction a) where show (AF _ _ n) = "ActivationFunction " ++ n

tanhActivation = AF tanh ((**2) . sech) "tanh/sech^2" where sech = (1/) . cosh

data NeuralNetwork a = NN {
    getWeightMatrices :: [Matrix a],
    getActivationFunction :: ActivationFunction a
    } deriving Show

-- (forwardPropagate nn x) returns the values at all the intermediate layers
forwardPropagate :: Numeric a => NeuralNetwork a -> Vector a -> [Vector a]
forwardPropagate (NN mats (AF theta _ _)) x = scanl ((cmap theta .) . flip app . V.cons 1) x mats

-- (backPropagate nn x y) returns all the errors (in the same shape as the weights) of applying nn to x, with target value y
backPropagate :: (Num (Vector a), Numeric a) => NeuralNetwork a -> Vector a -> Vector a -> [Matrix a]
backPropagate nn@(NN mats (AF _ theta' _)) x y = errors where
    deltaStep (acc, prev) (v, w) = let d = V.tail $ (cmap theta' $ V.cons 1 v) * (app (tr w) prev) in (d:acc, d)
    v : vs = reverse $ forwardPropagate nn x
    ws = reverse mats
    lastDelta = (cmap theta' v) * (y - v)
    deltas = fst $ foldl deltaStep ([], lastDelta) (zip vs ws)
    errors = reverse $ zipWith outer (v:vs) deltas

initializeMatrix mkEntry (m, n) = runState (replicateM (m*n) (state mkEntry) <#> matrix m)

randomlyWeightedNetwork seed dims af = NN (evalState (mapM mkMat (zip dims (tail dims))) (mkStdGen seed)) af where
    mkMat (m, n) = state $ initializeMatrix (randomR (0, 1)) (m+1,n)
