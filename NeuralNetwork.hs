{-# LANGUAGE FlexibleContexts, NoMonomorphismRestriction #-}
module NeuralNetwork where
import Control.Arrow hiding (app)
import Control.Monad
import Control.Monad.Trans.State.Lazy
import Data.List
import Numeric.LinearAlgebra.HMatrix
import System.Random
import qualified Data.Vector.Storable as V

-- ActivationFunction wraps a function with its derivative, along with a name (for printing)
data ActivationFunction a = AF (a -> a) (a -> a) String
instance Show (ActivationFunction a) where show (AF _ _ n) = "ActivationFunction " ++ n

tanhAF = AF tanh (\x -> 1 - (tanh x)^2) "tanh"
logisticAF = AF f (\x -> f x * (1 - f x)) "logistic" where f x = (1 / (1 + exp (-x)))

data NeuralNetwork a = NN {
    getWeightMatrices :: [Matrix a],
    getActivationFunction :: ActivationFunction a
    } deriving Show

applyNN :: Numeric a => NeuralNetwork a -> Vector a -> Vector a
applyNN (NN mats (AF theta _ _)) x = foldl' ((cmap theta .) . flip (app . tr) . V.cons 1) x mats

-- (forwardPropagate nn x) returns the values at all the intermediate layers, before and after the activation function is applied
forwardPropagate :: Numeric a => NeuralNetwork a -> Vector a -> ([Vector a], [Vector a])
forwardPropagate (NN mats (AF theta _ _)) x = first tail . unzip $ scanl aux (undefined, V.cons 1 x) mats where
    aux (_,z) w = let v = app (tr w) z in (v, V.cons 1 (cmap theta v))

-- (backPropagate nn x y) returns all the errors (in the same shape as the weights) of applying nn to x, with target value y
backPropagate :: (Num (Vector a), Numeric a) => NeuralNetwork a -> Vector a -> Vector a -> [Matrix a]
backPropagate nn@(NN mats (AF theta theta' _)) x y = gradient where
    deltaStep (ds, d) (v, w) = let d' = (cmap theta' v) * (V.tail $ app w d) in (d':ds, d')
    (v:vs, zs) = first reverse $ forwardPropagate nn x
    ws = reverse mats
    dLast = 2 * (cmap theta v - y)
    ds = fst $ foldl' deltaStep ([dLast], dLast) (zip vs ws)
    gradient = zipWith outer zs ds

batchUpdate alpha nn@(NN oldWeights _) dataset = let
    (gradient, size) = foldr (\(x, y) (gr, n) -> (zipWith (+) gr (backPropagate nn x y), n+1)) (cycle [scalar 0], 0) dataset
    step = map (*scalar (alpha / size)) gradient
    newWeights = zipWith (-) oldWeights step
    in nn {getWeightMatrices = newWeights}

stochasticGradientDescent dataset alpha nn = foldl' (\nn' pt -> batchUpdate alpha nn' [pt]) nn dataset

initializeMatrix mkEntry (m, n) = runState (fmap (matrix m) $ replicateM (m*n) (state mkEntry))

randomlyWeightedNetwork seed dims af = NN (evalState (mapM mkMat (zip dims (tail dims))) (mkStdGen seed)) af where
    mkMat (m, n) = state $ initializeMatrix (randomR (-1, 1)) (n,m+1)
