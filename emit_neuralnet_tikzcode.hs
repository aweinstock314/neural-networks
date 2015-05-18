{-# LANGUAGE QuasiQuotes #-}
module Main where
import Control.Monad
import Text.Printf.TH

nn_xs 0 = [0..4]
nn_xs 1 = [0..5]
nn_xs 2 = [0..3]
nn_xs _ = error "Sample NN only has 3 layers"

nn_ys = [0..2]
node_size = 0.5

adjacents xs = zip xs (tail xs)

emit_bias_node y = [sP|\\draw (-2, %d) circle (%f) node {$1$};|] (2*y) node_size

emit_nn_layer y = do
    forM_ (nn_xs y) $ \x -> [sP|\\draw (%d, %d) circle (%f) node{$x_{%d}^{(\\ell_{%d})}$};|] (2*x) (2*y) node_size (x+1) y
    emit_bias_node y

emit_connections (y1, y2) = forM_ ((-1):(nn_xs y1)) (emit_connections_from (y1, y2))

emit_connections_from (y1, y2) x1 = forM_ (nn_xs y2) $ \x2 -> [sP|\\draw [->] (%d, %d+%f) -- (%d, %d-%f);|] (2*x1) (2*y1) node_size (2*x2) (2*y2) node_size
    
main = do
    forM_ nn_ys $ \y -> emit_nn_layer y
    mapM_ emit_connections $ adjacents nn_ys
    forM_ ((-1):(nn_xs 2)) $ \x -> [sP|\\draw [->] (%d, 4+%f) -- (4, 6-%f);|] (2*x) node_size node_size
    [sP|\\draw (4, 6) circle (%f) node{$x_0^{(\\ell_3)}$};|] node_size
    forM_ (3:nn_ys) $ \y -> [sP|\\draw (12, %d) node {$\\ell_{%d}$};|] (2*y) y
