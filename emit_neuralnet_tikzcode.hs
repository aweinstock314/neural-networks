{-# LANGUAGE QuasiQuotes #-}
module Main where
import Control.Monad
import Text.Printf.TH

nn_xs = [0..4]

emit_nn_layer y = forM_ nn_xs $ \x -> [sP|\\draw (%d, %d) circle (0.25cm);|] (2*x) y

emit_connections (y1, y2) = forM_ nn_xs $ \x1 ->
    forM_ nn_xs $ \x2 -> [sP|\\draw [->] (%d, %d+0.25) -- (%d, %d-0.25);|] (2*x1) (2*y1) (2*x2) (2*y2)
    
main = do
    forM_ [0..2] $ \y -> emit_nn_layer (2*y)
    mapM_ emit_connections [(0,1), (1,2)]
    forM_ nn_xs $ \x -> [sP|\\draw [->] (%d, 4+0.25) -- (4, 6-0.25);|] (2*x)
    [sP|\\draw (4, 6) circle (0.25cm);|]
