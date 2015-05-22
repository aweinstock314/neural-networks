{-# LANGUAGE OverloadedStrings #-}
module VisualizeFunction where
import Control.Monad
import Data.Binary.Put
import Numeric.LinearAlgebra.HMatrix
import System.IO
import qualified Data.ByteString.Lazy as L
import qualified Data.Vector.Storable as V

-- derived from description at https://en.wikipedia.org/wiki/BMP_file_format
putBMPHeader headerLength imageLength = do
    putLazyByteString "BM" -- magic identifier
    putWord32le (headerLength + imageLength) -- total length of file (bytes)
    putLazyByteString "HELO" -- apparently a reserved field which can be filled arbitrarily
    putWord32le headerLength -- offset of pixel array in the file (bytes)

putBitmapInfoHeader (width, height) bpp = do
    putWord32le 40 -- size of BITMAPINFOHEADER
    putWord32le $ fromIntegral width
    putWord32le $ fromIntegral (-height) -- negative height -> first row is top of image
    putWord16le 1 -- # of color planes
    putWord16le bpp -- bits per pixel
    putWord32le 0 -- compression method (uncompressed RGB is 0)
    putWord32le 0 -- uncompressed size (optional if no compression is used)
    replicateM_ 2 $ putWord32le 0 -- dummy values for {x,y}-resolution
    putWord32le 0 -- use all 2^bpp colors
    putWord32le 0 -- # of "important colors", which is "generally ignored"

addBitmapHeader (w, h) image = do
    let headerLength = (2 + 4 + 4 + 4) + 40
    let imageLength = fromIntegral $ L.length image
    putBMPHeader headerLength imageLength
    putBitmapInfoHeader (w, h) 32
    putLazyByteString image

putRGB r g b = mapM_ putWord8 [b,g,r,0]

scaleFromTo (a, b) (a', b') x = ((x-a)/(b-a))*(b'-a')+a'

putPixel x y z = putRGB r g b where
    minimum = 0
    maximum = 2^8 - 1
    r = round $ if z < 0 then scaleFromTo (0, -1) (minimum, maximum) z else minimum
    g = round $ minimum
    b = round $ if z > 0 then scaleFromTo (0, 1) (minimum, maximum) z else minimum

visualizeFunction (xmin, ymin) (xmax, ymax) (w, h) (zmin, zmax) f = runPut image where
    xs = linspace w (xmin, xmax)
    ys = linspace h (ymin, ymax)
    f' = scaleFromTo (zmin, zmax) (-1, 1) . f
    image = V.forM_ ys (\y -> V.forM_ xs (\x -> putPixel x y (f' $ vector [x,y])))

visualizeFunctionBMP min max dims range = runPut . addBitmapHeader dims . visualizeFunction min max dims range

saveImage name img = withBinaryFile name WriteMode $ \h -> L.hPutStr h img

emitSample = saveImage "sample.bmp" $ visualizeFunctionBMP (-1,-1) (1,1) (2048, 2048) (-1,1) (tanh . V.sum)
