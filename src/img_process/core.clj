(ns img-process.core
  (:require [image-access :as ims]))

(def text (ims/load-image-resource "resources/written.jpg"))

(.getHeight text)
(.getWidth text)
(def c (new java.awt.Color (.getRGB text 1 1)))

(.getRed c)


;by pixel binary
;by row binary total
;by row bin
