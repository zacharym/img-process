(ns img-process.core
  (:require
            [clojure.math.numeric-tower :as math]))
;; [clojure.math.numeric-tower :as math]
(def text (ims/load-image-resource "resources/written.jpg"))

(.getHeight text)
(.getWidth text)
(def c (new java.awt.Color (.getRGB text 1 1)))

(.getRed c)


;by pixel binary
;by row binary total
;by row bin


(apply * [9 9 9])

(math/sqrt 49)
