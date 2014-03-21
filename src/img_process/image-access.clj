(ns image-access
  (:require [clojure.java.io :refer [file resource]])
  (:require [protocols :as protos])
  (:require [clojure.math.numeric-tower :as math])
  (:import  [java.awt.image BufferedImage BufferedImageOp])
  (:import  [java.awt.color]))


(defn new-buffered
  "Creates java buffered image"
  (^BufferedImage [width height]
    (BufferedImage. (int width) (int height) BufferedImage/TYPE_INT_RGB)))

(defn load-image
  "Loads a BufferedImage from a string, file or a URL representing a resource
  on the classpath.
  Usage:
    (load-image \"/some/path/to/image.png\")
    ;; (require [clojure.java.io :refer [resource]])
    (load-image (resource \"some/path/to/image.png\"))"
  (^BufferedImage [resource] (protos/as-image resource)))

(defn load-image-resource [res-path]
  "Loads an image from a named resource on the classpath.
   Equivalent to (load-image (clojure.java.io/resource res-path))"
  (load-image res-path))

(defn get-px-bin [pos-x pos-y src]
  "retuns a binary score for whether or not a mark is present at a particular
  pixel location"
  (let [c (new java.awt.Color (.getRGB src pos-x pos-y))]
   (if (<= 70 (/ (+ (.getRed c) (.getGreen c) (.getBlue c)) 3))
     0
     1)))

(defn get-row-score [width pos-y src]
  "returns a sum of the binary pixel score for the row of pixels at pos-y"
  (let [inds (vec (range width))]
    (reduce + (map #(get-px-bin % pos-y src) inds))))

(defn get-row-scores [height width src]
  "returns a vector of row scores (sums of binary px scores) for each row of image"
  (let [inds (vec (range height))]
    (map #(get-row-score width % src) inds)))

(defn get-non-zeros [values]
  (filter pos? values))

(defn get-mean [values] (float (/ (reduce + values) (count values))))

(defn std-dev [values] (let [mean (get-mean values)]
  (math/sqrt (/ (reduce + (map #(math/expt (- % mean) 2) values)) (- (count values) 1)))))

(def text (load-image-resource "resources/written.jpg"))

(def rss (get-row-scores 288 600 text))


