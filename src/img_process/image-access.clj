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
   (if (<= 120 (/ (+ (.getRed c) (.getGreen c) (.getBlue c)) 3))
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
  "returns the vector of only the positive values"
  (filter pos? values))

(defn get-mean [values]
  "returns the mean of a vector"
  (float (/ (reduce + values) (count values))))

(defn std-dev [values]
  "returns the standard deviation of a vector"
  (let [mean (get-mean values)]
  (math/sqrt (/ (reduce + (map #(math/expt (- % mean) 2) values)) (- (count values) 1)))))

(defn get-std-diff [std-dev mean value]
  "returns the distance from the mean as measured in number of standard deviations"
  (/ (- value mean) std-dev))

(defn get-std-diff-values [values]
  "maps the values of a vector to their distances from the mean of the values in terms of standard deviations,
  returns 0 for 0. Standard deviation and mean represent the vector without any zeros"
  (let [mean (get-mean (get-non-zeros values))
        std-dev (std-dev (get-non-zeros values))]
    (map #(get-std-diff std-dev mean %) values)))

(defn get-std-diff-values2 [values]
  "maps the values of a vector to their distances from the mean of the values in terms of standard deviations,
  returns 0 for 0. Standard deviation and mean represent the complete vector including zeros"
  (let [mean (get-mean values)
        std-dev (std-dev values)]
    (map #(get-std-diff std-dev mean %) values)))

;; hash map of distances
(defn mapper [values] (apply sorted-map (interleave (range (count values)) values)))

;; continous sub-vectors where all values are positive
(defn  get-continuous-fill [values] (reduce (fn [res number]
            (if (pos? (val number))
              (update-in res [(dec (count res))] (fnil conj []) (key number))
              (assoc res (count res) [])))
        []
        values))

(defn get-bounds-with-padding [values]
  (map #(let [padding (math/round (/ (count %) 4))]
         (assoc {} :begin (- (first %) padding) :end (+ (last %) padding))) values))

(defn get-bounds [values]
  (map #(assoc {} :begin (first %) :end (last %)) values))

(defn get-column-score [begin end pos-x src]
  "returns a sum of the binary pixel scores for a column of pixels for a row defined by begin and end"
  (let [inds (vec (range begin end))]
    (reduce + (map #(get-px-bin pos-x % src) inds))))

(defn get-column-scores [begin end src]
  "returns a vector of column scores for each column of text beginning and ending at the row specified
  by begin and end"
  (let [inds (vec (range (.getWidth src)))]
    (map #(get-column-score begin end % src) inds)))

(defn strip-empties [values]
  (filter #(pos? (count %)) values))

(defn get-char-map [begin end src]
  (strip-empties (get-continuous-fill (mapper (get-column-scores begin end src)))))

(defn get-letters [begin end src]
  (strip-empties (get-continuous-fill (mapper (get-column-scores begin end src)))))

;; source image
(def text (load-image-resource "resources/written.jpg"))

;; scoring rows for how many pixels are marked
(def rss (get-row-scores (.getHeight text) (.getWidth text) text))

;; distances from mean as f(std-dev)
(def std-diffs (get-std-diff-values2 rss))


(def row-map (mapper std-diffs))

;; continuously filled rows where row is greater than 3 pixel thick
(def text-rows (filter #(if (< 3 (count %))
                          true
                          false) (get-continuous-fill row-map)))


;;top and bottom of each row
(def row-bounds (get-bounds-with-padding text-rows))

;;right and left side of each letter in each row
(def letter-bounds (map #(get-bounds (strip-empties (get-continuous-fill (mapper (get-column-scores (:begin %) (:end %) text))))) row-bounds))

