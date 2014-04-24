(ns img-process.image-access
  (:require [clojure.java.io :refer [file resource]]
            [img-process.protocols :as protos]
            [clojure.math.numeric-tower :as math])
  (:import  [java.awt.image BufferedImage BufferedImageOp]
            [java.awt.Rectangle]
            [awt.color]))

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
    (if (<= 40 (/ (+ (.getRed c) (.getGreen c) (.getBlue c)) 3))
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
  (let [mean (get-mean values)
        squared-deviations (map #(math/expt (- % mean) 2) values)
        variance (/ (reduce + squared-deviations)
                    (dec (count values)))]
    (math/sqrt variance)))

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
(defn mapper [values]
  (into (sorted-map)
        (map-indexed vector values)))

;; continous sub-vectors where all values are positive
(defn get-continuous-fill [values]
  (reduce (fn [res number]
            (if (pos? (val number))
              (update-in res [(dec (count res))] (fnil conj []) (key number))
              (assoc res (count res) [])))
          []
          values))

(defn get-bounds-with-padding [values]
  (map (fn [val]
         (let [padding (math/round (/ (count val) 4))]
           (assoc {} :begin (- (first val) padding)
                     :end (+ (last val) padding))))
       values))

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

(defn get-letter-bounds [row-bounds src]
    (vec (map (fn [{:keys [begin end]}]
         (-> (get-column-scores begin end src)
             mapper
             get-continuous-fill
             strip-empties
             get-bounds))
       row-bounds)))

(defn get-row-bounds [src]
    (vec (-> (get-row-scores (.getHeight src) (.getWidth src) src)
        get-std-diff-values2
        mapper
        get-continuous-fill
        strip-empties
        get-bounds-with-padding)))

(defn get-full-bounds [src]
  (let [row-bounds (get-row-bounds src)]
  {:rows row-bounds :letters (vec (get-letter-bounds row-bounds src))}))


(defn get-sub-section [bounds src]
  (let [begin-x (:x-begin bounds) end-x (:x-end bounds) begin-y (:y-begin bounds) end-y (:y-end bounds)]
    (let [rect (new java.awt.Rectangle (- end-x begin-x) (- end-y begin-y) begin-x begin-y)]
      (.getData src rect))))

(defn square-bounds [bounds]
  (let [width (- (:x-end bounds) (:x-begin bounds)) height (- (:y-end bounds) (:y-begin bounds))]
    (let [diff (- width height)]
      (if (odd? diff)
        (recur (assoc bounds :x-end (inc (:x-end bounds))))
        (if (pos? diff)
          {:x-begin (:x-begin bounds) :x-end (:x-end bounds)
           :y-begin (- (:y-begin bounds) (/ diff 2)) :y-end (+ (:y-end bounds) (/ diff 2))}
          {:x-begin (+ (:x-begin bounds) (/ diff 2)) :x-end (- (:x-end bounds) (/ diff 2))
           :y-begin (:y-begin bounds) :y-end (:y-end bounds)})))))

(defn square-bounds-1 [bounds]
  bounds)



(defn get-characters [bounds]
 (map (fn [row letters]
   (map (fn [%] {:x-begin (:begin %) :x-end (:end %) :y-begin (:begin row) :y-end (:end row)}) letters))
  (:rows bounds) (:letters bounds)))

(defn get-sub-images [img-file]
  (let [src (load-image-resource img-file)
        char-bounds (get-characters (get-full-bounds src))]
    (map (fn[bounds]
           (map #(get-sub-section % src) bounds)) char-bounds)))

()

(get-characters (get-full-bounds (load-image-resource "resources/numbers.jpg")))


;;get sub image
;;resize image into 28x28
;;take greyscale score for each
