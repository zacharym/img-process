(ns img-process.core
  (:import (java.io BufferedReader FileReader)))

(require '[clojure.data.json :as json])

(defn get-val [dirty-text] (->> dirty-text ;;scrubs file line and gives int
                                   (re-seq #"[0-9]+")
                                   (last)
                                   (Integer/parseInt)))

(defn px-val [rgb-vec] ;; takes vec of lines from file returns pixel binary
  (let [rgb-vec (map get-val rgb-vec)]
    (<
      (/ (apply + rgb-vec)
         3)
      125)))


(defn px-val1 [rgb-vec] ;; takes vect of ints, returns pixel binary
    (<
      (/ (apply + rgb-vec)
         3)
      125))

(def width1 2)
(def height1 3)

(def height ;; read height from first line of file
  (get-val (first (take 1 (line-seq (clojure.java.io/reader "resources/data2.txt"))))))

(def width ;; read width from first line of file
  (get-val (last (take 2 (line-seq (clojure.java.io/reader "resources/data2.txt"))))))

;; take all lines corresponding to pixel data from file, return lazy-seq
(def value-strings (drop 3 (line-seq (clojure.java.io/reader "resources/data2.txt"))))

;; map strings from files to ints
(def value-ints (map get-val value-strings))


(loop [px-bin [], ind 0, values value-ints, wide-px width1, height-px height1]
  (if (<=  (* 4 wide-px height-px) ind)
    px-bin
    (recur (conj px-bin (px-val1 (vec (butlast (take 4 (drop ind values))))))
           (+ 4 ind)
           values
           wide-px
           height-px)))
