(ns img-process.core
  (:import (java.io BufferedReader FileReader)))

(defn get-val [dirty-text] (->> dirty-text
                                   (re-seq #"[0-9]+")
                                   (last)
                                   (Integer/parseInt)))

(defn px-val [rgb-vec]
    (<
      (/ (apply + rgb-vec)
         3)
      125))

(def value-strings (drop 3 (line-seq (clojure.java.io/reader "resources/data2.txt"))))

(def value-ints (map get-val value-strings))

value-ints

(loop [px-bin [], ind 0, values value-ints]
  (if (<= (count values) (* 4 (count px-bin)))
    px-bin
    (recur (conj px-bin (px-val (vec (butlast (take 4 (drop (* 4 ind) values))))))
           (inc ind)
           values)))
