(ns img-process.core
  (:import (java.io BufferedReader FileReader)))

(defn get-val [dirty-text] (->> dirty-text
                                   (re-seq #"[0-9]+")
                                   (last)
                                   (Integer/parseInt)))

(defn px-val [rgb-vec]
  (let [rgb-vec (map get-val rgb-vec)]
    (<
      (/ (apply + rgb-vec)
         3)
      125)))

(with-open [rdr (clojure.java.io/reader "resources/data2.txt")]
  (line-seq rdr)
  (line-seq rdr)
  (line-seq rdr)
  (line-seq rdr)
  (doseq [l (partition 4 (line-seq rdr))]
      (println (px-val (take 3 l)))))
