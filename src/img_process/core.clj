(ns img-process.core
  (:import (java.io BufferedReader FileReader)))

(defn process-file [file-name line-func line-acc]
  (with-open [rdr (BufferedReader. (FileReader. file-name))]
    (reduce line-func line-acc (line-seq rdr))))

(defn process-line [acc line]
  (+ acc 1))

(prn (process-file "resources/img-data.txt" process-line 1))

(with-open [rdr (clojure.java.io/reader "resources/data2.txt")]
  (println (take 3 (line-seq rdr)))
  (println (take 2 (line-seq rdr))))

(def sample "'4': 251,
     '5': 251,
     '6': 251,
     '7': 255,")

(defn clean-text [dirty-text] (vec (-> dirty-text
                                   (clojure.string/replace #"['\s]" "")
                                   (clojure.string/replace #"[,]" "\n")
                                   (clojure.string/replace #"[:]" " ")
                                   (clojure.string/split #"\n"))))

(def sample2 (clean-text sample))

(defn get-val [string-pair] (-> string-pair
                                (clojure.string/split #"\s")
                                (second)
                                (Integer/parseInt)))

(defn get-px-data [px-text]
  (apply hash-map (interleave
                   [:r :g :b :a]
                   (mapv get-val (clean-text px-text)))))


