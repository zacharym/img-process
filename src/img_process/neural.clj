(ns img-process.neural
  (:require [clojure.java.io :refer [file resource]]
            [img-process.protocols :as protos]
            [img-process.initial :as initial]
            [clojure.data.json :as json]
            [clojure.math.numeric-tower :as math]))

(defn sigmoid
  "for a given input returns a value in range (0,1) determined by the sigmoid
  neuron function, which is a essentially a smoothed out step function with
  a point of inflection at x=0"
  [z]
  (/ 1 (+ 1 (math/expt 2.71828 (* -1 z)))))


(defn dot [vec0 vec1]
  "returns the dot product of two vectors"
  (reduce +
          (for [i (range (count vec0))]
            (* (vec0 i) (vec1 i)))))

(defn sig-out
  "takes an input vector, a 2d vector representing weights of connections between the input and
  next layer and the biases of the next layer. Returns the out of the layer of the second layer
  of sigmoids. Settings vectors are one wider than inputs to account for biases thus we must conj
  a 1 to the end of the input."
  [in settings]
  (let [inputs (conj in 1)]
    (vec (map (fn[%] (sigmoid (dot inputs %))) settings))))

(defn vec-length [vec1]
  "returns the length of a vector in multiple dimensions (each element represents a scalar in a different
  dimension). Interestingly the pythagorean theorem is the 2 dimensional case of this."
  (->> vec1
   (map #(math/expt % 2))
   (reduce +)
   math/sqrt))

(defn get-cost [dsrd actl]
  "returns the cost of the network for a given input defined by the network's actual result for that
  input and the correct result for that input"
  (* 0.5
     (reduce +
             (map (fn [%] (math/expt % 2))
                  (vec (for [i (range (count dsrd))]
                    (- ((vec dsrd) i) ((vec actl) i))))))))

(defn round-3
  "returns a float of the input rounded to 3 decimal places"
  [in-val]
  (double (/ (math/round (* in-val 1000)) 1000)))

(defn evaluate
  "takes a vector of inputs, settings (weights and biases 2d vector as wide as the next layer +1 for the biases),
  weights for the hidden and output layer, returns the resulting vector of the inputs propagated through the network."
  [inputs settings-a settings-b]
  (sig-out (sig-out inputs settings-a) settings-b))

(defn json-2-training-pair [json-vec]
  "takes vector created by json/read-str and converts in to a key value mapping of the input
  data and correct result"
  {:input (json-vec 0) :result (json-vec 1)})

(defn get-random-set [samp-n pop-n]
  "returns a samp-n long sequence of random integers in the range [0,pop-n).
  No repeat values in return sequence"
  (loop [res (list)]
    (if (>= (count res) samp-n)
      res
      (let [temp (rand-int pop-n)]
        (if (some #{temp} res)
          (recur res)
          (recur (conj res temp)))))))

(defn get-training-data [batch-size batch-num]
  ;;;;;;;      THIS WILL BE UPDATED DEPENDING ON WHICH FILE WE ARE READING DATA FROM       ;;;;;;;;;;;;;
  "takes the size of each training batch and the number of batches, returns a 2D vector
  of randomly selected input and output data"
  (let [line-numbers (get-random-set (* batch-size batch-num) 49990)]
    (with-open [rdr (clojure.java.io/reader "resources/training.txt")]
      (doall (let [lines (line-seq rdr)
            indexed-lines (map-indexed (fn [idx itm] [idx itm]) lines)]
            (map (fn [idx-item] (json-2-training-pair (json/read-str (idx-item 1))))
             (filter #(some #{(first %)} line-numbers) indexed-lines)))))))

(defn reshape [vect width]
  "takes a 1D vector and reshapes to as a 2d vector of the specified width"
  (loop [i 0 f (/ (count vect) width) res []]
    (if (>= i f)
      res
      (recur
       (inc i)
       f
       (conj res (subvec vect (* i width) (+ width (* i width))))))))

(defn set-cost [training-set network]
  "takes a training set of key value pairs of inputs and results and returns the cost representing how
  wrong the weights and biases are for that random training set"
   (reduce + (map #(get-cost (:result %) (evaluate (:input %) (:settings-a network) (:settings-b network))) training-set)))

(defn settings-change
  "accepts a training set the network as is and a step size, returns the as it should look to follow gradient descent
  settings-a updated to follow gradient descent to minimize wrongness."
  [training-set network step-size settings-key]
  (let [settings (settings-key network)]
    (loop [out-ct 0 res-out []]
      (if (>= out-ct (count settings))
        res-out
        (recur
         (inc out-ct)
         (conj res-out
               (loop [in-ct 0 res-in []]
                   (if (>= in-ct (count (settings out-ct)))
                     res-in
                    (let [mutant (assoc (settings out-ct) in-ct (+ ((settings out-ct) in-ct) step-size))]
                       (let [mutant-settings (assoc settings out-ct mutant)]
                         (let [mutant-network (assoc network settings-key mutant-settings)]
                           (if (> (set-cost training-set network)
                                  (set-cost training-set mutant-network))
                             (recur
                              (inc in-ct)
                              (conj res-in (+ ((settings out-ct) in-ct) step-size)))
                             (recur
                              (inc in-ct)
                              (conj res-in (- ((settings out-ct) in-ct) step-size)))))))))))))))

(defn settings-update
  "accepts a training set of data (a vector of :input :result maps) the current settings of the network and reutrns the
  network such that the settings have been changed to reduce the wrongness of the network for that training set."
  [training-set network step-size]
  (let [alpha (settings-change training-set network step-size :settings-a)
        beta  (settings-change training-set network step-size :settings-b)]
      {:settings-a alpha
       :settings-b beta}))


(defn train-network [network-state jump batch-size num-gens]
  "takes the current network, the initial jump size for changing settings, the number of correct input/output pairs in
  each training batch, and the number of generations that
  you intend to train for. Proceeds to read the proper number of training samples and organize them into the proper
  shape given the network, and proceeds to train the network over the specified number of generations while logging
  results as they are produced."
  (let [training-data (reshape (shuffle (vec (get-training-data batch-size num-gens))) batch-size)]
    (reduce
     (fn [accum cur-set]
       (with-open [wrtr (clojure.java.io/writer "resources/training-log/network4.txt" :append true)]
         (.write wrtr (json/write-str accum)))
       {:network (settings-update cur-set (:network accum) jump) :gen (inc (:gen accum))})
     network-state
     training-data)))

(defn go [jump batch-size num-gens dst]
    (let [res (train-network initial/initial-values jump batch-size num-gens)]
      (spit dst res :append true)))
(defn -main[]
  (go 0.05 1 1 "resources/training-log/results1.txt"))
