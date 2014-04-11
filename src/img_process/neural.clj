(ns img-process.neural
  (:require [clojure.java.io :refer [file resource]]
            [img-process.protocols :as protos]
            [clojure.math.numeric-tower :as math]
            [clojure.data.json :as json]))

(defn sigmoid
  "for a given input returns a value in range (0,1) determined by the sigmoid
  neuron function, which is a essentially a smoothed out step function with
  a point of inflection at x=0"
  [z]
  (/ 1 (+ 1 (math/expt 2.718281828 (* -1 z)))))

(defn biases-init [n]
  "takes the number of neurons in a layer and returns a vector of randoms in range
  [0,1) representing initial biases for that layer"
  (vec (take n (repeatedly rand))))

(defn weights-init [n-in n-hd]
  "takes the number of inputs and number of hidden neurons those inputs will go to, and
  initializes a weight of a random value in range [0,1) for each connection. Returns a
  2-D vector of weights"
  (vec (map #(biases-init %) (vec (repeat n-hd n-in)))))  ;;;why is the outer vec necessary here?

(defn dot [vec0 vec1]
  "returns the dot product of two vectors"
  (reduce +
          (for [i (range (count vec0))]
            (* (vec0 i) (vec1 i)))))

(defn sig-out [in-vec weight-vec bias-vec]
  "takes a vector of inputs, a vector of neurons defined by the bias-vector, and corresponding 2-D
  vector of weights for the connections between those two. Returns the output of a layer of sigmoids
  defined by the bias vector"
    (vec (map #(sigmoid (- (dot (weight-vec %) in-vec) (bias-vec %))) (range (count bias-vec)))))

(defn vec-length [vec1]
  "returns the length of a vector in multiple dimensions (each element represents a scalar in a different
  dimension. Interestingly the pythagorean theorem is the 2 dimensional case of this."
  (->> vec1
   (map #(math/expt % 2))
   (reduce +)
   math/sqrt))

(defn get-cost [dsrd actl]
  "returns the cost of the network for a given input defined by the network's actual result for that
  input and the correct result for that input"
  (* 0.5
     (vec-length
      (for [i (range (count dsrd))]
        (- (dsrd i) (actl i))))))

(defn evaluate [inputs weights-a biases-a weights-b biases-b]
  "takes a vector of inputs, biases for the hidden layer (a), biases for the output layer (b),
  weights for the input->a connections, weights for the input->b connections, and produces the
  output vector"
  (sig-out (sig-out inputs weights-a biases-a) weights-b biases-b))

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
  ;;;;;;; THIS WILL BE UPDATED DEPENDING ON WHICH FILE WE ARE READING DATA FROM    ;;;;;;;;;;;;;
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

(defn set-cost [train-set weights-a biases-a weights-b biases-b]
  "takes a training set of key value pairs of inputs and results and returns the cost representing how
  wrong the weights and biases are for that random training set"
  (reduce + (map #(get-cost (:result %) (evaluate (:input %) weights-a biases-a weights-b biases-b)) train-set)))

(defn up-ten-percent [value]
  "accepts a value and returns that value increased by 10% unless an increase of 10% makes that value greater than 1
  in which case returns 0.99"
  (if (< (* 1.1 value) 1)
    (* 1.1 value)
    0.99))

(defn down-ten-percent [value]
  "accepts a value and returns that value decreased by 10% with a minimum value of 0.01"
  (if (< (* 0.9 value) 0.01)
    0.01
    (* 0.9 value)))


(defn train-network [training-data input-n layer-a-n layer-b-n sample-n]
  "takes an array of correct input/output pairs of training data, the number of inputs to the network, the
  number of neurons in the first and second layers, and the sample size for each training batch. Returns
  a key value mapping of `trained` weights and biases for the network. Works by randomly initializing weights
  and biases and then following a gradient decent algorithm to minimize the cost of the network over each
  generation w.r.t. each weight and bias."
  (let [settings {:weights-a (weights-init input-n layer-a-n) :weights-b (weights-init layer-a-n layer-b-n)
                  :biases-a (biases-init layer-a-n) :biases-b (biases-init layer-b-n)}]
    (reduce
     (fn [accum cur-set]
       (compute-changes cur-set (:weights-a accum) (:biases-a accum) (:weights-b accum) (:biases-b accum)))
     settings
     training-data)))

(defn train-initialized-network [training-data input-n layer-a-n layer-b-n sample-n init-w-a init-b-a init-w-b init-b-b]
  "takes an array of correct input/output pairs of training data, the number of inputs to the network, the
  number of neurons in the first and second layers, the sample size for each training batch, and previously computed
  network settings to take as initial settings. Returns a key value mapping of `trained` weights and biases
  for the network. Works by randomly initializing weights and biases and then following a gradient decent algorithm
  to minimize the cost of the network over each generation w.r.t. each weight and bias."
  (let [settings {:weights-a init-w-a :weights-b init-w-b
                  :biases-a init-b-a :biases-b init-b-b}]
    (reduce
     (fn [accum cur-set]
       (compute-changes cur-set (:weights-a accum) (:biases-a accum) (:weights-b accum) (:biases-b accum)))
     settings
     training-data)))

(defn compute-changes [train-set weights-a biases-a weights-b biases-b]
  "accepts a training batch, weights and biases for the network, returns a settings collection
  representing the new state of the network after following the gradient decent method to reduce error"
  {:weights-a (weights-a-change train-set weights-a biases-a weights-b biases-b)
   :biases-a (biases-a-change train-set weights-a biases-a weights-b biases-b)
   :weights-b (weights-b-change train-set weights-a biases-a weights-b biases-b)
   :biases-b (biases-b-change train-set weights-a biases-a weights-b biases-b)})


(defn biases-a-change [train-set weights-a biases weights-b biases-b]
  "takes a training batch, a vector of biases for the first layer of connections, the rest of the network
  settings and returns an identity matrix where true means that the partial derivative of the total cost
  with respect to that neuron's bias is negative [for a step size of 10% of the current bias
  value] (increase that value to decrease the total cost)"
  (loop [res [] ct 0]
    (if (>= ct (count biases))
      res
     (recur
      (let [mutant (assoc biases ct (up-ten-percent (biases ct)))]
        (if (>=
             (set-cost train-set weights-a mutant weights-b biases-b)
             (set-cost train-set weights-a biases weights-b biases-b))
           (conj res (down-ten-percent (biases ct)))
          (conj res (up-ten-percent (biases ct)))))
      (inc ct)))))

(defn biases-b-change [train-set weights-a biases-a weights-b biases]
  "takes a training batch, a vector of biases for the second layer of connections, the rest of the network
  settings and returns an identity matrix where true means that the partial derivative of the total cost
  with respect to that neuron's bias is negative [for a step size of 10% of the current bias
  value] (increase that value to decrease the total cost)"
  (loop [res [] ct 0]
    (if (>= ct (count biases))
      res
     (recur
      (let [mutant (assoc biases ct (up-ten-percent (biases ct)))]
        (if (>=
             (set-cost train-set weights-a biases-a weights-b mutant)
             (set-cost train-set weights-a biases-a weights-b biases))
           (conj res (down-ten-percent (biases ct)))
          (conj res (up-ten-percent (biases ct)))))
      (inc ct)))))

(defn weights-a-change [train-set weights biases-a weights-b biases-b]
  "takes a training batch, a vector of weights for the first layer of connections, the rest of the network's
  settings and returns an identiy matrix where true means that the partial derivative of the total cost
  with respect to that connection's weight is negative [for a step size of 10% of the weight's current
  value] (increase that value to decrease the total cost)"
  (loop [res [] outer-ct 0]
    (if (>= outer-ct (count weights))
      res
     (recur
      (conj res
       (loop [res-in [] inner-ct 0]
         (if (>= inner-ct (count (weights outer-ct)))
           res-in
          (let [mutant (assoc weights outer-ct (assoc (weights outer-ct) inner-ct (up-ten-percent ((weights outer-ct) inner-ct))))]
            (if (>=
                  (set-cost train-set mutant biases-a weights-b biases-b)
                  (set-cost train-set weights biases-a weights-b biases-b)) ;;partial derivative is positive
                (recur
                 (conj res-in (down-ten-percent ((weights outer-ct) inner-ct)))
                 (inc inner-ct))
               (recur
                (conj res-in (up-ten-percent ((weights outer-ct) inner-ct)))
                (inc inner-ct)))))))
      (inc outer-ct)))))

(defn weights-b-change [train-set weights-a biases-a weights biases-b]
  "takes a training batch, a vector of weights for the second layer of connections, the rest of the networks
  settings and returns an identiy matrix where true means that the partial derivative of the total cost
  with respect to that connection's weight is negative [for a step size of 10% of the weight's current
  value] (increase that value to decrease the total cost)"
    (loop [res [] outer-ct 0]
    (if (>= outer-ct (count weights))
      res
     (recur
      (conj res
       (loop [res-in [] inner-ct 0]
         (if (>= inner-ct (count (weights outer-ct)))
           res-in
          (let [mutant (assoc weights outer-ct (assoc (weights outer-ct) inner-ct (up-ten-percent ((weights outer-ct) inner-ct))))]
            (if (>=
                  (set-cost train-set weights-a biases-a mutant biases-b)
                  (set-cost train-set weights-a biases-a weights biases-b)) ;;partial derivative is positive
                (recur
                 (conj res-in (down-ten-percent ((weights outer-ct) inner-ct)))
                 (inc inner-ct))
               (recur
                (conj res-in (up-ten-percent ((weights outer-ct) inner-ct)))
                (inc inner-ct)))))))
      (inc outer-ct)))))


(def a (reshape (shuffle (vec (get-training-data 10 30))) 10))  ;; 2 generations, 3 input/output pairs per batch
(def settings (train-network a 784 30 10 10))
settings
(time (train-network a 784 15 10 2))  ;;~224 seconds

(:weights-a settings)


(def test-datum (json-2-training-pair (json/read-str "[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15625, 0.4375, 0.625, 0.72265625, 0.6640625, 0.4765625, 0.0859375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09375, 0.7734375, 0.9453125, 0.9921875, 0.9921875, 0.9921875, 0.9921875, 0.9921875, 0.80859375, 0.0390625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2265625, 0.80078125, 0.9921875, 0.9921875, 0.98828125, 0.953125, 0.8203125, 0.953125, 0.98046875, 0.9921875, 0.359375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08984375, 0.8203125, 0.9921875, 0.9921875, 0.81640625, 0.34765625, 0.0, 0.0, 0.0, 0.2734375, 0.89453125, 0.73828125, 0.0546875, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.9921875, 0.9921875, 0.73828125, 0.04296875, 0.0, 0.0, 0.0, 0.0, 0.0078125, 0.72265625, 0.9921875, 0.18359375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1953125, 0.9296875, 0.9921875, 0.95703125, 0.1640625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.203125, 0.9921875, 0.9921875, 0.484375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.40625, 0.9921875, 0.9921875, 0.91796875, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.52734375, 0.9921875, 0.9921875, 0.546875, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04296875, 0.91796875, 0.9921875, 0.93359375, 0.22265625, 0.0078125, 0.0, 0.0, 0.140625, 0.7421875, 0.984375, 0.9921875, 0.9921875, 0.62890625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.66015625, 0.9921875, 0.9921875, 0.9921875, 0.8515625, 0.84765625, 0.84765625, 0.94140625, 0.9921875, 0.9921875, 0.9921875, 0.9921875, 0.875, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.390625, 0.9921875, 0.9921875, 0.9921875, 0.9921875, 0.9921875, 0.9921875, 0.9921875, 0.9921875, 0.9921875, 0.9921875, 0.9921875, 0.68359375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.11328125, 0.57421875, 0.8515625, 0.9921875, 0.9921875, 0.9921875, 0.99609375, 0.8359375, 0.54296875, 0.625, 0.9921875, 0.9921875, 0.921875, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.078125, 0.21875, 0.109375, 0.109375, 0.109375, 0.04296875, 0.0, 0.078125, 0.9921875, 0.9921875, 0.78125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.328125, 0.9921875, 0.9921875, 0.62890625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.40625, 0.9921875, 0.9921875, 0.546875, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.40625, 0.9921875, 0.9921875, 0.546875, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.40625, 0.9921875, 0.9921875, 0.546875, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.40625, 0.9921875, 0.9921875, 0.6953125, 0.22265625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.40625, 0.9921875, 0.9921875, 0.9921875, 0.90625, 0.02734375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.23046875, 0.9921875, 0.9921875, 0.98046875, 0.453125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01171875, 0.5, 0.7578125, 0.57421875, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]")))


;; seeing how wrong we are with those settings
(get-cost (:result test-datum) (evaluate (:input test-datum) (:weights-a settings) (:biases-a settings) (:weights-b settings) (:biases-b settings)))


;; seeing how wrong we are with random settings
(get-cost (:result test-datum) (evaluate (:input test-datum) (weights-init 784 30) (biases-init 30) (weights-init 30 10) (biases-init 10)))


;;; getting new random training data
(def b (reshape (shuffle (vec (get-training-data 10 30))) 10))

;;; restarting the training with the previous settings
(def settings-2 (train-initialized-network b 784 30 10 10 (:weights-a settings) (:biases-a settings) (:weights-b settings) (:biases-b settings)))


settings-2

