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

(defn weights-init
  "takes the number of inputs and number of hidden neurons those inputs will go to, and
  initializes a weight of a random value in range [0,1) for each connection. Returns a
  2-D vector of weights"
  [n-in n-hd]
  (vec (map (fn [_] (vec (map (fn [_] (rand)) (vec (range n-in))))) (vec (range n-hd)))))

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
  "returns the length of a vector in multiple dimensions (each element represents a length in a different
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
  (let [line-numbers (get-random-set (* batch-size batch-num) 99)]
    (with-open [rdr (clojure.java.io/reader "resources/train_hundo.txt")]
      (let [lines (line-seq rdr)]
        (loop [i 0 f (apply max line-numbers) res [] lines lines]
          (if (> i f)
            res
            (if (some #{i} line-numbers)
              (recur
               (inc i)
               f
               (conj res (json-2-training-pair (json/read-str (first lines))))
               (drop 1 lines))
              (recur
               (inc i)
               f
               res
               (drop 1 lines)))))))))

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

(defn train-network [train-data input-n layer-a-n layer-b-n]
  "takes a 2D array of correct input/output pairs of training data - a single row to be used for each
  generation of training, the number of inputs to the network, the number of neurons in the first and
  second layers. Returns a key value mapping of `trained` weights and biases for the network. W
  orks by randomly initializing weights and biases and then following a gradient decent algorithm to
  minimize the cost of the network over each generation w.r.t. each weight and bias."
  )



(def a (get-training-data 10 5))



((init-weights-a 1) 1)
(count (init-weights-a 0))


(time (def sample-result (weights-a-change test-value-inputs init-weights-a init-biases-a init-weights-b init-biases-b)))
(def sample-result2 (biases-a-change test-value-inputs init-weights-a init-biases-a init-weights-b init-biases-b))
sample-result2
(def sample-result)


init-biases-a
(def init-biases-aa [0.56 0.4321580631740959 0.57710040224414 0.6185965659086016 0.18969736398984938 0.8749276162565557 0.7980151147402715 0.664393999821422 0.9569106331833737 0.9815324868138454 0.3432571362384952 0.6501960631145083 0.3426291107100178 0.36863149128690365 0.7569718521150473])


(map #(:result %) test-value-inputs)
(count test-values)

(set-cost test-value-inputs init-weights-a init-biases-a init-weights-b init-biases-b)
(set-cost1 test-value-inputs init-weights-a in1it-biases-a init-weights-b init-biases-b)
(set-cost2 test-value-inputs init-weights-a init-biases-a init-weights-b init-biases-b)
test-value-inputs


(evaluate (:input (test-value-inputs 1)) init-weights-a init-biases-a init-weights-b init-biases-b)




(def init-weights-a (weights-init 784 15))
(def init-weights-b (weights-init 15 10))

(def init-biases-a (vec (map (fn [_] (rand)) (range 15))))
(def init-biases-b (vec (map (fn [_] (rand)) (range 10))))

(def test-input [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.01171875 0.0703125 0.0703125 0.0703125 0.4921875 0.53125 0.68359375 0.1015625 0.6484375 0.99609375 0.96484375 0.49609375 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.1171875 0.140625 0.3671875 0.6015625 0.6640625 0.98828125 0.98828125 0.98828125 0.98828125 0.98828125 0.87890625 0.671875 0.98828125 0.9453125 0.76171875 0.25 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.19140625 0.9296875 0.98828125 0.98828125 0.98828125 0.98828125 0.98828125 0.98828125 0.98828125 0.98828125 0.98046875 0.36328125 0.3203125 0.3203125 0.21875 0.15234375 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0703125 0.85546875 0.98828125 0.98828125 0.98828125 0.98828125 0.98828125 0.7734375 0.7109375 0.96484375 0.94140625 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.3125 0.609375 0.41796875 0.98828125 0.98828125 0.80078125 0.04296875 0.0 0.16796875 0.6015625 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0546875 0.00390625 0.6015625 0.98828125 0.3515625 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.54296875 0.98828125 0.7421875 0.0078125 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.04296875 0.7421875 0.98828125 0.2734375 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.13671875 0.94140625 0.87890625 0.625 0.421875 0.00390625 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.31640625 0.9375 0.98828125 0.98828125 0.46484375 0.09765625 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.17578125 0.7265625 0.98828125 0.98828125 0.5859375 0.10546875 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0625 0.36328125 0.984375 0.98828125 0.73046875 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.97265625 0.98828125 0.97265625 0.25 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.1796875 0.5078125 0.71484375 0.98828125 0.98828125 0.80859375 0.0078125 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.15234375 0.578125 0.89453125 0.98828125 0.98828125 0.98828125 0.9765625 0.7109375 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.09375 0.4453125 0.86328125 0.98828125 0.98828125 0.98828125 0.98828125 0.78515625 0.3046875 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.08984375 0.2578125 0.83203125 0.98828125 0.98828125 0.98828125 0.98828125 0.7734375 0.31640625 0.0078125 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0703125 0.66796875 0.85546875 0.98828125 0.98828125 0.98828125 0.98828125 0.76171875 0.3125 0.03515625 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.21484375 0.671875 0.8828125 0.98828125 0.98828125 0.98828125 0.98828125 0.953125 0.51953125 0.04296875 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.53125 0.98828125 0.98828125 0.98828125 0.828125 0.52734375 0.515625 0.0625 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ])
(def test-output [0 0 0 0 0 1 0 0 0 0])
















(defn biases-a-change [train-set weights-a biases weights-b biases-b]
  "takes a training set, a vector of biases for the first layer of connections, the rest of the network
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
  "takes a training set, a vector of biases for the second layer of connections, the rest of the network
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
  "takes a training set, a vector of weights for the first layer of connections, the rest of the network's
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
  "takes a training set, a vector of weights for the second layer of connections, the rest of the networks
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
