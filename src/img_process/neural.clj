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

(defn datum-cost [inputs value weights-a biases-a weights-b biases-b]
  (get-cost (evaluate inputs weights-a biases-a weights-b biases-b) value))

(defn get-random-set [samp-n pop-n]
  "returns the line numbers of the random sample set"
  (set (sort (take samp-n (repeatedly #(rand-int pop-n))))))

(defn get-set-values [indices file max-index]
  "returns a collection of lines with maximum max-index for the given indices of the file"
   (with-open [rdr (clojure.java.io/reader file)]
     (let [lines (line-seq rdr)]
       (loop [data [] ct 0]
         (if (> ct (+ 1 max-index))
           data
           (if (not (empty? (filter #(= ct %) indices))) ;;if the ct is one of the selected indices
             (recur
              (conj data (last (take ct lines)))
              (inc ct))
           (do
             (recur
             data
             (inc ct)))))))))

(defn json-2-KV [json-str]
  "converts a json string of an input and correct output to a KV mapping of :input and :result as keys"
  (let [datum (json/read-str json-str)]
    {:input (datum 0) :result (datum 1)}))

(defn extract-set-values [train-strings]
  "takes a vector of strings as JSON representing training inputs and correct results,
  returns them as a vector of collections with :input and :result as keys"
  (map json-2-KV train-strings))

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

(up-ten-percent 0.98)


(defn weights-b-change [train-set weights-a biases-a weights biases-b]
  "takes a training set, a vector of weights for the second layer of connections, the rest of the networks
  settings and returns an identiy matrix where true means that the partial derivative of the total cost
  with respect to that connection's weight is negative [for a step size of 10% of the weight's current
  value] (increase that value to decrease the total cost)"
  )


(defn weights-a-change [train-set weights biases-a weights-b biases-b]
  "takes a training set, a vector of weights for the first layer of connections, the rest of the network's
  settings and returns an identiy matrix where true means that the partial derivative of the total cost
  with respect to that connection's weight is negative [for a step size of 10% of the weight's current
  value] (increase that value to decrease the total cost)"
  (loop [res [] outer-ct 0]
    (if (>= outer-ct (count weights))
      res
      (do
        (loop [res-in [] inner-ct 0]
          (if (< inner-ct (count (weights outer-ct)))
            (let [mutant (assoc weights outer-ct (update-in (weights outer-ct) inner-ct up-ten-percent))]
              (if (>=
                   (set-cost train-set mutant biases-a weights-b biases-b)
                   (set-cost train-set weights biases-a weights-b biases-b)) ;;partial derivative is positive
                (recur
                 (assoc res-in 0)
                 (inc inner-ct))
               (recur
                (assoc res-in 1)
                (inc inner-ct))))
          (assoc res res-in)))
      (recur
       res
       (inc outer-ct))))))

((init-weights-a 1) 1)
(count (init-weights-a 0))
(weights-a-change test-value-inputs init-weights-a init-biases-a init-weights-b init-biases-b)



init-biases-a
(def init-biases-aa [0.56 0.4321580631740959 0.57710040224414 0.6185965659086016 0.18969736398984938 0.8749276162565557 0.7980151147402715 0.664393999821422 0.9569106331833737 0.9815324868138454 0.3432571362384952 0.6501960631145083 0.3426291107100178 0.36863149128690365 0.7569718521150473])

(def indices '(3 1 2 9 43 8))
indices
(def test-values (get-set-values indices "resources/train_hundo.txt" (apply max indices)))
(def test-value-inputs (extract-set-values test-values))
test-values
test-value-inputs
(map #(:result %) test-value-inputs)
(count test-values)

(set-cost test-value-inputs init-weights-a init-biases-a init-weights-b init-biases-b)
(set-cost test-value-inputs init-weights-a init-biases-aa init-weights-b init-biases-b)
(set-cost1 test-value-inputs init-weights-a init-biases-a init-weights-b init-biases-b)
(set-cost2 test-value-inputs init-weights-a init-biases-a init-weights-b init-biases-b)
test-value-inputs


(evaluate (:input (test-value-inputs 1)) init-weights-a init-biases-a init-weights-b init-biases-b)




(def init-weights-a (weights-init 784 15))
(def init-weights-b (weights-init 15 10))

(def init-biases-a (vec (map (fn [_] (rand)) (range 15))))
(def init-biases-b (vec (map (fn [_] (rand)) (range 10))))

(def test-input [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.01171875 0.0703125 0.0703125 0.0703125 0.4921875 0.53125 0.68359375 0.1015625 0.6484375 0.99609375 0.96484375 0.49609375 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.1171875 0.140625 0.3671875 0.6015625 0.6640625 0.98828125 0.98828125 0.98828125 0.98828125 0.98828125 0.87890625 0.671875 0.98828125 0.9453125 0.76171875 0.25 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.19140625 0.9296875 0.98828125 0.98828125 0.98828125 0.98828125 0.98828125 0.98828125 0.98828125 0.98828125 0.98046875 0.36328125 0.3203125 0.3203125 0.21875 0.15234375 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0703125 0.85546875 0.98828125 0.98828125 0.98828125 0.98828125 0.98828125 0.7734375 0.7109375 0.96484375 0.94140625 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.3125 0.609375 0.41796875 0.98828125 0.98828125 0.80078125 0.04296875 0.0 0.16796875 0.6015625 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0546875 0.00390625 0.6015625 0.98828125 0.3515625 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.54296875 0.98828125 0.7421875 0.0078125 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.04296875 0.7421875 0.98828125 0.2734375 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.13671875 0.94140625 0.87890625 0.625 0.421875 0.00390625 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.31640625 0.9375 0.98828125 0.98828125 0.46484375 0.09765625 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.17578125 0.7265625 0.98828125 0.98828125 0.5859375 0.10546875 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0625 0.36328125 0.984375 0.98828125 0.73046875 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.97265625 0.98828125 0.97265625 0.25 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.1796875 0.5078125 0.71484375 0.98828125 0.98828125 0.80859375 0.0078125 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.15234375 0.578125 0.89453125 0.98828125 0.98828125 0.98828125 0.9765625 0.7109375 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.09375 0.4453125 0.86328125 0.98828125 0.98828125 0.98828125 0.98828125 0.78515625 0.3046875 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.08984375 0.2578125 0.83203125 0.98828125 0.98828125 0.98828125 0.98828125 0.7734375 0.31640625 0.0078125 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0703125 0.66796875 0.85546875 0.98828125 0.98828125 0.98828125 0.98828125 0.76171875 0.3125 0.03515625 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.21484375 0.671875 0.8828125 0.98828125 0.98828125 0.98828125 0.98828125 0.953125 0.51953125 0.04296875 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.53125 0.98828125 0.98828125 0.98828125 0.828125 0.52734375 0.515625 0.0625 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ])
(def test-output [0 0 0 0 0 1 0 0 0 0])







