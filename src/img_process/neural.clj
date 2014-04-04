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
  (let [res []]
    (for [i (range (count bias-vec))]
    (conj res (sigmoid (- (dot (weight-vec i) in-vec) (bias-vec i)))))))

(defn sig-out1 [in-vec weight-vec bias-vec]
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


(defn evaluate1 [inputs weights-a biases-a]
  "takes a vector of inputs, biases for the hidden layer (a), biases for the output layer (b),
  weights for the input->a connections, weights for the input->b connections, and produces the
  output vector"
  (sig-out1 inputs weights-a biases-a))

(defn sig-out-neuron [inputs weights-of-in biases-a]
  (sigmoid (- (dot (weights-of-in) inputs) (biases- ))))




(defn datum-cost [inputs value weights-a biases-a weights-b biases-b]
  (get-cost (evaluate inputs weights-a biases-a weights-b biases-b) value))

(def init-weights-a (weights-init 784 15))
(def init-weights-b (weights-init 15 10))

(def init-biases-a (vec (map (fn [_] (rand)) (range 15))))
(def init-biases-b (vec (map (fn [_] (rand)) (range 10))))

(evaluate1 test-input init-weights-a init-biases-a)

(with-open [test-rdr (clojure.java.io/reader "resources/train_hundo.txt")]
  (doseq [line (line-seq test-rdr)]
    (print (datum-cost
            ((json/read-str line) 0)
            ((json/read-str line) 1)
            init-weights-a
            init-biases-a
            init-weights-b
            init-biases-b))))

(with-open [test-rdr (clojure.java.io/reader "resources/train_hundo.txt")]
  (doseq [line (line-seq test-rdr)]
    (print
            ((json/read-str line) 0)
           )))

(def test-input [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.01171875 0.0703125 0.0703125 0.0703125 0.4921875 0.53125 0.68359375 0.1015625 0.6484375 0.99609375 0.96484375 0.49609375 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.1171875 0.140625 0.3671875 0.6015625 0.6640625 0.98828125 0.98828125 0.98828125 0.98828125 0.98828125 0.87890625 0.671875 0.98828125 0.9453125 0.76171875 0.25 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.19140625 0.9296875 0.98828125 0.98828125 0.98828125 0.98828125 0.98828125 0.98828125 0.98828125 0.98828125 0.98046875 0.36328125 0.3203125 0.3203125 0.21875 0.15234375 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0703125 0.85546875 0.98828125 0.98828125 0.98828125 0.98828125 0.98828125 0.7734375 0.7109375 0.96484375 0.94140625 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.3125 0.609375 0.41796875 0.98828125 0.98828125 0.80078125 0.04296875 0.0 0.16796875 0.6015625 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0546875 0.00390625 0.6015625 0.98828125 0.3515625 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.54296875 0.98828125 0.7421875 0.0078125 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.04296875 0.7421875 0.98828125 0.2734375 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.13671875 0.94140625 0.87890625 0.625 0.421875 0.00390625 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.31640625 0.9375 0.98828125 0.98828125 0.46484375 0.09765625 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.17578125 0.7265625 0.98828125 0.98828125 0.5859375 0.10546875 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0625 0.36328125 0.984375 0.98828125 0.73046875 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.97265625 0.98828125 0.97265625 0.25 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.1796875 0.5078125 0.71484375 0.98828125 0.98828125 0.80859375 0.0078125 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.15234375 0.578125 0.89453125 0.98828125 0.98828125 0.98828125 0.9765625 0.7109375 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.09375 0.4453125 0.86328125 0.98828125 0.98828125 0.98828125 0.98828125 0.78515625 0.3046875 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.08984375 0.2578125 0.83203125 0.98828125 0.98828125 0.98828125 0.98828125 0.7734375 0.31640625 0.0078125 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0703125 0.66796875 0.85546875 0.98828125 0.98828125 0.98828125 0.98828125 0.76171875 0.3125 0.03515625 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.21484375 0.671875 0.8828125 0.98828125 0.98828125 0.98828125 0.98828125 0.953125 0.51953125 0.04296875 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.53125 0.98828125 0.98828125 0.98828125 0.828125 0.52734375 0.515625 0.0625 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ])
(def test-output [0 0 0 0 0 1 0 0 0 0])
init-weights-a
(count init-weights-a)
(count (init-weights-a 1))
(evaluate test-input init-weights-a init-biases-a init-weights-b init-biases-b)

























(defn translate-to-vector [value]
  "takes an integer value [0,9] and returns a vectorized representation"
  (assoc (vec (replicate 10 0)) value 1))
