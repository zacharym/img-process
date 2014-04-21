(ns img-process.test.neural
  (:require [clojure.test :refer :all]
            [clojure.test.check :as tc]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [img-process.neural :refer :all]))

;;Sigmoid Gnerative Tests
(def sig-greater-0
  (prop/for-all [z gen/ratio]
                (let [sig-out (sigmoid z)]
                  (< 0 sig-out))))

(def sig-less-1
  (prop/for-all [z gen/ratio]
                (let [sig-out (sigmoid z)]
                  (> 1 sig-out))))

(def sig-less-equal-1
  (prop/for-all [z gen/ratio]
                (let [sig-out (sigmoid z)]
                  (>= 1 sig-out))))

(def sig-pos-deriv
  (prop/for-all [z1 gen/ratio z2 gen/ratio]
                (let [sig-out-1 (sigmoid z1)
                      sig-out-2 (sigmoid z2)]
                  (if (< z1 z2)
                    (< sig-out-1 sig-out-2)
                    (>= sig-out-1 sig-out-2)))))

(tc/quick-check 10000 sig-greater-0)
(tc/quick-check 10000 sig-less-1)
(tc/quick-check 10000 sig-less-equal-1)
(tc/quick-check 10000 sig-pos-deriv)














































(deftest sigmoid-test
  (testing "sigmoid function"
    (is (= (sigmoid 0) 0.5))
    (is (< (sigmoid 24) 1))
    (is (> (sigmoid -24) 0))))

(deftest dot-test
  (testing "dot product"
    (is (= (dot [1 2 9] [3 4 5]) 56))
    (is (= (dot [0 0 0] [1 1 1]) 0))))

(deftest vec-length-test
  (testing "vector length function"
    (is (= (vec-length [3 4]) 5))
    (is (= (vec-length [0 0 0]) 0))))

(deftest get-cost-test
  (testing "cost function"
    (is (< 0 (get-cost [(rand 1) (rand 1) (rand 1) (rand 1)][0.1 0.4 1.435 4.3])))
    (is (= (round-3 (get-cost [0 0 0] [1 1 1])) 1.50))
    (is (= (round-3 (get-cost [0 0 0 1 0 0 0 0 0] [2 -1 2.4 0.2 9 4 2 1 3.2])) 61.82))))

(deftest sig-out-test
  (testing "out signal function"
    (is (= (map round-3 (sig-out [0.334 -0.269 -0.994] [[-0.69 -0.8 -0.44]
                                               [-0.52 0.43 -0.51]
                                               [1.65 1.11 1.52]
                                               [1.79 0.31 1.19]] [0.061 -1.1 2.3]) [1.857 0.722 3.283])))))

(gen/sample (gen/choose -4 4))

;;(deftest sig-out-test
  ;;(testing "signal out function"
    ;;(is (= (sig-out [])))))


;;(defn sig-out [in-vec weight-vec bias-vec]
  ;;"takes a vector of inputs, a vector of neurons defined by the bias-vector, and corresponding 2-D
  ;;vector of weights for the connections between those two. Returns the output of a layer of sigmoids
  ;;defined by the bias vector"
    ;;(vec (map #(sigmoid (- (dot (weight-vec %) in-vec) (bias-vec %))) (range (count bias-vec)))))

