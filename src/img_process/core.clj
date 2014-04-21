(ns img-process.core
  (:require [img-process.image-access :as ims]))

(def text (ims/load-image-resource "resources/written.jpg"))

