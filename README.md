# img-process

This is Clojure project for recognizing images of handwritten characters representing the digits 0-9

The code is broken down into the following sections:
-image_access.clj: this is the vision module for analyzing an image of a block of text, identifying the pixel coordinates
where characters lay in the image, accessing the locations of those characters, re-sizing them to square blocks that in
the proper dimensions that the neural network is configured to receive, and processing those image blocks to prepare a
vectorized representation each character of it's lightness or darkness score on a scale from 0-1.
-neural.clj: this is the main file for defining the neural networks processing and training algorithms
-initial.clj: this is where the neural net reads networks settings from for initializing before training or for processing
an input
-resources/training-log: update the destination in the neural.clj file when beginning training
-resources: training data, sample images to test with

## Usage

To train the network you need to update neural.clj and then you can lein run the project. A live version where can upload
an image to be analyzed will be live soon.

## License

Copyright © 2014

/*
 * ----------------------------------------------------------------------------
 * "THE BEER-WARE LICENSE":
 * <zachmarkin@gmail.com> wrote this file. As long as you retain this notice you
 * can do whatever you want with this stuff. If we meet some day, and you think
 * this stuff is worth it, you can buy me a beer in return, Zachary Markin
 * ----------------------------------------------------------------------------
 */
