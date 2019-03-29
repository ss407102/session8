# Session 8
 Paper reading session Very Deep Convolutional Networks For Large-Scale Image Recognition by Karen Simonyan & Andrew Zisserman+ -2014
 ## Resources
 Link to paper -- [Very Deep Convolutional Networks For Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)
 ## Summary
  * Introduction

   * Accuracy of model can be increased significantly by increasing depth (16-19) layers model and using small (3*3) convolution filters.
   * This model basically improves Alexnet model by using small filters,small strides and using better cropping and scaling.

* Architecture

 * Configuration :
   *  Models with increasing depth by increasing  conv layers.
   *  Fixed no of FC layers (3).
   *  A-E models A-11 layers model and E-19 weight layer model.
 * Input :
   *  fixed size 224 x 224 rgb image.
 * Preprocessing -
   * subtracting the mean rgb value.
 * Overview :-
    * Five (2 x 2) Max pool layers with stride=2.
    * 3 FC layers ( First two - 4096 channels).
    * Third FC layer - 1000 way softmax.
 * Key Points :-
    *  (1 x 1) filters are used followed by non-linear function to give non-linearity to input channels.
    * Padding = 1 pixel for (3*3) conv layers.
    *  LRN did not worked well in improving accuracy so LRN was only used once in A-LRN (11 layers) model.
    *  Relu activation function was used for all hidden layers.


* Discussion

 * Alexnet (11 × 11 with stride 4).
 * 7 × 7 with stride 2 in (Zeiler & Fergus,2013; Sermanet et al., 2014))
 *  Above 2 models uses large receptive field.
 *   VGG uses small receptive field of (3 x 3) with stride 1.
 * Stack of two 3 x 3 layers without pooling equal to (5 x 5) receptive field.
 * Stack of three 3 x 3 layers without pooling is equal to  7 x 7 receptive field.
 * Above step makes decision function more discriminative(3 non-linear relu)
  * Decreases parameters by approx 81 %.
 * 1 x 1 is used to increase the non-linearity of the decision function without affecting the receptive fields of the conv. Layers.


* Training

   * Trained on multi scale training images(diff from AlexNet).
   * Overview :-
     * Optimizer=multinomial logistic regression.
    *  Mini Batch gradient descent (batch size=256).
    *  Momentum=0.9.
    * L2 penalty multiplier=5 x 10 −4.
    *  Dropout=0.5(first two FC layer).
   * Learning rate=Initially set to 10 −2 , and then decreased by a factor of 10 when the validation set accuracy stopped improving.
   * learning rate was decreased 3 times and learning was stopped after 370K iterations (74 epochs).
   * Takes less epochs to coverge due to -
    * Implicit regularisation imposed by greater depth and smaller conv.filter sizes.
    * pre-initialisation of certain layers.

 * Initialization -
   *  Deeper Model was initialized by using layers of model A(trained with random initialization.
    * Learning rate was not changed for these layers.
    * Remaining layers initialized randomly using concept similar to Glorot & Bengio initializer (2010).
  * Random cropping (224 x 224) and horizontal flipping,RGB color shift  for data augmentation.
  * Trained on two fixed scale s=256 and s=384 and multi scale chosen from range [256,512].


* Testing

 *  Rescaled to smallest side Q (!=S).
 * Variable resolution :-
   *   Converting model to Fully conv models.
   *  1st FC 7 x 7 and other two 1 x 1 .
   *  Then sum pooled for fixed size vector.
 * Horizontal flip and averaging them.
 * Key Points :-
   * Using more than one values of Q for same S increased accuracy.
   *  Multi crop and dense are complementary due to increase in receptive field.


* Implementation Details

 * Each batch trained parallelly on multiple GPU’s and then taking average to get gradients of full batch.


* Single  Scale Evaluation

 * Model with (3 x 3) perform better than (1 x 1)  so it is important to capture spatial context.
 * Replacing two (3 x 3) by single (5 x 5)  increases error by 7%.
 * Scale Jittering is helpful for capturing multi-scale image statistics.
 * Q = S for fixed S, and Q = 0.5 x (S min + S max ) for jittered S ∈ [S min , S max ].


 * Multi Scale Evaluation

  * Evaluation :-
    * Evaluated over three test image sizes close to training one.
    * For fixed S: Q ={S − 32, S, S + 32}.
    * For variable S: Q = {  Smin  , 0.5(Smin + Smax)     , Smax    } .
    * Result = average of 3 rescaled version.
  * Key Points :-
    * Scale jittering at training time allows the network to be applied to a wider range of scales at test time.
    * 16 and 19 layers model’s error decreases by 0.7% as compared to single scale.
    * Scale jittering, both at test time and training time, leads to better performance.

* Multi Crop Evaluation

  * Multi crop (error=24.6) is better than dense (error=24.8).
  * Multi crop and dense (error=24.4).


* ConvNet Fusion

    * ensemble of 7 networks has 7.3% error.
    * ensembling best two (D and E)  error=7% dense and multi+dense error =6.8%.
(GoogleNet error=6.7%).

**  In terms of the single-net performance, this architecture achieves the best
result (7.0% test error).**
