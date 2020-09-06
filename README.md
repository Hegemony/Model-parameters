# Model-parameters


## Dependencies
* [Python 3.5+](https://www.continuum.io/downloads)
* [PyTorch 0.4.0+](http://pytorch.org/)

### 对于Sequential实例中含模型参数的层，我们可以通过Module类的parameters()或者named_parameters方法来访问所有参数（以迭代器的形式返回），后者除了返回参数Tensor外还会返回其名字。
