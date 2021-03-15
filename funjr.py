# Se calculan las metricas de interes a partir de las predicciones y los valores reales
def calculate_metrics(prediction, y_test):
  cm = confusion_matrix(y_test, prediction)
  total = sum(sum(cm))

  specificity = cm[0,0]/(cm[0,0]+cm[0,1])
 
  # sensitivity = recall
  sensitivity = cm[1,1]/(cm[1,0]+cm[1,1])
 
  balanced_accuracy = (sensitivity + specificity)/2
 
  precision = cm[1,1] / (cm[1,1] + cm[0,1])
 
  f1 = 2 * ((precision * sensitivity) / (precision + sensitivity))

  metrics = [specificity, sensitivity, balanced_accuracy, f1]

  # Retorna vector con las metricas
  return metrics


def bothSam(x_train, y_train, over_percentage):
  # Se define la estratedia de remuestreo
  oversample = RandomOverSampler(sampling_strategy= over_percentage)
  
  # Se aplica la transformación
  x_train_pre, y_train_pre = oversample.fit_resample(np.array(x_train).reshape(-1, 1), y_train)
  
  # Se define la estratedia de remuestreo
  undersample = RandomUnderSampler(sampling_strategy='majority')
  
  # Se aplica la transformación
  x_train_both, y_train_both = undersample.fit_resample(x_train_pre, y_train_pre)
  
  return x_train_both[0:len(x_train_both),0], y_train_both


def dot_product(x, kernel):
  """
  Wrapper for dot product operation, in order to be compatible with both
  Theano and Tensorflow
  Args:
    x (): input
    kernel (): weights
  Returns:
  """
  if K.backend() == 'tensorflow':
    return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
  else:
    return K.dot(x, kernel)


class AttentionWithContext(Layer):

  def __init__(self,
               W_regularizer=None, u_regularizer=None, b_regularizer=None,
               W_constraint=None, u_constraint=None, b_constraint=None,
               bias=True, **kwargs):

    self.supports_masking = True
    self.init = initializers.get('glorot_uniform')

    self.W_regularizer = regularizers.get(W_regularizer)
    self.u_regularizer = regularizers.get(u_regularizer)
    self.b_regularizer = regularizers.get(b_regularizer)

    self.W_constraint = constraints.get(W_constraint)
    self.u_constraint = constraints.get(u_constraint)
    self.b_constraint = constraints.get(b_constraint)

    self.bias = bias
    super(AttentionWithContext, self).__init__(**kwargs)

  def build(self, input_shape):
    assert len(input_shape) == 3

    self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                             initializer=self.init,
                             name='{}_W'.format(self.name),
                             regularizer=self.W_regularizer,
                             constraint=self.W_constraint)
    if self.bias:
        self.b = self.add_weight(shape=(input_shape[-1],),
                                 initializer='zero',
                                 name='{}_b'.format(self.name),
                                 regularizer=self.b_regularizer,
                                 constraint=self.b_constraint)

    self.u = self.add_weight(shape=(input_shape[-1],),
                             initializer=self.init,
                             name='{}_u'.format(self.name),
                             regularizer=self.u_regularizer,
                             constraint=self.u_constraint)

    super(AttentionWithContext, self).build(input_shape)

  def compute_mask(self, input, input_mask=None):
    # do not pass the mask to the next layers
    return None

  def call(self, x, mask=None):
    uit = dot_product(x, self.W)

    if self.bias:
      uit += self.b

    uit = K.tanh(uit)
    ait = dot_product(uit, self.u)

    a = K.exp(ait)

    # apply mask after the exp. will be re-normalized next
    if mask is not None:
      # Cast the mask to floatX to avoid float64 upcasting in theano
      a *= K.cast(mask, K.floatx())

    # in some cases especially in the early stages of training the sum may be almost zero
    # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
    # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
    a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

    a = K.expand_dims(a)
    weighted_input = x * a
    return K.sum(weighted_input, axis=1)

  def compute_output_shape(self, input_shape):
    return input_shape[0], input_shape[-1]




