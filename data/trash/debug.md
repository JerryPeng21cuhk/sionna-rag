
================================================================================
```
import os
gpu_num = 0 # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Import Sionna
try:
    import sionna
except ImportError as e:
    # Install Sionna if package is not already installed
    import os
    os.system("pip install sionna")
    import sionna
# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')
```

```
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import sys
from sionna.utils import BinarySource, QAMSource, ebnodb2no, compute_ser, compute_ber, PlotBER
from sionna.channel import FlatFadingChannel, KroneckerModel
from sionna.channel.utils import exp_corr_mat
from sionna.mimo import lmmse_equalizer
from sionna.mapping import SymbolDemapper, Mapper, Demapper
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
```
================================================================================

================================================================================
It is very easy add spatial correlation to the `FlatFadingChannel` using the `SpatialCorrelation` class. We can, e.g., easily setup a Kronecker (`KroneckerModel`) (or two-sided) correlation model using exponetial correlation matrices (`exp_corr_mat`).

```
# Create transmit and receive correlation matrices
r_tx = exp_corr_mat(0.4, num_tx_ant)
r_rx = exp_corr_mat(0.9, num_rx_ant)
# Add the spatial correlation model to the channel
channel.spatial_corr = KroneckerModel(r_tx, r_rx)
```

Next, we can validate that the channel model applies the desired spatial correlation by creating a large batch of channel realizations from which we compute the empirical transmit and receiver covariance matrices:

```
h = channel.generate(1000000)
# Compute empirical covariance matrices
r_tx_hat = tf.reduce_mean(tf.matmul(h, h, adjoint_a=True), 0)/num_rx_ant
r_rx_hat = tf.reduce_mean(tf.matmul(h, h, adjoint_b=True), 0)/num_tx_ant
# Test that the empirical results match the theory
assert(np.allclose(r_tx, r_tx_hat, atol=1e-2))
assert(np.allclose(r_rx, r_rx_hat, atol=1e-2))
```

Now, we can transmit the same symbols `x` over the channel with spatial correlation and compute the SER:

```
y, h = channel([x, no])
x_hat, no_eff = lmmse_equalizer(y, h, s)
x_ind_hat = symbol_demapper([x_hat, no])
compute_ser(x_ind, x_ind_hat)
```

```
<tf.Tensor: shape=(), dtype=float64, numpy=0.115234375>
```

The result cleary show the negative effect of spatial correlation in this setting. You can play around with the `a` parameter defining the exponential correlation matrices and see its impact on the SER.
================================================================================

================================================================================
Next, we will wrap everything that we have done so far in a Keras model for convenient BER simulations and comparison of model parameters. Note that we use the `@tf.function(jit_compile=True)` decorator which will speed-up the simulations tremendously. See <a class="reference external" href="https://www.tensorflow.org/guide/function">https://www.tensorflow.org/guide/function</a> for further information. You need to enable the <a class="reference external" href="https://nvlabs.github.io/sionna/api/config.html#sionna.Config.xla_compat">sionna.config.xla_compat</a> feature prior to executing the model.

```
sionna.config.xla_compat=True
class Model(tf.keras.Model):
    def __init__(self, spatial_corr=None):
        super().__init__()
        self.n = 1024
        self.k = 512
        self.coderate = self.k/self.n
        self.num_bits_per_symbol = 4
        self.num_tx_ant = 4
        self.num_rx_ant = 16
        self.binary_source = BinarySource()
        self.encoder = LDPC5GEncoder(self.k, self.n)
        self.mapper = Mapper("qam", self.num_bits_per_symbol)
        self.demapper = Demapper("app", "qam", self.num_bits_per_symbol)
        self.decoder = LDPC5GDecoder(self.encoder, hard_out=True)
        self.channel = FlatFadingChannel(self.num_tx_ant,
                                         self.num_rx_ant,
                                         spatial_corr=spatial_corr,
                                         add_awgn=True,
                                         return_channel=True)
    @tf.function(jit_compile=True)
    def call(self, batch_size, ebno_db):
        b = self.binary_source([batch_size, self.num_tx_ant, self.k])
        c = self.encoder(b)
        x = self.mapper(c)
        shape = tf.shape(x)
        x = tf.reshape(x, [-1, self.num_tx_ant])
        no = ebnodb2no(ebno_db, self.num_bits_per_symbol, self.coderate)
        no *= np.sqrt(self.num_rx_ant)
        y, h = self.channel([x, no])
        s = tf.complex(no*tf.eye(self.num_rx_ant, self.num_rx_ant), 0.0)
        x_hat, no_eff = lmmse_equalizer(y, h, s)
        x_hat = tf.reshape(x_hat, shape)
        no_eff = tf.reshape(no_eff, shape)
        llr = self.demapper([x_hat, no_eff])
        b_hat = self.decoder(llr)
        return b,  b_hat
```

We can now instantiate different version of this model and use the `PlotBer` class for easy Monte-Carlo simulations.

```
ber_plot = PlotBER()
```

```
model1 = Model()
ber_plot.simulate(model1,
        np.arange(-2.5, 0.25, 0.25),
        batch_size=4096,
        max_mc_iter=1000,
        num_target_block_errors=100,
        legend="Uncorrelated",
        show_fig=False);
```

```
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
     -2.5 | 1.1113e-01 | 9.1742e-01 |      932264 |     8388608 |        15031 |       16384 |         6.7 |reached target block errors
    -2.25 | 7.9952e-02 | 7.8546e-01 |      670690 |     8388608 |        12869 |       16384 |         0.2 |reached target block errors
     -2.0 | 5.0242e-02 | 5.9320e-01 |      421458 |     8388608 |         9719 |       16384 |         0.2 |reached target block errors
    -1.75 | 2.6710e-02 | 3.7158e-01 |      224061 |     8388608 |         6088 |       16384 |         0.2 |reached target block errors
     -1.5 | 1.1970e-02 | 1.9104e-01 |      100415 |     8388608 |         3130 |       16384 |         0.2 |reached target block errors
    -1.25 | 4.6068e-03 | 7.8186e-02 |       38645 |     8388608 |         1281 |       16384 |         0.2 |reached target block errors
     -1.0 | 1.2861e-03 | 2.5818e-02 |       10789 |     8388608 |          423 |       16384 |         0.2 |reached target block errors
    -0.75 | 2.7883e-04 | 7.8735e-03 |        2339 |     8388608 |          129 |       16384 |         0.2 |reached target block errors
     -0.5 | 8.5314e-05 | 2.2990e-03 |        2147 |    25165824 |          113 |       49152 |         0.7 |reached target block errors
    -0.25 | 1.2082e-05 | 3.1738e-04 |        2027 |   167772160 |          104 |      327680 |         4.4 |reached target block errors
      0.0 | 1.9351e-06 | 7.1681e-05 |        1396 |   721420288 |          101 |     1409024 |        19.0 |reached target block errors
```

```
r_tx = exp_corr_mat(0.4, num_tx_ant)
r_rx = exp_corr_mat(0.7, num_rx_ant)
model2 = Model(KroneckerModel(r_tx, r_rx))
ber_plot.simulate(model2,
        np.arange(0,2.6,0.25),
        batch_size=4096,
        max_mc_iter=1000,
        num_target_block_errors=200,
        legend="Kronecker model");
```

```
EbNo [dB] |        BER |       BLER |  bit errors |    num bits | block errors |  num blocks | runtime [s] |    status
---------------------------------------------------------------------------------------------------------------------------------------
      0.0 | 7.2249e-02 | 7.1844e-01 |      606070 |     8388608 |        11771 |       16384 |         4.9 |reached target block errors
     0.25 | 4.7019e-02 | 5.4126e-01 |      394421 |     8388608 |         8868 |       16384 |         0.2 |reached target block errors
      0.5 | 2.7080e-02 | 3.5498e-01 |      227161 |     8388608 |         5816 |       16384 |         0.2 |reached target block errors
     0.75 | 1.3981e-02 | 2.0380e-01 |      117280 |     8388608 |         3339 |       16384 |         0.2 |reached target block errors
      1.0 | 6.1550e-03 | 9.6802e-02 |       51632 |     8388608 |         1586 |       16384 |         0.2 |reached target block errors
     1.25 | 2.2970e-03 | 4.0100e-02 |       19269 |     8388608 |          657 |       16384 |         0.2 |reached target block errors
      1.5 | 7.9966e-04 | 1.4282e-02 |        6708 |     8388608 |          234 |       16384 |         0.2 |reached target block errors
     1.75 | 2.4907e-04 | 4.5166e-03 |        6268 |    25165824 |          222 |       49152 |         0.7 |reached target block errors
      2.0 | 7.0466e-05 | 1.3767e-03 |        5320 |    75497472 |          203 |      147456 |         2.0 |reached target block errors
     2.25 | 1.4114e-05 | 3.0670e-04 |        4736 |   335544320 |          201 |      655360 |         9.0 |reached target block errors
      2.5 | 2.8881e-06 | 7.0971e-05 |        4167 |  1442840576 |          200 |     2818048 |        38.7 |reached target block errors
```
================================================================================
