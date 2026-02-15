# %% [code]
from keras.datasets import mnist

import numpy as np

np.random.seed(10)

from IPython.core.display import display, HTML

display(HTML("<style>.container { width: 80% !important; }</style>"))


# %% [code]
from time import time

import numpy as np

import tensorflow as tf

import keras

#import tensorflow.keras.backend as K

from keras.engine.topology import Layer, InputSpec

from keras.layers import Dense, Input

from keras.models import Model

from keras.optimizers import SGD

from keras import callbacks

from keras.initializers import VarianceScaling

from sklearn.cluster import KMeans

import pandas as pd

import metrics

import warnings

from keras import backend as K

# %% [code]
print(keras.__version__)

import warnings

warnings.filterwarnings( 'ignore' )





tf.test.is_gpu_available(

    cuda_only=False,

    min_cuda_compute_capability=None

)

# %% [code]
def autoencoder(dims, act='relu', init='glorot_uniform'):

    """

    Fully connected auto-encoder model, symmetric.

    Arguments:

        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.

            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1

        act: activation, not applied to Input, Hidden and Output layers

    return:

        (ae_model, encoder_model), Model of autoencoder and model of encoder

    """

    n_stacks = len(dims) - 1

    # input

    input_img = Input(shape=(dims[0],), name='input')

    x = input_img

    # internal layers in encoder

    for i in range(n_stacks-1):

        x = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(x)



    # hidden layer

    encoded = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(x)

    

    # hidden layer, features are extracted from here



    x = encoded

    # internal layers in decoder

    for i in range(n_stacks-1, 0, -1):

        x = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(x)



    # output

    x = Dense(dims[0], kernel_initializer=init, name='decoder_0')(x)

    decoded = x

    return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded, name='encoder')

# %% [code]
## ?낅젰?먮즺瑜??뺤씤?섍퀬 input format ?섏젙



trains = np.loadtxt("C:/Users/User/Documents/KOTI/怨꾨갚濡?input_normalize.csv",

                    skiprows=1, delimiter=',', dtype=float)



x = trains[:, 531:795].reshape([len(trains), 264])# ?댁쓽 媛?닔瑜??섏젙

#x2 = trains[:, 531:795].reshape([len(trains), 264])

#x = np.concatenate((x1, x2), axis=1)



print(x.shape)



y_year = trains[:, 0].reshape([len(trains)])

y_weekday = trains[:, 1].reshape([len(trains)])

y = trains[:, 2].reshape([len(trains)])



x = x.reshape((x.shape[0], -1))

#x = np.divide(x, 256.)        #normalized ?곗씠?곗뿉?쒕뒗 ??젣

#x = np.divide(x, 256.)        #normalized ?곗씠?곗뿉?쒕뒗 ??젣

#?띾룄 3-267 援먰넻??267-531 諛??531-795 

#1媛?264 2媛?528 3媛?792

# %% [code]
n_clusters = 5 #len(np.unique(y))  # cluster??媛?닔瑜?吏곸젒 ?좎뼵?댁빞 ??n
x.shape

print ([y_year, y_weekday, y])

# %% [code]
#kmeans = KMeans(n_clusters=n_clusters, n_init=20, n_jobs=4)

#y_pred_kmeans = kmeans.fit_predict(x)

# %% [code]
#metrics.acc(y, y_pred_kmeans)  # ?곕━ ?먮즺?먮뒗 label???놁쑝誘濡?accuracy瑜?援ы븷 ???놁쓬

# %% [code]
#dims = [x.shape[-1], 500, 500, 2000, 10]

dims = [x.shape[-1], 500, 1000, 2000, 3000, 5000, 10]   # hidden layer瑜??대뼸寃?援ъ꽦??寃껋씤吏 怨좊? 

init = VarianceScaling(scale=1. / 3., mode='fan_in',

                           distribution='uniform')

pretrain_optimizer = SGD(lr=0.1, momentum=0.9)

pretrain_epochs = 1000

batch_size = 128

save_dir = './results'

# %% [code]
autoencoder, encoder = autoencoder(dims, init=init)

# %% [code]
autoencoder.summary()

# %% [code]
autoencoder.compile(optimizer=pretrain_optimizer, loss='mse')

history = autoencoder.fit(x, x, batch_size=batch_size, epochs=pretrain_epochs) #, callbacks=cb)

autoencoder.save_weights(save_dir + '/ae_weights.h5')

# %% [code]
import matplotlib.pyplot as plt

def plot_loss(history):

    plt.plot(history.history['loss'])

    plt.title('Model Loss')

    plt.ylabel('Loss')

    plt.xlabel('Epoch')

    plt.legend(['Train'], loc =0)



plot_loss(history)

plt.show()

# %% [code]
decoded_images = autoencoder.predict(x)



print(decoded_images.shape, x.shape)

N, n_i = x.shape

x1 = x.reshape(N, 44, -1)

decoded_images = decoded_images.reshape(N, 44, -1)



n = 20

plt.figure(figsize=(40,8))

for i in range(n):

    ax = plt.subplot(2, n, i+1)

    plt.imshow(x1[i])

    

    ax = plt.subplot(2, n, i+1+n)

    plt.imshow(decoded_images[i])

    

plt.show()

# %% [code]
autoencoder.load_weights(save_dir + '/ae_weights.h5')

# %% [code]
class ClusteringLayer(Layer):

    """

    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the

    sample belonging to each cluster. The probability is calculated with student's t-distribution.



    # Example

    ```

        model.add(ClusteringLayer(n_clusters=10))

    ```

    # Arguments

        n_clusters: number of clusters.

        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.

        alpha: degrees of freedom parameter in Student's t-distribution. Default to 1.0.

    # Input shape

        2D tensor with shape: `(n_samples, n_features)`.

    # Output shape

        2D tensor with shape: `(n_samples, n_clusters)`.

    """



    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:

            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(ClusteringLayer, self).__init__(**kwargs)

        self.n_clusters = n_clusters

        self.alpha = alpha

        self.initial_weights = weights

        self.input_spec = InputSpec(ndim=2)



    def build(self, input_shape):

        assert len(input_shape) == 2

        input_dim = input_shape[1]

        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))

        self.clusters = self.add_weight(shape = (self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')

        if self.initial_weights is not None:

            self.set_weights(self.initial_weights)

            del self.initial_weights

        self.built = True



    def call(self, inputs, **kwargs):

        """ student t-distribution, as same as used in t-SNE algorithm.

         Measure the similarity between embedded point z_i and centroid 쨉_j.

                 q_ij = 1/(1+dist(x_i, 쨉_j)^2), then normalize it.

                 q_ij can be interpreted as the probability of assigning sample i to cluster j.

                 (i.e., a soft assignment)

        Arguments:

            inputs: the variable containing data, shape=(n_samples, n_features)

        Return:

            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)

        """

        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))

        q **= (self.alpha + 1.0) / 2.0

        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure each sample's 10 values add up to 1.

        return q



    def compute_output_shape(self, input_shape):

        assert input_shape and len(input_shape) == 2

        return input_shape[0], self.n_clusters



    def get_config(self):

        config = {'n_clusters': self.n_clusters}

        base_config = super(ClusteringLayer, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

# %% [code]
clustering_layer = ClusteringLayer(n_clusters, name ='clustering')(encoder.output)

model = Model(inputs=encoder.input, outputs=clustering_layer)

model.compile(optimizer=SGD(0.01, 0.9), loss='kld')

kmeans = KMeans(n_clusters=n_clusters, n_init=20)

y_pred = kmeans.fit_predict(encoder.predict(x))

y_pred_last = np.copy(y_pred)

model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])



# computing an auxiliary target distribution

def target_distribution(q):

    weight = q ** 2 / q.sum(0)

    return (weight.T / weight.sum(1)).T



loss = 0

index = 0

maxiter = 20000

update_interval = 140

index_array = np.arange(x.shape[0])

tol = 0.001 # tolerance threshold to stop training

# %% [code]
for ite in range(int(maxiter)):

    if ite % update_interval == 0:

        q = model.predict(x, verbose=0)

        p = target_distribution(q)  # update the auxiliary target distribution p



        # evaluate the clustering performance

        y_pred = q.argmax(1)

        if y is not None:

            acc = np.round(metrics.acc(y, y_pred), 5)

            nmi = np.round(metrics.nmi(y, y_pred), 5)

            ari = np.round(metrics.ari(y, y_pred), 5)

            loss = np.round(loss, 5)

            print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), ' ; loss=', loss)



        # check stop criterion - model convergence

        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]

        y_pred_last = np.copy(y_pred)

        if ite > 0 and delta_label < tol:

            print('delta_label ', delta_label, '< tol ', tol)

            print('Reached tolerance threshold. Stopping training.')

            break

    idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]

    loss = model.train_on_batch(x=x[idx], y=p[idx])

    index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0



model.save_weights(save_dir + '/DEC_model_final.h5')

# %% [code]
model.load_weights(save_dir + '/DEC_model_final.h5')

# %% [code]
# Eval.

############ redefine test data set ##############



trains = np.loadtxt("C:/Users/User/Documents/KOTI/怨꾨갚濡?input_normalize.csv",

                    skiprows=1, delimiter=',', dtype=float)



test = np.loadtxt("C:/Users/User/Documents/KOTI/怨꾨갚濡?input_normalize.csv", skiprows=1, delimiter=',', dtype=float)

x = test[:, 531:795].reshape([len(trains), 264])

#x1 = test[:, 3:267].reshape([len(test), 264])  # ?댁쓽 媛?닔瑜??섏젙

#x2 = test[:, 531:795].reshape([len(test), 264])

#x = np.concatenate((x1, x2), axis=1)

#?띾룄 3-267 援먰넻??267-531 諛??531-795 

#1媛?264 2媛?528 3媛?792

y_year = test[:, 0].reshape([len(test)])

y_weekday = test[:, 1].reshape([len(test)])

y = test[:, 2].reshape([len(test)])



x = x.reshape((x.shape[0], -1))

#x = np.divide(x, 255.)    #normalized ?곗씠?곗뿉?쒕뒗 ??젣





q = model.predict(x, verbose=0)

p = target_distribution(q)  # update the auxiliary target distribution p



# evaluate the clustering performance

y_pred = q.argmax(1)

if y is not None:

    acc = np.round(metrics.acc(y, y_pred), 5)

    nmi = np.round(metrics.nmi(y, y_pred), 5)

    ari = np.round(metrics.ari(y, y_pred), 5)

    loss = np.round(loss, 5)

    print('Acc = %.5f, nmi = %.5f, ari = %.5f' % (acc, nmi, ari), ' ; loss=', loss)

# %% [code]
import seaborn as sns

import sklearn.metrics

import matplotlib.pyplot as plt

sns.set(font_scale=3)

confusion_matrix = sklearn.metrics.confusion_matrix(y, y_pred)



plt.figure(figsize=(16, 14))

sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20});

plt.title("Confusion matrix", fontsize=30)

plt.ylabel('True label', fontsize=25)

plt.xlabel('Clustering label', fontsize=25)

plt.show()

# %% [code]
from sklearn.utils.linear_assignment_ import linear_assignment



y_true = y.astype(np.int64)

D = max(y_pred.max(), y_true.max()) + 1

w = np.zeros((D, D), dtype=np.int64)

# Confusion matrix.

for i in range(y_pred.size):

    w[y_pred[i], y_true[i]] += 1

ind = linear_assignment(-w)



sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

# %% [code]
y_pred = y_pred.reshape(-1, 1)

y = y.reshape(-1, 1)



df = pd.DataFrame(y_year, columns = ['year'])

df['weekday'] = y_weekday

df['true'] = y

df['cluster_id'] = y_pred

df.to_excel('C:/Users/User/Documents/KOTI/怨꾨갚濡?results/result.xlsx', index = False )

# %% [code]

