import os
import sys
import cv2
import numpy as np 
import tensorflow as tf 
import tensorflow.keras as keras
import tensorflow.keras.backend as keras_backend

os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'

class FeatureCorrelationMatchingModule:
    def __init__(self):
        self.treshold = 0.6
        self.Lambda = 2

    def distance_func(self, pixel_mat, feature_map):
        '''
        :param pixel_mat: [P_i, P_I, ...] N*N matrix with depth of M
        :param feature_map: N x N x M Feature map
        '''
        pixel_matching_value = tf.negative(
            tf.math.top_k(
                tf.negative(
                    tf.reduce_mean(
                        tf.abs(
                            tf.subtract(pixel_mat, feature_map)
                        ), axis=1
                    )
                ), k=3, sorted=True
            ).values
        )
        return self.metric_func(pixel_matching_value)

    
    def metric_func(self, pixel_matching_value):
        '''
        :param pixel_matching_value: A tf tensor: [Pci,i, Pci,j, Pci,k] according to Eq (5) in paper
                                     P_j, P_k are two points which has the second and third-smallest negative correlation coefficient to P_i
        '''
        if tf.truediv(pixel_matching_value[1],pixel_matching_value[2]) < self.treshold:
            metric_value = tf.truediv(2., tf.add(1., tf.exp(pixel_matching_value[1])))
        else:
            metric_value = tf.truediv(2., tf.add(1., tf.multiply(self.Lambda, tf.exp(pixel_matching_value[1]))))

        return metric_value


    def __call__(self, feature_map):
        '''
        :param feature_map: Output of Feature extractor [batchsz, N, N, M]
        '''
        size = feature_map.shape
        # Setting up the NxNxM size according to the paper
        N = size[1]
        M = size[-1]
        matching_maps = []
        # Unpack batch dimension of feature maps tensor
        feature_map = tf.unstack(feature_map, axis=0)
        for fmap in feature_map:  
            matching_map = []
            # fmap: feature map for each image in a batch
            # Reshape feature of size [N, N, M] to size [N*N, M]
            fmap = tf.reshape(fmap, (N*N, M))
            # Unpack the first dimension [(1, M)...]
            flatten_fmap = tf.unstack(fmap, axis=0)
            # For each pixel in NxN feature map
            for pixel in flatten_fmap:
                # For each pixel with M-dimension features values
                pixel = tf.reshape(pixel, [1, M])  # Reshape ot to [1, M] for replicating
                # Replicate pixel M-dimension feature values to a [NxN, M] matrix according to the Eq (4) in paper
                pixel_mat = tf.tile(pixel, [N*N, 1])
                # Eq (4) in paper compute
                pixel_metric = self.distance_func(pixel_mat, flatten_fmap)
                matching_map.append(pixel_metric)
            matching_map = tf.stack(matching_map, axis=0)
            matching_map = tf.reshape(matching_map, (N, N))
            matching_maps.append(matching_map)
        matching_maps = tf.stack(matching_maps, axis=0)
        print (matching_maps.shape)
        return matching_maps




class PreProcessingBlock(tf.keras.layers.Layer):
    def __init__(self):
        super(PreProcessingBlock, self).__init__()
        self.conv = tf.keras.layers.Conv2D(
            filters=24, kernel_size=(9, 9),
            strides=(1, 1), padding='SAME', kernel_initializer='glorot_normal',
        )
        self.batch_normal = tf.keras.layers.BatchNormalization(
            axis=-1, epsilon=1e-5
        )
        self.avg_pool = tf.keras.layers.AveragePooling2D(
            pool_size=(2,2), strides=2
        )
    def call(self, inputs, training=None, **kwargs):
        x = tf.keras.activations.relu(self.batch_normal(self.conv(inputs), training=True))
        x = self.avg_pool(x)
        return x

class TransitionBlock(tf.keras.layers.Layer):
    def __init__(self, out_channels=None):
        super(TransitionBlock, self).__init__()
        self.conv = tf.keras.layers.Conv2D(
            filters=out_channels, kernel_size=(1,1),
            strides=(1, 1), padding='SAME',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01),
            bias_initializer='zeros'
        )
        self.batch_normal = tf.keras.layers.BatchNormalization(
            axis=-1, epsilon=1e-5
        )
        self.avg_pool = tf.keras.layers.AveragePooling2D(
            pool_size=(2, 2), strides=2
        )
    def call(self, inputs, training=None, **kwargs):
        x = tf.keras.activations.relu(self.batch_normal(self.conv(inputs), training=training))
        x = self.avg_pool(x)
        return x

class DenseInceptionBottleNeck(tf.keras.layers.Layer):
    def __init__(self, growth_rate=6, incept_size=None):
        super(DenseInceptionBottleNeck, self).__init__()
        # Layers for branch 1
        self.br1_conv_1 = tf.keras.layers.Conv2D(
            filters=4 * growth_rate, kernel_size=(1,1),
            strides=(1, 1), padding='SAME',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01),
            bias_initializer='zeros'
        )
        self.br1_batch_normal_1 = tf.keras.layers.BatchNormalization(
            axis=1, epsilon=1e-5
        )
        self.br1_conv_2 = tf.keras.layers.Conv2D(
            filters=growth_rate, kernel_size=incept_size[0],
            strides=(1, 1), padding='SAME',
            kernel_initializer='glorot_normal',
            bias_initializer='zeros'
        )
        self.br1_batch_normal_2 = tf.keras.layers.BatchNormalization(
            axis=1, epsilon=1e-5
        )
        # Layers for branch 2
        self.br2_conv_1 = tf.keras.layers.Conv2D(
            filters=4 * growth_rate, kernel_size=(1,1),
            strides=(1, 1), padding='SAME',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01),
            bias_initializer='zeros'
        )
        self.br2_batch_normal_1 = tf.keras.layers.BatchNormalization(
            axis=1, epsilon=1e-5
        )
        self.br2_conv_2 = tf.keras.layers.Conv2D(
            filters=growth_rate, kernel_size=incept_size[1],
            strides=(1, 1), padding='SAME',
            kernel_initializer='glorot_normal',
            bias_initializer='zeros'
        )
        self.br2_batch_normal_2 = tf.keras.layers.BatchNormalization(
            axis=1, epsilon=1e-5
        )
    def call(self, inputs, training=None, **kwargs):
        # Branch 1 
        br1 = tf.keras.activations.relu(self.br1_batch_normal_1(self.br1_conv_1(inputs), training=training))
        br1 = tf.keras.activations.relu(self.br1_batch_normal_2(self.br1_conv_2(br1), training=training))
        # Branch 2
        br2 = tf.keras.activations.relu(self.br2_batch_normal_1(self.br2_conv_1(inputs), training=training))
        br2 = tf.keras.activations.relu(self.br2_batch_normal_2(self.br2_conv_2(br2), training=training))
        # Concat two branchs
        out = tf.concat(values=[br1, br2], axis=-1)
        
        return out


class PyramidFeatureExtractorBlock(tf.keras.layers.Layer):
    def __init__(self, num_layers, growth_rate, incept_size=None):
        super(PyramidFeatureExtractorBlock, self).__init__()
        self.growth_rate = growth_rate
        self.incept_size = incept_size
        self.num_layers = num_layers
        self.feature_list = []

    def _make_layer(self, x, training):
        y = DenseInceptionBottleNeck(growth_rate=self.growth_rate, incept_size=self.incept_size)(x, training=training)
        self.feature_list.append(y)
        y = tf.concat(self.feature_list, axis=-1)
        return y

    def call(self, inputs, training=None, **kwargs):
        self.feature_list.append(inputs)
        x = self._make_layer(inputs, training=training)
        for i in range(1, self.num_layers):
            x = self._make_layer(x, training=training)
        self.feature_list.clear()
        return x

class PyramidFeatureExtractorModule(tf.keras.models.Model):
    def __init__(self, growth_rate=6):
        super(PyramidFeatureExtractorModule, self).__init__()
        self.pre_process = PreProcessingBlock()
        self.transition_1 = TransitionBlock(out_channels=36)
        self.pfe_1 = PyramidFeatureExtractorBlock(num_layers=4, growth_rate=growth_rate, incept_size=[(5,5), (7,7)])
        self.transition_2 = TransitionBlock(out_channels=48)
        self.pfe_2 = PyramidFeatureExtractorBlock(num_layers=5, growth_rate=growth_rate, incept_size=[(3,3), (5,5)])
        self.transition_3 = TransitionBlock(out_channels=60)
        self.pfe_3 = PyramidFeatureExtractorBlock(num_layers=6, growth_rate=growth_rate, incept_size=[(1,1), (3,3)])
        
    def call(self, inputs, training=None, **kwargs):
        x = self.pre_process(inputs, training=training)
        x = self.pfe_1(x, training=training)
        feature_map_1 = self.transition_1(x, training=training)
        x = self.pfe_2(feature_map_1, training=training)
        feature_map_2 = self.transition_2(x, training=training)
        x = self.pfe_3(feature_map_2, training=training)
        feature_map_3 = self.transition_3(x, training=training)

        return feature_map_1, feature_map_2, feature_map_3



if __name__ == '__main__':
    pfe_module = PyramidFeatureExtractorModule()
    
    image = cv2.resize(cv2.cvtColor(cv2.imread('test.jpg'), cv2.COLOR_BGR2RGB).astype(np.float32)/255, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    image = tf.convert_to_tensor(image)
    image = tf.reshape(image, (1, 256, 256, 3))
    print (image.shape)
    feature_1, feature_2, feature_3 = pfe_module(image, training=True)
    print (feature_1.shape)
    print (feature_2.shape)
    print (feature_3.shape)
    # pfe_module.summary()
    print (pfe_module.trainable_variables)
    FeatureCorrelationMatchingModule()(feature_1)