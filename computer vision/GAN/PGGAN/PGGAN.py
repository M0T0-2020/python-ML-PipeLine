import math
from numpy import prod

import torch
import torch.nn as nn
import torch.nn.functional as F

def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

def miniBatchStdDev(x, subGroupSize=4):
    r"""
    Add a minibatch standard deviation channel to the current layer.
    In other words:
        1) Compute the standard deviation of the feature map over the minibatch
        2) Get the mean, over all pixels and all channels of thsi ValueError
        3) expand the layer and cocatenate it with the input
    Args:
        - x (tensor): previous layer
        - subGroupSize (int): size of the mini-batches on which the standard deviation
        should be computed
    """
    size = x.size()
    subGroupSize = min(size[0], subGroupSize)
    if size[0] % subGroupSize != 0:
        subGroupSize = size[0]
    G = int(size[0] / subGroupSize)
    if subGroupSize > 1:
        y = x.view(-1, subGroupSize, size[1], size[2], size[3])
        y = torch.var(y, 1)
        y = torch.sqrt(y + 1e-8)
        y = y.view(G, -1)
        y = torch.mean(y, 1).view(G, 1)
        y = y.expand(G, size[2]*size[3]).view((G, 1, 1, size[2], size[3]))
        y = y.expand(G, subGroupSize, -1, -1, -1)
        y = y.contiguous().view((-1, 1, size[2], size[3]))
    else:
        y = torch.zeros(x.size(0), 1, x.size(2), x.size(3), device=x.device)

    return torch.cat([x, y], dim=1)


class NormalizationLayer(nn.Module):

    def __init__(self):
        super(NormalizationLayer, self).__init__()

    def forward(self, x, epsilon=1e-8):
        return x * (((x**2).mean(dim=1, keepdim=True) + epsilon).rsqrt())

class PseudoLayer(nn.Module):

    def __init__(self):
        super(PseudoLayer, self).__init__()

    def forward(self, x):
        return x

def Upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1:
        return x
    s = x.size()
    x = x.view(-1, s[1], s[2], 1, s[3], 1)
    x = x.expand(-1, s[1], s[2], factor, s[3], factor)
    x = x.contiguous().view(-1, s[1], s[2] * factor, s[3] * factor)
    return x


def getLayerNormalizationFactor(x):
    r"""
    Get He's constant for the given layer
    https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
    """
    size = x.weight.size()
    fan_in = prod(size[1:])

    return math.sqrt(2.0 / fan_in)


class ConstrainedLayer(nn.Module):
    r"""
    A handy refactor that allows the user to:
    - initialize one layer's bias to zero
    - apply He's initialization at runtime
    """

    def __init__(self,
                 module,
                 equalized=True,
                 lrMul=1.0,
                 initBiasToZero=True):
        r"""
        equalized (bool): if true, the layer's weight should evolve within
                         the range (-1, 1)
        initBiasToZero (bool): if true, bias will be initialized to zero
        """

        super(ConstrainedLayer, self).__init__()

        self.module = module
        self.equalized = equalized

        if initBiasToZero:
            self.module.bias.data.fill_(0)
        if self.equalized:
            self.module.weight.data.normal_(0, 1)
            self.module.weight.data /= lrMul
            self.weight = getLayerNormalizationFactor(self.module) * lrMul

    def forward(self, x):

        x = self.module(x)
        if self.equalized:
            x *= self.weight
        return x


class EqualizedConv2d(ConstrainedLayer):

    def __init__(self,
                 nChannelsPrevious,
                 nChannels,
                 kernelSize,
                 padding=0,
                 bias=True,
                 **kwargs):
        r"""
        A nn.Conv2d module with specific constraints
        Args:
            nChannelsPrevious (int): number of channels in the previous layer
            nChannels (int): number of channels of the current layer
            kernelSize (int): size of the convolutional kernel
            padding (int): convolution's padding
            bias (bool): with bias ?
        """

        ConstrainedLayer.__init__(self,
                                  nn.Conv2d(nChannelsPrevious, nChannels,
                                            kernelSize, padding=padding,
                                            bias=bias),
                                  **kwargs)

class ResidualBlock(nn.Module):

    def __init__(self):
        super(ResidualBlock, self).__init__()

    def forward(self, x):
        return x


class EqualizedLinear(ConstrainedLayer):

    def __init__(self,
                 nChannelsPrevious,
                 nChannels,
                 bias=True,
                 **kwargs):
        r"""
        A nn.Linear module with specific constraints
        Args:
            nChannelsPrevious (int): number of channels in the previous layer
            nChannels (int): number of channels of the current layer
            bias (bool): with bias ?
        """

        ConstrainedLayer.__init__(self,
                                  nn.Linear(nChannelsPrevious, nChannels,
                                  bias=bias), **kwargs)


class GNet(nn.Module):

    def __init__(self,
                 dimLatent,
                 depthScale0,
                 initBiasToZero=True,
                 leakyReluLeak=0.2,
                 normalization=True,
                 generationActivation=None,
                 dimOutput=3,
                 equalizedlR=True):
        r"""
        Build a generator for a progressive GAN model
        Args:
            - dimLatent (int): dimension of the latent vector
            - depthScale0 (int): depth of the lowest resolution scales
            - initBiasToZero (bool): should we set the bias to zero when a
                                    new scale is added
            - leakyReluLeak (float): leakyness of the leaky relu activation
                                    function
            - normalization (bool): normalize the input latent vector
            - generationActivation (function): activation function of the last
                                               layer (RGB layer). If None, then
                                               the identity is used. exp -> sigmoid, tanh
            - dimOutput (int): dimension of the output image. 3 -> RGB, 1 -> grey levels
            - equalizedlR (bool): set to true to initiualize the layers with
                                  N(0,1) and apply He's constant at runtime
        """
        super(GNet, self).__init__()

        self.equalizedlR = equalizedlR
        self.initBiasToZero = initBiasToZero

        # Leaky relu activation
        self.leakyRelu = torch.nn.LeakyReLU(leakyReluLeak)
        # upsample2d
        self.upsample2d = nn.UpsamplingNearest2d(scale_factor=2)

        # Last layer activation function
        self.generationActivation = generationActivation
        self.depthScale0 = depthScale0

        # Initalize the scales
        self.scalesDepth = [depthScale0]
        self.scaleLayers = nn.ModuleList()
        self.toRGBLayers = nn.ModuleList()

        # Initialize the scale 0
        self.initFormatLayer(dimLatent)
        self.dimOutput = dimOutput

        # normalization
        self.normalizationLayer = PseudoLayer()
        if normalization:
            self.normalizationLayer = NormalizationLayer()


        self.groupScale0 = nn.Sequential(
            EqualizedConv2d(depthScale0, depthScale0, 3,
                            equalized=equalizedlR, initBiasToZero=initBiasToZero,padding=1),
            self.leakyRelu,
            self.normalizationLayer,  
        )

        self.toRGBLayers.append(EqualizedConv2d(depthScale0, self.dimOutput, 1,
                            equalized=equalizedlR, initBiasToZero=initBiasToZero))
        # Initalize the upscaling parameters
        # alpha : when a new scale is added to the network, the previous
        # layer is smoothly merged with the output in the first stages of
        # the training
        self.alpha = 0


    def initFormatLayer(self, dimLatentVector):
        r"""
        The format layer represents the first weights applied to the latent
        vector. It converts a 1xdimLatent input into a 4 x 4 xscalesDepth[0]
        layer.
        """

        self.dimLatent = dimLatentVector
        self.formatLayer = nn.Sequential(
            EqualizedLinear(self.dimLatent, 16 * self.scalesDepth[0], equalized=self.equalizedlR,
                            initBiasToZero=self.initBiasToZero), self.leakyRelu,  )

    def getOutputSize(self):
        r"""
        Get the size of the generated image.
        """
        side = 4 * (2**(len(self.toRGBLayers) - 1))
        return (side, side)

    def addScale(self, depthNewScale, layerType="single"):
        r"""
        Add a new scale to the model. Increasing the output resolution by
        a factor 2
        Args:
            - depthNewScale (int): depth of each conv layer of the new scale
        """
        depthLastScale = self.scalesDepth[-1]

        self.scalesDepth.append(depthNewScale)
        if layerType=='single':
            net = nn.Sequential(
                self.upsample2d,

                EqualizedConv2d(depthLastScale, depthNewScale, 3, padding=1,
                                equalized=self.equalizedlR,initBiasToZero=self.initBiasToZero),
                self.leakyRelu,
                self.normalizationLayer,
            )
        else:
            net = nn.Sequential(
                self.upsample2d,

                EqualizedConv2d(depthLastScale, depthNewScale, 3, padding=1,
                                equalized=self.equalizedlR,initBiasToZero=self.initBiasToZero),
                self.leakyRelu,
                self.normalizationLayer,  
                
                EqualizedConv2d(depthNewScale, depthNewScale, 3, padding=1, 
                                equalized=self.equalizedlR, initBiasToZero=self.initBiasToZero),
                self.leakyRelu,
                self.normalizationLayer,  
            )
        self.scaleLayers.append(net)

        self.toRGBLayers.append(EqualizedConv2d(depthNewScale, self.dimOutput, 1,
                                                equalized=self.equalizedlR, initBiasToZero=self.initBiasToZero))

    def setNewAlpha(self, alpha):
        r"""
        Update the value of the merging factor alpha
        Args:
            - alpha (float): merging factor, must be in [0, 1]
        """

        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be in [0,1]")

        if not self.toRGBLayers:
            raise AttributeError("Can't set an alpha layer if only the scale 0"
                                 "is defined")

        self.alpha = alpha

    def forward(self, x):

        ## Normalize the input ?
        x = self.normalizationLayer(x)
        # flatten
        x = x.view(-1, num_flat_features(x))
        # format layer
        x = self.formatLayer(x)
        x = x.view(x.size()[0], -1, 4, 4)
        x = self.normalizationLayer(x)

        # Scale 0 (no upsampling)
        x = self.groupScale0(x)

        # Dirty, find a better way
        if self.alpha > 0 and len(self.scaleLayers) == 1:
            y = self.upsample2d( self.toRGBLayers[-2](x) )

        # Upper scales
        for scale, layerGroup in enumerate(self.scaleLayers, 0):
            x = layerGroup(x)
            if self.alpha > 0 and scale == (len(self.scaleLayers) - 2):
                y = self.toRGBLayers[-2](x)
                y = self.upsample2d(y)

        # To RGB (no alpha parameter for now)
        x = self.toRGBLayers[-1](x)

        # Blending with the lower resolution output when alpha > 0 and self.scaleLayers exists
        if self.alpha > 0 and len(self.scaleLayers) >= 1:
            #print('add residual')
            x = self.alpha * y + (1.0-self.alpha) * x

        if self.generationActivation is not None:
            x = self.generationActivation(x)

        return x


class DNet(nn.Module):

    def __init__(self,
                 depthScale0,
                 initBiasToZero=True,
                 leakyReluLeak=0.2,
                 sizeDecisionLayer=1,
                 miniBatchNormalization=False,
                 generationActivation=None,
                 dimInput=3,
                 equalizedlR=True):
        r"""
        Build a discriminator for a progressive GAN model
        Args:
            - depthScale0 (int): depth of the lowest resolution scales
            - initBiasToZero (bool): should we set the bias to zero when a
                                    new scale is added
            - leakyReluLeak (float): leakyness of the leaky relu activation
                                    function
            - decisionActivation: activation function of the decision layer. If
                                  None it will be the identity function.
                                  For the training stage, it's advised to set
                                  this parameter to None and handle the
                                  activation function in the loss criterion.
            - sizeDecisionLayer: size of the decision layer. Will typically be
                                 greater than 2 when ACGAN is involved
            - miniBatchNormalization: do we apply the mini-batch normalization
                                      at the last scale ?
            - generationActivation (function): activation function of the last
                                               layer (RGB layer). If None, then
                                               the identity is used. exp -> sigmoid, tanh
            - dimInput (int): 3 (RGB input), 1 (grey-scale input)
        """
        super(DNet, self).__init__()

        # Initialization paramneters
        self.initBiasToZero = initBiasToZero
        self.equalizedlR = equalizedlR
        self.dimInput = dimInput

        # Leaky relu activation
        self.leakyRelu = torch.nn.LeakyReLU(leakyReluLeak)

        self.generationActivation = generationActivation

        # Initalize the scales
        self.scalesDepth = [depthScale0]
        self.scaleLayers = nn.ModuleList()
        self.fromRGBLayers = nn.ModuleList()

        self.mergeLayers = nn.ModuleList()

        # Initialize the last layer
        self.initDecisionLayer(sizeDecisionLayer)

        # Layer 0
        self.groupScaleZero = nn.ModuleList()
        self.fromRGBLayers.append(EqualizedConv2d(dimInput, depthScale0, 1,
                                                  equalized=equalizedlR,
                                                  initBiasToZero=initBiasToZero))

        # Minibatch standard deviation
        dimEntryScale0 = depthScale0
        if miniBatchNormalization:
            dimEntryScale0 += 1

        self.miniBatchNormalization = miniBatchNormalization
        self.groupScaleZero.append(EqualizedConv2d(dimEntryScale0, depthScale0,
                                                   3, padding=1,
                                                   equalized=equalizedlR,
                                                   initBiasToZero=initBiasToZero))

        self.groupScaleZero.append(EqualizedLinear(depthScale0 * 16,
                                                   depthScale0,
                                                   equalized=equalizedlR,
                                                   initBiasToZero=initBiasToZero))

        # Initalize the upscaling parameters
        self.alpha = 0

       

    def addScale(self, depthNewScale, layerType='single'):

        depthLastScale = self.scalesDepth[-1]
        self.scalesDepth.append(depthNewScale)
        if layerType=='single':
            net = nn.Sequential(
                EqualizedConv2d(depthNewScale, depthLastScale, 3, padding=1,
                                equalized=self.equalizedlR, initBiasToZero=self.initBiasToZero),
                self.leakyRelu,

            )
        else:
            net = nn.Sequential(
                EqualizedConv2d(depthNewScale, depthNewScale, 3, padding=1,
                                equalized=self.equalizedlR, initBiasToZero=self.initBiasToZero),
                self.leakyRelu,
                EqualizedConv2d(depthNewScale, depthLastScale, 3, padding=1,
                                equalized=self.equalizedlR, initBiasToZero=self.initBiasToZero),
                self.leakyRelu,
            )

        self.scaleLayers.append(net)

        self.fromRGBLayers.append(EqualizedConv2d(self.dimInput,
                                                  depthNewScale,
                                                  1,
                                                  equalized=self.equalizedlR,
                                                  initBiasToZero=self.initBiasToZero))

    def setNewAlpha(self, alpha):
        r"""
        Update the value of the merging factor alpha
        Args:
            - alpha (float): merging factor, must be in [0, 1]
        """

        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be in [0,1]")

        if not self.fromRGBLayers:
            raise AttributeError("Can't set an alpha layer if only the scale 0"
                                 "is defined")

        self.alpha = alpha

    def initDecisionLayer(self, sizeDecisionLayer):
        if self.generationActivation is not None:
            self.decisionLayer = nn.Sequential(
                EqualizedLinear(self.scalesDepth[0], sizeDecisionLayer, equalized=self.equalizedlR,
                                initBiasToZero=self.initBiasToZero),
                self.generationActivation,
            )
        else:
            self.decisionLayer = EqualizedLinear(self.scalesDepth[0], sizeDecisionLayer, 
                                                equalized=self.equalizedlR, initBiasToZero=self.initBiasToZero)


    def forward(self, x, getFeature = False):

        # For Alpha blending
        if self.alpha > 0 and len(self.fromRGBLayers) > 1:
            y = nn.AdaptiveAvgPool2d(x.size(-1)//2)(x)
            y = self.leakyRelu(self.fromRGBLayers[- 2](y))
        # From RGB layer
        x = self.leakyRelu(self.fromRGBLayers[-1](x))

        # Caution: we must explore the layers group in reverse order !
        # Explore all scales before 0
        mergeLayer = self.alpha > 0 and len(self.scaleLayers) > 1
        for groupLayer in reversed(self.scaleLayers):
            x = groupLayer(x)
            x = nn.AdaptiveAvgPool2d(x.size(-1)//2)(x)
            if mergeLayer:
                mergeLayer = False
                x = self.alpha * y + (1-self.alpha) * x
        # Now the scale 0
        # Minibatch standard deviation
        if self.miniBatchNormalization:
            x = miniBatchStdDev(x)

        x = self.leakyRelu(self.groupScaleZero[0](x))

        x = x.view(-1, num_flat_features(x))
        x = self.leakyRelu(self.groupScaleZero[1](x))

        out = self.decisionLayer(x)
        if not getFeature:
            return out
        return out, x