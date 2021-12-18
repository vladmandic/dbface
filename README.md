# DBFace: Face Detection for TFJS and NodeJS

**DBFace-MobileNetV3** model converted from **PyTorch** to **TFJS Graph Model**  
With native decode bounds and NMS functions in `TypeScript` to process results

## Run

> npm start

```js
{ application: 'dbface', version: '0.0.1' }
{ user: 'vlado', platform: 'linux', arch: 'x64', node: 'v17.2.0' }
{ tensorflow: '3.12.0', backend: 'tensorflow', gpuEnabled: true, gpuActive: true }
{ model: { modelPath: 'file://./model/dbface.json', minScore: 0.2, iouThreshold: 0.1, maxResults: 1000, inputSize: [ 640, 480 ], bytes: 7124128, tensors: 299 } }
{ input: './media/in/models.jpg', bytes: 24602400, resolution: [ 3600, 2278 ], tensor: [ 1, 480, 640, 3 ], type: 'float32' }
2021-12-18 09:49:29 DATA:  {
  results: 25,
  scores: [
    68.6, 68.5, 68.2, 67.8, 66.8,
    66.7, 66.6, 66.5, 66.3, 65.9,
    64.6, 63.8, 63.2, 62.6, 62.4,
    61.4, 61.0, 60.4, 60.0, 59.9,
    59.6, 59.1, 56.7, 56.1, 55.4
  ]
}
2021-12-18 09:49:29 STATE: { output: './media/out/models.jpg', resolution: [ 3600, 2278 ] }
```

## Example

![Example Image](media/out/models.jpg)

## Model Signature

```js
INFO:  graph model: dbface/model/dbface.json
INFO:  created on: 2021-12-17T14:46:10.793Z
INFO:  metadata: { generatedBy: 'https://github.com/dlunion/DBFace', convertedBy: 'https://github.com/vladmandic' }
INFO:  model inputs based on signature
{ name: 'input:0', dtype: 'DT_FLOAT', shape: [ 1, 480, 640, 3 ] }
INFO:  model outputs based on signature
{ id: 0, name: 'Identity_1:0', dytpe: 'DT_FLOAT', shape: [ 1, 120, 160, 4 ] }
{ id: 1, name: 'Identity_2:0', dytpe: 'DT_FLOAT', shape: [ 1, 120, 160, 1 ] }
{ id: 2, name: 'Identity:0', dytpe: 'DT_FLOAT', shape: [ 1, 120, 160, 10 ] }
INFO:  tensors: 299
DATA:  weights: {
  files: [ 'dbface.bin' ],
  size: { disk: 7124128, memory: 7124128 },
  count: { total: 299, float32: 242, int32: 57 },
  quantized: { none: 299 },
  values: { total: 1781032, float32: 1780785, int32: 247 }
}
DATA:  kernel ops: {
  graph: [ 'Const', 'Placeholder', 'Identity' ],
  transformation: [ 'Pad', 'ExpandDims' ],
  convolution: [ '_FusedConv2D', 'FusedDepthwiseConv2dNative' ],
  arithmetic: [ 'AddV2', 'Mul', 'Add' ],
  basic_math: [ 'Relu6', 'Sigmoid', 'Exp' ],
  reduction: [ 'Mean' ],
  image: [ 'ResizeNearestNeighbor' ],
  slice_join: [ 'StridedSlice', 'ConcatV2' ]
}
```

## Credit

- <https://github.com/dlunion/DBFace>
