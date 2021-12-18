import * as tf from '@tensorflow/tfjs-node-gpu';
import * as log from '@vladmandic/pilogger';
import * as image from './image';
import * as decode from './decode';
import type { Tensor, Region, Image } from './types';

const inImage = './media/in/models.jpg';
const outImage = './media/out/models.jpg';

const modelOptions = {
  modelPath: 'file://./model/dbface.json',
  minScore: 0.20,
  iouThreshold: 0.10,
  maxResults: 1000,
  inputSize: [640, 480],
};

async function main() {
  log.headerJson();
  // process.env.CUDA_VISIBLE_DEVICES = '-1';
  // process.env.TF_CPP_MIN_LOG_LEVEL = '2';
  tf.setBackend('tensorflow');
  await tf.ready();
  // @ts-ignore backendInstance is private
  log.data({ tensorflow: tf.version['tfjs-node'], backend: tf.getBackend(), gpuEnabled: tf.engine().backendInstance.isGPUPackage, gpuActive: tf.engine().backendInstance.isUsingGpuDevice });
  const model: tf.GraphModel = await tf.loadGraphModel(modelOptions.modelPath); // load model
  log.data({ model: { ...modelOptions, bytes: tf.engine().memory().numBytes, tensors: tf.engine().memory().numTensors } });
  log.data({ input: inImage, output: outImage });
  // modelOptions.inputSize = [Object.values(model.modelSignature.inputs)[0].tensorShape.dim[2].size, Object.values(model.modelSignature.inputs)[0].tensorShape.dim[1].size]; // use to autodetect inputSize
  const img: Image = await image.load(inImage, modelOptions.inputSize as [number, number]); // load image
  const logits: Tensor[] = await model.predict(img.tensor) as Tensor[]; // run model
  const regions: Region[] = await decode.boxes(logits, modelOptions.minScore); // decode results
  const nms: Region[] = await decode.nms(regions, modelOptions.iouThreshold, modelOptions.maxResults); // run nms on results
  logits.forEach((tensor) => tf.dispose(tensor)); // dispose model results
  log.data({ results: nms.length, scores: nms.map((region) => Math.round(1000 * region.score) / 10) });
  await image.save(img, outImage, nms);
}

main();
