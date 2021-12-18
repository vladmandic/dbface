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
  await tf.ready();
  log.headerJson();
  log.info({ tensorflow: tf.version['tfjs-node'] });
  const model: tf.GraphModel = await tf.loadGraphModel(modelOptions.modelPath);
  // modelOptions.inputSize = [Object.values(model.modelSignature.inputs)[0].tensorShape.dim[2].size, Object.values(model.modelSignature.inputs)[0].tensorShape.dim[1].size]; // use to autodetect inputSize
  log.data({ input: inImage, output: outImage });
  log.data({ modelOptions });
  const img: Image = await image.load(inImage, modelOptions.inputSize as [number, number]);
  const logits: Tensor[] = await model.predict(img.tensor) as Tensor[];
  const regions: Region[] = await decode.boxes(logits, modelOptions.minScore);
  const nms: Region[] = await decode.nms(regions, modelOptions.iouThreshold, modelOptions.maxResults);
  log.data({ results: nms.length, scores: nms.map((region) => Math.round(1000 * region.score) / 10) });
  await image.save(img, outImage, nms);
}

main();
