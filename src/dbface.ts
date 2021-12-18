import * as fs from 'fs';
import * as tf from '@tensorflow/tfjs-node-gpu';
import * as log from '@vladmandic/pilogger';
import { Command } from 'commander';
import * as image from './image';
import * as decode from './decode';
import type { Tensor, Region, Image } from './types';

const options = {
  inImage: 'media/in/models.jpg',
  outImage: 'media/out/models.jpg',
  modelPath: 'file://models/mb3-f32-800-1280/dbface.json',
  minScore: 0.20,
  iouThreshold: 0.10,
  maxResults: 1000,
  inputSize: [0, 0], // autodetected
};

async function validate() {
  const ok = (f: string): boolean => fs.existsSync(f) && fs.statSync(f).isFile();
  const exit = (msg: string) => { log.error(msg); process.exit(1); };
  const cli = new Command();
  cli
    .option('--model <path>', 'path to model')
    .option('--input <path>', 'input image')
    .option('--output <path>', 'output image')
    .option('--score <number>', 'minimum score confidence')
    .parse();
  if (cli.opts().model && ok(cli.opts().model)) options.modelPath = 'file://' + cli.opts().model;
  if (cli.opts().input) options.inImage = cli.opts().input;
  if (cli.opts().output) options.outImage = cli.opts().output;
  if (cli.opts().score) options.minScore = parseFloat(cli.opts().score);
  if (!ok(options.inImage)) exit('invalid input image');
}

async function main() {
  log.headerJson();
  log.data({ options });
  tf.setBackend('tensorflow');
  await tf.ready();
  // @ts-ignore backendInstance is private
  log.data({ tensorflow: tf.version['tfjs-node'], backend: tf.getBackend(), gpuEnabled: tf.engine().backendInstance.isGPUPackage, gpuActive: tf.engine().backendInstance.isUsingGpuDevice });
  const model: tf.GraphModel = await tf.loadGraphModel(options.modelPath); // load model
  log.data({ memory: { bytes: tf.engine().memory().numBytes, tensors: tf.engine().memory().numTensors } });
  // @ts-ignore inputs is undefined
  options.inputSize = [Object.values(model.modelSignature.inputs)[0].tensorShape.dim[2].size, Object.values(model.modelSignature.inputs)[0].tensorShape.dim[1].size]; // use to autodetect inputSize
  const img: Image = await image.load(options.inImage, options.inputSize as [number, number]); // load image
  const t0 = process.hrtime.bigint();
  const logits: Tensor[] = await model.predict(img.tensor) as Tensor[]; // run model
  const t1 = process.hrtime.bigint();
  const regions: Region[] = await decode.boxes(logits, options.minScore); // decode results
  const nms: Region[] = await decode.nms(regions, options.iouThreshold, options.maxResults); // run nms on results
  const t2 = process.hrtime.bigint();
  logits.forEach((tensor) => tf.dispose(tensor)); // dispose model results
  log.data({ predictTime: Math.round(Number(t1 - t0) / 10000000), processTime: Math.round(Number(t2 - t1) / 10000000), results: regions.length, nms: nms.length });
  log.data({ scores: nms.map((region) => Math.round(100 * region.score) / 100) });
  await image.save(img, options.outImage, nms);
}

validate();
main();
