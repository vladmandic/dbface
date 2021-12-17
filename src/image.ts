import * as fs from 'fs';
import * as canvas from 'canvas';
import * as tf from '@tensorflow/tfjs-node-gpu';
// @ts-ignore // pilogger does not have typedefs
import * as log from '@vladmandic/pilogger';
import type { Image, Region } from './types';

export async function save(img: Image, target: string, regions: Region[]) {
  const c = new canvas.Canvas(img.inputShape[0], img.inputShape[1]);
  const ctx = c.getContext('2d');
  const original = await canvas.loadImage(img.fileName);
  ctx.drawImage(original, 0, 0, c.width, c.height);
  const fontSize = Math.round(Math.sqrt(c.width) / 2);
  ctx.lineWidth = 2;
  ctx.strokeStyle = 'white';
  ctx.fillStyle = 'white';
  ctx.font = `${fontSize}px "Segoe UI"`;
  for (let i = 0; i < regions.length; i++) {
    ctx.fillText(`#${i}: ${Math.round(100 * regions[i].score)}%`, regions[i].box[0] * c.width + 5, regions[i].box[1] * c.height - 3);
    ctx.rect(regions[i].box[0] * c.width, regions[i].box[1] * c.height, regions[i].box[2] * c.width, regions[i].box[3] * c.height);
    ctx.stroke();
    for (const [_key, val] of Object.entries(regions[i].landmarks)) {
      ctx.beginPath();
      ctx.arc(val[0] * c.width, val[1] * c.height, 3, 0, 2 * Math.PI);
      ctx.stroke();
    }
  }
  ctx.stroke();
  const out = fs.createWriteStream(target);
  out.on('finish', () => log.state({ output: target, resolution: [c.width, c.height] }));
  out.on('error', (err) => log.error({ output: target, error: err }));
  const stream = c.createJPEGStream({ quality: 0.75, progressive: true, chromaSubsampling: true });
  stream.pipe(out);
}

export async function load(fileName: string, inputSize: [number, number]): Promise<Image> {
  const data = fs.readFileSync(fileName);
  const buffer = tf.node.decodeImage(data);
  const resize = tf.image.resizeBilinear(buffer, [inputSize[1], inputSize[0]]);
  const norm = tf.div(resize, 255);
  const tensor = tf.expandDims(norm, 0);
  const img = { fileName, tensor, inputShape: [buffer.shape[1], buffer.shape[0]] as [number, number], outputShape: tensor.shape, size: buffer.size, dtype: tensor.dtype };
  tf.dispose([buffer, resize, norm]);
  log.state({ input: img.fileName, size: img.size, resolution: img.inputShape, tensor: img.outputShape, type: img.dtype });
  return img;
}
