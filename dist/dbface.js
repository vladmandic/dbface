/*
  DBFace
  author: <https://github.com/vladmandic>'
*/

var __create = Object.create;
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __getProtoOf = Object.getPrototypeOf;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __markAsModule = (target) => __defProp(target, "__esModule", { value: true });
var __reExport = (target, module2, desc) => {
  if (module2 && typeof module2 === "object" || typeof module2 === "function") {
    for (let key of __getOwnPropNames(module2))
      if (!__hasOwnProp.call(target, key) && key !== "default")
        __defProp(target, key, { get: () => module2[key], enumerable: !(desc = __getOwnPropDesc(module2, key)) || desc.enumerable });
  }
  return target;
};
var __toModule = (module2) => {
  return __reExport(__markAsModule(__defProp(module2 != null ? __create(__getProtoOf(module2)) : {}, "default", module2 && module2.__esModule && "default" in module2 ? { get: () => module2.default, enumerable: true } : { value: module2, enumerable: true })), module2);
};

// src/dbface.ts
var tf2 = __toModule(require("@tensorflow/tfjs-node-gpu"));
var log2 = __toModule(require("@vladmandic/pilogger"));

// src/image.ts
var fs = __toModule(require("fs"));
var canvas = __toModule(require("canvas"));
var tf = __toModule(require("@tensorflow/tfjs-node-gpu"));
var log = __toModule(require("@vladmandic/pilogger"));
async function save(img, target, regions) {
  const c = new canvas.Canvas(img.inputShape[0], img.inputShape[1]);
  const ctx = c.getContext("2d");
  const original = await canvas.loadImage(img.fileName);
  ctx.drawImage(original, 0, 0, c.width, c.height);
  const fontSize = Math.round(Math.sqrt(c.width) / 2);
  ctx.lineWidth = 2;
  ctx.strokeStyle = "white";
  ctx.fillStyle = "white";
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
  out.on("finish", () => log.state({ output: target, resolution: [c.width, c.height] }));
  out.on("error", (err) => log.error({ output: target, error: err }));
  const stream = c.createJPEGStream({ quality: 0.75, progressive: true, chromaSubsampling: true });
  stream.pipe(out);
}
async function load(fileName, inputSize) {
  const data2 = fs.readFileSync(fileName);
  const decoded = tf.node.decodeImage(data2);
  const resize = tf.image.resizeBilinear(decoded, [inputSize[1], inputSize[0]]);
  const norm = tf.div(resize, 255);
  const tensor = tf.expandDims(norm, 0);
  const img = { fileName, tensor, inputShape: [decoded.shape[1], decoded.shape[0]], outputShape: tensor.shape, size: decoded.size, dtype: tensor.dtype };
  tf.dispose([decoded, resize, norm]);
  log.state({ input: img.fileName, size: img.size, resolution: img.inputShape, tensor: img.outputShape, type: img.dtype });
  return img;
}

// src/decode.ts
var regionLandmarks = ["eyeRight", "eyeLeft", "nose", "mouthRight", "mouthLeft"];
function exp(v) {
  if (Math.abs(v) < 1)
    return v * Math.E;
  if (v > 0)
    return Math.exp(v);
  return -Math.exp(-v);
}
async function boxes(logits, minScore) {
  const bboxPtr = await logits[0].data();
  const scoresPtr = await logits[1].data();
  const landmarkPtr = await logits[2].data();
  const scoreW = logits[1].shape[2];
  const scoreH = logits[1].shape[1];
  const regions = [];
  for (let y = 0; y < scoreH; y++) {
    for (let x = 0; x < scoreW; x++) {
      const idx = y * scoreW + x;
      const score = scoresPtr[idx];
      if (score < minScore)
        continue;
      const bx = bboxPtr[4 * idx + 0];
      const by = bboxPtr[4 * idx + 1];
      const bw = bboxPtr[4 * idx + 2];
      const bh = bboxPtr[4 * idx + 3];
      const x0 = (x - bx) / scoreW;
      const y0 = (y - by) / scoreH;
      const x1 = (x + bw) / scoreW;
      const y1 = (y + bh) / scoreH;
      const landmarks = {};
      for (let i = 0; i < regionLandmarks.length; i++) {
        const lmidx = 2 * regionLandmarks.length * idx + i;
        let lx = landmarkPtr[lmidx] * 4;
        let ly = landmarkPtr[lmidx + regionLandmarks.length] * 4;
        lx = (exp(lx) + x) / scoreW;
        ly = (exp(ly) + y) / scoreH;
        landmarks[regionLandmarks[i]] = [lx, ly];
      }
      const region = {
        box: [x0, y0, x1 - x0, y1 - y0],
        score,
        landmarks
      };
      regions.push(region);
    }
  }
  return regions;
}
function iou(region0, region1) {
  const sx0 = region0.box[0];
  const sy0 = region0.box[1];
  const ex0 = region0.box[2] + region0.box[0];
  const ey0 = region0.box[3] + region0.box[1];
  const sx1 = region1.box[0];
  const sy1 = region1.box[1];
  const ex1 = region1.box[2] + region1.box[0];
  const ey1 = region1.box[3] + region1.box[1];
  const xmin0 = Math.min(sx0, ex0);
  const ymin0 = Math.min(sy0, ey0);
  const xmax0 = Math.max(sx0, ex0);
  const ymax0 = Math.max(sy0, ey0);
  const xmin1 = Math.min(sx1, ex1);
  const ymin1 = Math.min(sy1, ey1);
  const xmax1 = Math.max(sx1, ex1);
  const ymax1 = Math.max(sy1, ey1);
  const area0 = (ymax0 - ymin0) * (xmax0 - xmin0);
  const area1 = (ymax1 - ymin1) * (xmax1 - xmin1);
  if (area0 <= 0 || area1 <= 0)
    return 0;
  const intersectArea = Math.max(Math.min(ymax0, ymax1) - Math.max(ymin0, ymin1), 0) * Math.max(Math.min(xmax0, xmax1) - Math.max(xmin0, xmin1), 0);
  return intersectArea / (area0 + area1 - intersectArea);
}
async function nms(regions, iouThreshold, maxResults) {
  regions.sort((r0, r1) => r1.score - r0.score);
  const nmsRegions = [];
  for (let i = 0; i < regions.length; i++) {
    let ignore = false;
    for (let j = 0; j < nmsRegions.length; j++) {
      if (iou(regions[i], nmsRegions[j]) >= iouThreshold) {
        ignore = true;
        break;
      }
    }
    if (!ignore) {
      nmsRegions.push(regions[i]);
      if (nmsRegions.length >= maxResults)
        break;
    }
  }
  return nmsRegions;
}

// src/dbface.ts
var inImage = "./media/in/models.jpg";
var outImage = "./media/out/models.jpg";
var modelOptions = {
  modelPath: "file://./model/dbface.json",
  minScore: 0.2,
  iouThreshold: 0.1,
  maxResults: 1e3,
  inputSize: [640, 480]
};
async function main() {
  await tf2.ready();
  log2.headerJson();
  log2.info({ tensorflow: tf2.version["tfjs-node"] });
  const model = await tf2.loadGraphModel(modelOptions.modelPath);
  log2.data({ input: inImage, output: outImage });
  log2.data({ modelOptions });
  const img = await load(inImage, modelOptions.inputSize);
  const logits = await model.predict(img.tensor);
  const regions = await boxes(logits, modelOptions.minScore);
  const nms2 = await nms(regions, modelOptions.iouThreshold, modelOptions.maxResults);
  log2.data({ results: nms2.length, scores: nms2.map((region) => Math.round(1e3 * region.score) / 10) });
  await save(img, outImage, nms2);
}
main();
//# sourceMappingURL=dbface.js.map
