import type { Tensor, Region } from './types';

export const regionLandmarks = ['eyeRight', 'eyeLeft', 'nose', 'mouthRight', 'mouthLeft'];

const exp = (v: number): number => {
  if (Math.abs(v) < 1.0) return v * Math.E;
  if (v > 0.0) return Math.exp(v);
  return -Math.exp(-v);
};

export async function boxes(logits: Array<Tensor>, minScore: number): Promise<Region[]> {
  const boxRaw = await logits[0].data();
  const scoreRaw = await logits[1].data();
  const landmarkRaw = await logits[2].data();
  const strideX = logits[1].shape[2] as number;
  const strideY = logits[1].shape[1] as number;
  const regions: Region[] = [];
  for (let y = 0; y < strideY; y++) {
    for (let x = 0; x < strideX; x++) {
      const idx = y * strideX + x;
      const score = scoreRaw[idx];
      if (score < minScore) continue;
      const x0 = (x - boxRaw[4 * idx + 0]) / strideX;
      const y0 = (y - boxRaw[4 * idx + 1]) / strideY;
      const x1 = (x + boxRaw[4 * idx + 2]) / strideX;
      const y1 = (y + boxRaw[4 * idx + 3]) / strideY;
      const landmarks: Record<string, [number, number]> = {};
      for (let i = 0; i < regionLandmarks.length; i++) {
        const lmidx = 2 * regionLandmarks.length * idx + i;
        let lx = landmarkRaw[lmidx] * 4;
        let ly = landmarkRaw[lmidx + regionLandmarks.length] * 4;
        lx = (exp(lx) + x) / strideX;
        ly = (exp(ly) + y) / strideY;
        landmarks[regionLandmarks[i]] = [lx, ly];
      }
      regions.push({ box: [x0, y0, x1 - x0, y1 - y0] as [number, number, number, number], score, landmarks }); // box: [x, y, width, height]
    }
  }
  return regions;
}

export function iou(region0: Region, region1: Region): number {
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
  if (area0 <= 0 || area1 <= 0) return 0.0;
  const intersectArea = Math.max(Math.min(ymax0, ymax1) - Math.max(ymin0, ymin1), 0.0) * Math.max(Math.min(xmax0, xmax1) - Math.max(xmin0, xmin1), 0.0);
  return intersectArea / (area0 + area1 - intersectArea);
}

export async function nms(regions: Region[], iouThreshold: number, maxResults: number): Promise<Region[]> {
  regions.sort((r0, r1) => r1.score - r0.score);
  const nmsRegions: Region[] = [];
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
      if (nmsRegions.length >= maxResults) break;
    }
  }
  return nmsRegions;
}
