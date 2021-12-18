import type { Tensor } from '@tensorflow/tfjs-node-gpu';

export type { Tensor } from '@tensorflow/tfjs-node-gpu';
export type Region = { box: [number, number, number, number], score: number, landmarks: Record<string, [number, number]> }
export type Image = { fileName: string, tensor: Tensor, inputShape: [number, number], outputShape: number[], bytes: number, dtype: string }
