export class StreamResampler16k {
  constructor(inRate: number);
  inRate: number;
  outRate: number;
  passthrough: boolean;
  process(chunk: Float32Array): Float32Array;
}
