import { pipeline, env, PipelineType } from '@xenova/transformers';

// Allow local models
env.allowLocalModels = true;
// Specify a cache directory
env.cacheDir = '/tmp/transformers_cache';

// Define a singleton pipeline instance
let embedder: any;

export async function getEmbedding(text: string): Promise<number[]> {
  if (!embedder) {
    console.log('Initializing new embedder pipeline...');
    // Dynamically import the pipeline function
    const { pipeline } = await import('@xenova/transformers');
    embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
      quantized: true,
    });
    console.log('Embedder pipeline initialized.');
  }

  const result = await embedder(text, { pooling: 'mean', normalize: true });
  return Array.from(result.data);
}
