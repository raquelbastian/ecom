import { pipeline, env, PipelineType } from '@xenova/transformers';

// Allow local models
env.allowLocalModels = true;
// Specify a cache directory
env.cacheDir = '/tmp/transformers_cache';

class EmbeddingSingleton {
  static task: PipelineType = 'feature-extraction';
  static model = 'Xenova/all-MiniLM-L6-v2';
  static instance: any = null;

  static async getInstance(progress_callback?: any) {
    if (this.instance === null) {
      this.instance = pipeline(this.task, this.model, { progress_callback });
    }
    return this.instance;
  }
}

export async function getEmbedding(text: string) {
  const extractor = await EmbeddingSingleton.getInstance();
  const result = await extractor(text, { pooling: 'mean', normalize: true });
  return Array.from(result.data);
}
