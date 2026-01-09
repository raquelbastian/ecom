import { pipeline, env } from '@xenova/transformers';

// Configure the environment
env.allowLocalModels = true;
env.cacheDir = '/tmp/transformers_cache';

// Use a class to implement the singleton pattern
class EmbeddingSingleton {
    private static instance: any;

    static async getInstance(progress_callback?: Function) {
        if (!this.instance) {
            console.log('Initializing new embedder pipeline with all-MiniLM-L12-v2...');
            this.instance = pipeline('feature-extraction', 'Xenova/all-MiniLM-L12-v2', {
                quantized: true,
                progress_callback,
            });
            console.log('Embedder pipeline promise created.');
        }
        return this.instance;
    }
}

export async function getEmbedding(text: string): Promise<number[]> {
    const embedder = await EmbeddingSingleton.getInstance();
    console.log('Embedder instance retrieved.');
    
    const result = await embedder(text, { pooling: 'mean', normalize: true });
    console.log('Embedding generated successfully.');
    
    return Array.from(result.data);
}
