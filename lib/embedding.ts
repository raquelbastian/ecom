// Embeddings via Voyage AI (https://docs.voyageai.com)
// Replaces the previous Hugging Face Inference Endpoint implementation —
// serverless API, no cold starts, consistent response shape.

const VOYAGE_API_URL = 'https://api.voyageai.com/v1/embeddings';
const VOYAGE_MODEL = 'voyage-3.5-lite'; // 1024 dimensions

// Voyage distinguishes how the text will be used, which improves retrieval:
// 'query'    → search queries typed by users
// 'document' → product texts stored in the database
export type EmbeddingInputType = 'query' | 'document';

export async function getEmbeddings(
    texts: string[],
    inputType: EmbeddingInputType
): Promise<number[][]> {
    const apiKey = process.env.VOYAGE_API_KEY;
    if (!apiKey) {
        throw new Error('VOYAGE_API_KEY is not configured.');
    }

    let response: Response;
    try {
        response = await fetch(VOYAGE_API_URL, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${apiKey}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                input: texts,
                model: VOYAGE_MODEL,
                input_type: inputType,
            }),
        });
    } catch (err) {
        console.error('Network error calling Voyage AI API:', err);
        throw new Error('Network error when calling Voyage AI API.');
    }

    const resText = await response.text();

    if (!response.ok) {
        console.error('Voyage AI API error response body:', resText);
        throw new Error(`Voyage AI API error. Status: ${response.status}. Body: ${resText}`);
    }

    const data = JSON.parse(resText);
    if (!Array.isArray(data?.data) || data.data.length !== texts.length) {
        console.error('Unexpected Voyage AI response shape:', data);
        throw new Error('Unexpected response shape from Voyage AI API.');
    }

    // data.data entries carry an `index` matching the input order
    return data.data
        .sort((a: { index: number }, b: { index: number }) => a.index - b.index)
        .map((d: { embedding: number[] }) => d.embedding);
}

export async function getEmbedding(
    text: string,
    inputType: EmbeddingInputType = 'query'
): Promise<number[]> {
    const [embedding] = await getEmbeddings([text], inputType);
    if (!Array.isArray(embedding) || embedding.length === 0) {
        throw new Error('Empty embedding received from Voyage AI API.');
    }
    return embedding;
}
