export async function getEmbedding(text: string): Promise<number[]> {
    const endpointUrl = process.env.HF_INFERENCE_ENDPOINT_URL;
    const accessToken = process.env.HF_ACCESS_TOKEN;

    if (!endpointUrl || !accessToken) {
        throw new Error('Hugging Face Inference Endpoint URL or Access Token is not configured.');
    }

    console.log(`Querying Hugging Face Inference Endpoint for: "${text}"`);

    let response: Response;
    try {
        response = await fetch(endpointUrl, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${accessToken}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ inputs: text }),
        });
    } catch (err) {
        console.error('Network error calling Hugging Face Inference API:', err);
        throw new Error('Network error when calling Hugging Face Inference API.');
    }

    const resText = await response.text();

    if (!response.ok) {
        console.error('Hugging Face Inference API error response body:', resText);
        throw new Error(`Failed to get embedding from Hugging Face Inference API. Status: ${response.status}. Body: ${resText}`);
    }

    let data: any;
    try {
        data = JSON.parse(resText);
    } catch (err) {
        console.error('Failed to parse Hugging Face response as JSON:', resText);
        throw new Error('Invalid JSON response from Hugging Face Inference API.');
    }

    // If HF returned an error object
    if (data && typeof data === 'object' && 'error' in data) {
        console.error('Hugging Face returned error object:', data);
        throw new Error(`Hugging Face Inference API error: ${data.error}`);
    }

    // Common response shapes:
    // 1) Flat array: [0.1, 0.2, ...]
    // 2) Nested array: [[0.1, 0.2, ...]]
    // 3) Object with fields: { embedding: [...] } or { embeddings: [...] } or { outputs: [...] }

    if (Array.isArray(data)) {
        if (data.length === 0) {
            console.error('Empty array received from Hugging Face API');
            throw new Error('Empty embedding received from Hugging Face Inference API.');
        }
        // nested
        if (Array.isArray(data[0]) && typeof data[0][0] === 'number') {
            console.log('Embedding retrieved successfully from endpoint (nested array).');
            return data[0];
        }
        // flat
        if (typeof data[0] === 'number') {
            console.log('Embedding retrieved successfully from endpoint (flat array).');
            return data;
        }
    }

    if (data && typeof data === 'object') {
        if (Array.isArray(data.embedding) && typeof data.embedding[0] === 'number') {
            console.log('Embedding retrieved from object.embedding');
            return data.embedding;
        }
        if (Array.isArray(data.embeddings) && typeof data.embeddings[0] === 'number') {
            console.log('Embedding retrieved from object.embeddings');
            return data.embeddings;
        }
        if (Array.isArray(data.outputs) && data.outputs.length > 0) {
            const out = data.outputs[0];
            if (Array.isArray(out) && typeof out[0] === 'number') {
                console.log('Embedding retrieved from outputs[0]');
                return out;
            }
            if (out && Array.isArray(out.embedding)) {
                console.log('Embedding retrieved from outputs[0].embedding');
                return out.embedding;
            }
        }
    }

    console.error('Invalid embedding format received from Hugging Face Inference API:', data);
    throw new Error('Invalid embedding format received from Hugging Face Inference API.');
}
