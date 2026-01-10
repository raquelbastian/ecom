export async function getEmbedding(text: string): Promise<number[]> {
    const endpointUrl = process.env.HF_INFERENCE_ENDPOINT_URL;
    const accessToken = process.env.HF_ACCESS_TOKEN;

    if (!endpointUrl || !accessToken) {
        throw new Error('Hugging Face Inference Endpoint URL or Access Token is not configured.');
    }

    console.log(`Querying Hugging Face Inference Endpoint for: "${text}"`);

    const response = await fetch(endpointUrl, {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${accessToken}`,
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            inputs: text,
        }),
    });

    if (!response.ok) {
        const errorBody = await response.text();
        console.error('Hugging Face Inference API error:', errorBody);
        throw new Error(`Failed to get embedding from Hugging Face Inference API. Status: ${response.status}`);
    }

    const data = await response.json();
    
    // The response for sentence-transformers is typically an array containing one embedding array.
    if (data && Array.isArray(data) && Array.isArray(data[0])) {
        console.log('Embedding retrieved successfully from endpoint.');
        return data[0];
    }

    throw new Error('Invalid embedding format received from Hugging Face Inference API.');
}
