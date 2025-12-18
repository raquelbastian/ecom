import axios from 'axios';

export async function getMLPrediction(features: number[]) {
  try {
    const response = await axios.post('http://localhost:8000/predict', {
      features,
    });
    return response.data;
  } catch (error) {
    console.error('Error calling ML API:', error);
    return null;
  }
}
