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

export async function getRecommendations(productId: string, n: number = 5) {
  try {
    const response = await axios.get(`http://localhost:8000/recommend/${encodeURIComponent(productId)}?n=${n}`);
    return response.data; // expected shape: { recommendations: [...] }
  } catch (error) {
    console.error('Error calling recommendation API:', error);
    return null;
  }
}

export async function getRecommendationsPCA(productId: string, n: number = 5, n_components: number = 50) {
  try {
    const response = await axios.get(`http://localhost:8000/recommend_pca/${encodeURIComponent(productId)}?n=${n}&n_components=${n_components}`);
    return response.data; // expected shape: { recommendations: [...] }
  } catch (error) {
    console.error('Error calling PCA recommendation API:', error);
    return null;
  }
}

export async function getRecommendationsReviews(productId: string, n: number = 5) {
  try {
    const response = await axios.get(`http://localhost:8000/recommend_reviews/${encodeURIComponent(productId)}?n=${n}`);
    return response.data; // expected shape: { recommendations: [...] }
  } catch (error) {
    console.error('Error calling review-based recommendation API:', error);
    return null;
  }
}

export async function getRecommendationsContent(productId: string, n: number = 5) {
  try {
    const response = await axios.get(`http://localhost:8000/recommend_content/${encodeURIComponent(productId)}?n=${n}`);
    return response.data; // expected shape: { recommendations: [...] }
  } catch (error) {
    console.error('Error calling content-based recommendation API:', error);
    return null;
  }
}

export async function getRecommendationsSentiment(productId: string, n: number = 5) {
  try {
    const response = await axios.get(`http://localhost:8000/recommend_sentiment/${encodeURIComponent(productId)}?n=${n}`);
    return response.data; // expected shape: { recommendations: [...] }
  } catch (error) {
    console.error('Error calling sentiment-based recommendation API:', error);
    return null;
  }
}

export async function getRecommendationsContentPCA(productId: string, n: number = 5, n_components: number = 50) {
  try {
    const response = await axios.get(`http://localhost:8000/recommend_content_pca/${encodeURIComponent(productId)}?n=${n}&n_components=${n_components}`);
    return response.data; // expected shape: { recommendations: [...] }
  } catch (error) {
    console.error('Error calling PCA content-based recommendation API:', error);
    return null;
  }
}

export async function getRecommendationsTopic(productId: string, n: number = 5, n_topics: number = 10) {
  try {
    const response = await axios.get(`http://localhost:8000/recommend_topic/${encodeURIComponent(productId)}?n=${n}&n_topics=${n_topics}`);
    return response.data; // expected shape: { recommendations: [...] }
  } catch (error) {
    console.error('Error calling topic modeling recommendation API:', error);
    return null;
  }
}

export async function getRecommendationsReviewerOverlap(productId: string, n: number = 5) {
  try {
    const response = await axios.get(`http://localhost:8000/recommend_reviewer_overlap/${encodeURIComponent(productId)}?n=${n}`);
    return response.data; // expected shape: { recommendations: [...] }
  } catch (error) {
    console.error('Error calling reviewer-overlap recommendation API:', error);
    return null;
  }
}

export async function getRecommendationsHybrid(productId: string, n: number = 5) {
  try {
    const response = await axios.get(`http://localhost:8000/recommend_hybrid/${encodeURIComponent(productId)}?n=${n}`);
    return response.data; // expected shape: { recommendations: [...] }
  } catch (error) {
    console.error('Error calling hybrid recommendation API:', error);
    return null;
  }
}

export async function getRecommendationsWeightedHybrid(productId: string, n: number = 5, weights: Record<string, number>) {
  try {
    const response = await axios.post(`http://localhost:8000/recommend_weighted_hybrid/${encodeURIComponent(productId)}?n=${n}`, {
      weights
    });
    return response.data; // expected shape: { recommendations: [...] }
  } catch (error) {
    console.error('Error calling weighted hybrid recommendation API:', error);
    return null;
  }
}

export async function getTrendingProducts(n: number = 8) {
  try {
    const response = await axios.get(`http://localhost:8000/trending_products?n=${n}`);
    return response.data; // expected shape: { trending: [...] }
  } catch (error) {
    console.error('Error fetching trending products:', error);
    return null;
  }
}

export async function getTrendingProductsML(n: number = 8) {
  try {
    const response = await axios.get(`http://localhost:8000/trending_products_ml?n=${n}`);
    return response.data; // expected shape: { trending: [...] }
  } catch (error) {
    console.error('Error fetching ML-based trending products:', error);
    return null;
  }
}
