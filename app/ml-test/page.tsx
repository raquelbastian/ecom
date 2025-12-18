"use client";
import { getMLPrediction } from '@/lib/mlService';
import { useState } from 'react';

export default function MLTest() {
  const [result, setResult] = useState<string | null>(null);
  async function handlePredict() {
    const res = await getMLPrediction([1.2, 3.4, 5.6]);
    setResult(JSON.stringify(res));
  }
  return (
    <div className="flex flex-col items-center justify-center min-h-screen">
      <h1 className="text-2xl font-bold mb-4">Test ML Prediction</h1>
      <button
        onClick={handlePredict}
        className="px-6 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
      >
        Get Prediction
      </button>
      {result && (
        <div className="mt-4 p-4 bg-gray-100 rounded text-black">
          <strong>Result:</strong> {result}
        </div>
      )}
    </div>
  );
}