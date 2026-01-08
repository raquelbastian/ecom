# E-commerce Recommendation System

This project is a full-stack e-commerce application featuring a sophisticated hybrid recommendation system. It's built with a Next.js frontend and a Python (FastAPI) backend that serves machine learning models. The primary goal is to deliver relevant and diverse product recommendations to users, enhancing their shopping experience.

## Live Demo

A live version of this application is deployed and available here:
**[https://ecom-seven-xi-92.vercel.app/](https://ecom-seven-xi-92.vercel.app/)**

## Problem Statement

In a crowded e-commerce marketplace, helping users discover products they are likely to be interested in is crucial for engagement and sales. The challenge is to move beyond simple popularity-based suggestions and provide personalized recommendations that consider various factors like product content, user behavior, and item similarity. This project aims to build and evaluate a hybrid recommendation engine that combines multiple strategies to generate high-quality recommendations.

## Dataset

The recommendation models are trained on the **Amazon Product Review Dataset**. This dataset contains product metadata and user reviews for a wide range of items, which is ideal for building content-based, collaborative filtering, and hybrid models.

Key features include:
- `product_id`: Unique identifier for each product.
- `product_name`: The title of the product.
- `category`: The product's category.
- `rating`: User-provided rating for the product.
- `about_product`: Detailed description.
- `user_id`: Identifier for the user who wrote the review.

## Installation and Setup

This project consists of a Next.js frontend and a Python backend. Follow these steps to set it up locally.

**1. Environment Setup**
   - Create a `.env.local` file in the root directory and add your MongoDB connection string:
     ```
     MONGODB_URI="your_mongodb_connection_string"
     ```
   - **Important:** Ensure your MongoDB database is populated with the Amazon product dataset.

**2. Install Dependencies**
   - **Frontend (Node.js):**
     ```bash
     npm install
     ```
   - **Backend (Python):**
     It's recommended to use a virtual environment.
     ```bash
     # Create and activate a virtual environment
     python3 -m venv python-services/source
     source python-services/source/bin/activate

     # Install Python packages
     pip install -r requirements.txt
     ```

## How to Run the Code

**1. Build Machine Learning Artifacts**
   - This script connects to your database, processes the data, and builds the necessary model files.
   - From within your activated Python virtual environment:
     ```bash
     python python-services/build_artifacts.py
     ```

**2. Run the Application**
   - You need to run two services in separate terminals.

   - **Terminal 1: Start the Python ML Service**
     (Make sure your virtual environment is active)
     ```bash
     python3 -m uvicorn python-services.main:app --host 0.0.0.0 --port 8001
     ```

   - **Terminal 2: Start the Next.js Frontend**
     ```bash
     npm run dev
     ```

**3. View the Application**
   - Visit [http://localhost:3000](http://localhost:3000) to view the product listing.

## Results Summary

The final recommendation system is a weighted hybrid model that combines the outputs of several individual recommendation strategies. The optimal weights for combining these models were determined using a **Random Search optimization process**. The strategies include:
- **Content-Based (PCA) (Content-Based):** Utilizes PCA on content features for recommendations.
- **Review Text-Based (Behavioral / Semantic):** Analyzes the semantic content of user reviews.
- **Feature-Based (PCA) (Content-Based):** Employs PCA on product features to find similar items.
- **Optimized Collaborative Filtering (SVD) Behavioral / Collaborative:** Uses Singular Value Decomposition on user-item interactions.

The hybrid approach demonstrated superior performance compared to any single model, providing more relevant and diverse recommendations. The system is designed for real-time inference, with model artifacts pre-computed for low-latency responses.

## Author Information

- **Raquel Bastian**
  - [GitHub](https://github.com/raquelbastian)
  - [LinkedIn](https://www.linkedin.com/in/raquel-bastian/)
