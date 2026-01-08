# Data Dictionary

This document provides a description of the variables in the Amazon product dataset.

| Variable Name | Type | Description | Allowed Values | Notes |
|---------------|------|-------------|----------------|-------|
| `product_id` | String | Unique identifier for each product. | - | Primary key |
| `product_name` | String | The title or name of the product. | - | |
| `category` | String | The category or sub-category of the product. | - | |
| `actual_price` | Float | The original price of the product. | Numeric | |
| `discounted_price`| Float | The price of the product after discount. | Numeric | |
| `rating` | Float | The average user rating for the product. | 1.0 - 5.0 | |
| `rating_count` | Integer | The total number of ratings the product has received. | Numeric | |
| `about_product` | String | A detailed description of the product. | - | |
| `user_id` | String | The unique identifier for a user who reviewed the product. | - | |
| `review_title` | String | The title of the user's review. | - | |
| `review_content` | String | The full text content of the user's review. | - | |
| `img_link` | String | A URL to an image of the product. | URL | |
