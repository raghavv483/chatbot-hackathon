# Walmart RAG API

A FastAPI-based REST API for Walmart product search and recommendations using Retrieval-Augmented Generation (RAG).

## 🚀 Quick Start

### Deploy on Render

1. **Fork/Clone this repository**
2. **Connect to Render:**
   - Go to [render.com](https://render.com)
   - Create a new Web Service
   - Connect your GitHub repository
   - Render will automatically detect the `render.yaml` configuration

3. **Environment Variables (Optional):**
   - `GOOGLE_API_KEY`: Your Google Generative AI API key for enhanced responses
   - `PORT`: Port number (default: 8000)

4. **Deploy:**
   - Render will automatically build and deploy your API
   - The API will be available at: `https://your-app-name.onrender.com`

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API
python app.py

# Or with uvicorn
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## 📚 API Endpoints

### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "components": {
    "initialized": true,
    "data_loaded": true,
    "embeddings_generated": true,
    "collection_ready": true,
    "llm_available": false,
    "rag_chain_ready": false
  },
  "message": "RAG system is fully operational"
}
```

### 2. Query Products
```http
POST /query
Content-Type: application/json

{
  "query": "What are the best clothing deals in Jaipur?",
  "filters": {
    "store": "Jaipur",
    "category": "Clothing"
  },
  "n_results": 5
}
```

**Response:**
```json
{
  "response": "Based on the search results, here are some great clothing deals in Jaipur...",
  "products": [
    {
      "name": "Adidas T-Shirt",
      "brand": "Adidas",
      "category": "Clothing",
      "price": 1200,
      "discount": 15,
      "store": "Jaipur",
      "description": "Product: Adidas T-Shirt, Brand: Adidas, Category: Clothing..."
    }
  ],
  "metadata": {
    "found": true,
    "count": 3,
    "query": "What are the best clothing deals in Jaipur?",
    "filters": {
      "store": "Jaipur",
      "category": "Clothing"
    }
  }
}
```

### 3. Get Products (Filtered)
```http
GET /products?brand=Nike&category=Clothing&store=Jaipur&limit=10
```

**Response:**
```json
{
  "products": [
    {
      "name": "Nike Running Shoes",
      "brand": "Nike",
      "category": "Clothing",
      "price": 3500,
      "discount": 20,
      "store": "Jaipur",
      "description": "Product: Nike Running Shoes, Brand: Nike..."
    }
  ],
  "count": 1,
  "filters": {
    "brand": "Nike",
    "category": "Clothing",
    "store": "Jaipur"
  }
}
```

## 🔧 Integration Examples

### JavaScript/Node.js
```javascript
// Query products
const response = await fetch('https://your-api.onrender.com/query', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    query: "Show me Nike products under 3000 rupees",
    filters: {
      brand: "Nike",
      price: { $lt: 3000 }
    }
  })
});

const result = await response.json();
console.log(result.response);
console.log(result.products);
```

### Python
```python
import requests

# Query products
response = requests.post('https://your-api.onrender.com/query', json={
    'query': 'What are the best clothing deals in Jaipur?',
    'filters': {
        'store': 'Jaipur',
        'category': 'Clothing'
    }
})

result = response.json()
print(result['response'])
for product in result['products']:
    print(f"- {product['name']}: ₹{product['price']}")
```

### AR/VR Integration
```javascript
// For AR/VR applications
class WalmartRAGClient {
  constructor(apiUrl) {
    this.apiUrl = apiUrl;
  }

  async searchProducts(query, filters = {}) {
    const response = await fetch(`${this.apiUrl}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, filters })
    });
    return await response.json();
  }

  async getProductsByLocation(store) {
    const response = await fetch(`${this.apiUrl}/products?store=${store}`);
    return await response.json();
  }
}

// Usage in AR/VR app
const ragClient = new WalmartRAGClient('https://your-api.onrender.com');

// Search for products
const results = await ragClient.searchProducts(
  "Show me Nike shoes",
  { store: "Jaipur" }
);

// Display products in AR/VR
results.products.forEach(product => {
  // Create 3D product visualization
  createProductCard(product);
});
```

## 🛠️ Configuration

### Environment Variables
- `GOOGLE_API_KEY`: Google Generative AI API key (optional)
- `PORT`: Server port (default: 8000)

### Data Source
The API automatically looks for `walmart_products.csv` in the following locations:
1. Current directory
2. `./walmart_products.csv`
3. `../walmart_products.csv`
4. `/content/walmart_products.csv`

If no file is found, it uses sample data.

## 📊 Performance

- **Cold Start**: ~30-60 seconds (model loading)
- **Query Response**: ~1-3 seconds
- **Concurrent Requests**: Supports multiple simultaneous queries
- **Memory Usage**: ~2GB for 1000 products

## 🔍 Troubleshooting

### Common Issues

1. **API Not Responding**
   - Check `/health` endpoint
   - Verify all dependencies are installed
   - Check logs for initialization errors

2. **Slow Responses**
   - First request may be slow (cold start)
   - Subsequent requests should be faster
   - Consider using caching for frequently accessed data

3. **No Results Found**
   - Check if data is properly loaded
   - Verify query format
   - Try different search terms

### Health Check Status
- `healthy`: System fully operational
- `partial`: Basic functionality working (no LLM)
- `unhealthy`: System not properly initialized

## 📈 Monitoring

The API includes built-in monitoring:
- Health check endpoint
- Performance metrics
- Error logging
- System status tracking

## 🔐 Security

For production deployment:
1. Configure CORS properly
2. Add authentication if needed
3. Use HTTPS
4. Set up rate limiting
5. Monitor API usage

## 📞 Support

For issues or questions:
1. Check the health endpoint
2. Review the logs
3. Test with sample queries
4. Verify data format

## 🚀 Next Steps

1. **Deploy to Render** using the provided configuration
2. **Test the endpoints** using the examples above
3. **Integrate with your AR/VR app** using the provided client examples
4. **Add your Google API key** for enhanced AI responses
5. **Customize the prompts** for your specific use case 