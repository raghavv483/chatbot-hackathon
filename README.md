# 🛒 Walmart RAG System

A Retrieval-Augmented Generation (RAG) system for Walmart product search and recommendations, optimized for AR/VR applications.

## 🚀 Features

- **Smart Product Search**: Natural language queries for product discovery
- **Vector Database**: Fast semantic search using ChromaDB
- **AI-Powered Responses**: Google Gemini integration for intelligent responses
- **REST API**: FastAPI-based endpoints for easy integration
- **AR/VR Ready**: Optimized for AR/VR applications with CORS support
- **Production Ready**: Deployable on Render with comprehensive error handling

## 📁 Project Structure

```
├── app.py                 # FastAPI application
├── walmart_rag_optimized.py  # Optimized RAG system
├── walmart_rag_simple.py     # Simplified version
├── requirements.txt       # Python dependencies
├── render.yaml           # Render deployment config
├── README_API.md         # API documentation
├── DEBUGGING_GUIDE.md    # Debugging guide
└── .gitignore           # Git ignore rules
```

## 🛠️ Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/walmart-rag-system.git
   cd walmart-rag-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the API**
   ```bash
   python app.py
   ```

4. **Test the API**
   ```bash
   curl http://localhost:8000/health
   ```

### Deploy to Render

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Render**
   - Go to [render.com](https://render.com)
   - Connect your GitHub repository
   - Render will automatically detect the `render.yaml` configuration
   - Your API will be available at: `https://your-app-name.onrender.com`

## 📚 API Endpoints

### Health Check
```http
GET /health
```

### Query Products
```http
POST /query
Content-Type: application/json

{
  "query": "What are the best clothing deals in Jaipur?",
  "filters": {
    "store": "Jaipur",
    "category": "Clothing"
  }
}
```

### Get Products
```http
GET /products?brand=Nike&store=Jaipur&limit=10
```

## 🔧 AR/VR Integration

```javascript
// Example for AR/VR applications
const API_URL = 'https://your-app-name.onrender.com';

// Search for products
const response = await fetch(`${API_URL}/query`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: "Show me Nike shoes in Jaipur",
    filters: { store: "Jaipur", brand: "Nike" }
  })
});

const result = await response.json();
// result.products contains the product data for AR/VR display
```

## 🎯 Use Cases

- **AR Shopping**: Overlay product information in real-time
- **VR Store Navigation**: Intelligent product recommendations
- **Voice Search**: Natural language product queries
- **Smart Recommendations**: AI-powered product suggestions

## 🛠️ Configuration

### Environment Variables
- `GOOGLE_API_KEY`: Google Generative AI API key (optional)
- `PORT`: Server port (default: 8000)

### Data Source
The system automatically looks for `walmart_products.csv` in multiple locations. If not found, it uses sample data.

## 📊 Performance

- **Cold Start**: ~30-60 seconds (model loading)
- **Query Response**: ~1-3 seconds
- **Memory Usage**: ~2GB for 1000 products
- **Concurrent Requests**: Supports multiple simultaneous queries

## 🔍 Troubleshooting

See [DEBUGGING_GUIDE.md](DEBUGGING_GUIDE.md) for comprehensive troubleshooting information.

## 📖 Documentation

- [API Documentation](README_API.md) - Complete API reference
- [Debugging Guide](DEBUGGING_GUIDE.md) - Troubleshooting and optimization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:
1. Check the [Debugging Guide](DEBUGGING_GUIDE.md)
2. Test the `/health` endpoint
3. Review the API documentation
4. Open an issue on GitHub

## 🚀 Hackathon Ready

This project is optimized for hackathons with:
- ✅ Quick setup and deployment
- ✅ Comprehensive error handling
- ✅ AR/VR integration examples
- ✅ Production-ready API
- ✅ Detailed documentation

---

**Built with ❤️ for AR/VR Innovation** 