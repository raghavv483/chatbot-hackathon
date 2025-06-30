#!/usr/bin/env python3
"""
Walmart RAG API - FastAPI Application
Deployable API for Walmart product search and recommendations.
"""

import os
import sys
import time
import logging
import warnings
from typing import List, Dict, Any, Optional
from functools import wraps
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.ERROR)

# Set environment variables
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['GOOGLE_API_KEY'] = 'AIzaSyA2XmTAncsUZogi6RFkS_oUCbgNIKe5Aaw'

# Try to import required libraries with fallbacks
try:
    import pandas as pd  # type: ignore
    import numpy as np  # type: ignore
    PANDAS_AVAILABLE = True
except ImportError:
    print("⚠ pandas not available, using basic data structures")
    PANDAS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠ sentence-transformers not available")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import chromadb  # type: ignore
    from chromadb.config import Settings  # type: ignore
    CHROMADB_AVAILABLE = True
except ImportError:
    print("⚠ chromadb not available")
    CHROMADB_AVAILABLE = False

# Optional LLM imports
LLM_AVAILABLE = False
try:
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
    from langchain_core.prompts import PromptTemplate  # type: ignore
    from langchain.chains import LLMChain  # type: ignore
    LLM_AVAILABLE = True
except ImportError:
    print("⚠ langchain packages not available, using mock responses")

# FastAPI imports
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks  # type: ignore
    from fastapi.middleware.cors import CORSMiddleware  # type: ignore
    from pydantic import BaseModel  # type: ignore
    import uvicorn  # type: ignore
    FASTAPI_AVAILABLE = True
except ImportError:
    print("⚠ FastAPI not available")
    FASTAPI_AVAILABLE = False

# Pydantic models for API
class QueryRequest(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None
    n_results: Optional[int] = 5

class QueryResponse(BaseModel):
    response: str
    products: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    components: Dict[str, bool]
    message: str

class SimpleDataFrame:
    """Simple DataFrame replacement when pandas is not available."""
    
    def __init__(self, data: Dict[str, List]):
        self.data = data
        self.columns = list(data.keys())
    
    def __len__(self):
        return len(next(iter(self.data.values()))) if self.data else 0
    
    def iterrows(self):
        for i in range(len(self)):
            row_data = {col: self.data[col][i] for col in self.columns}
            yield i, row_data
    
    def fillna(self, value_dict):
        for col, value in value_dict.items():
            if col in self.data:
                for i in range(len(self.data[col])):
                    if self.data[col][i] is None or (isinstance(self.data[col][i], str) and self.data[col][i].strip() == ''):
                        self.data[col][i] = value
        return self
    
    def __getitem__(self, key):
        if isinstance(key, str):
            return self.data[key]
        return self
    
    def __setitem__(self, key, value):
        if isinstance(key, str):
            self.data[key] = value

class WalmartRAGSystem:
    """Optimized Walmart RAG System with comprehensive error handling."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the RAG system."""
        self.api_key = api_key
        self.df = None
        self.model = None
        self.collection = None
        self.llm = None
        self.rag_chain = None
        self.embeddings = []
        self.documents = []
        self.metadatas = []
        self.initialized = False
        
        print("🚀 Initializing Walmart RAG System...")
    
    def load_data(self, file_path: str = "walmart_products.csv") -> bool:
        """Load and validate Walmart product data."""
        try:
            # Try multiple possible file paths
            possible_paths = [
                file_path,
                "./walmart_products.csv",
                "../walmart_products.csv",
                "/content/walmart_products.csv"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    if PANDAS_AVAILABLE:
                        self.df = pd.read_csv(path)
                    else:
                        self.df = self._read_csv_manual(path)
                    print(f"✓ Data loaded from: {path}")
                    break
            
            if self.df is None:
                print("⚠ Creating sample data (file not found)")
                self.df = self._create_sample_data()
            
            # Clean and validate data
            self.df = self._clean_data(self.df)
            print(f"✓ Data loaded: {len(self.df)} products")
            return True
            
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            self.df = self._create_sample_data()
            return False
    
    def _read_csv_manual(self, file_path: str) -> SimpleDataFrame:
        """Read CSV file manually when pandas is not available."""
        data = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if not lines:
                return SimpleDataFrame({})
            
            # Parse header
            header = lines[0].strip().split(',')
            for col in header:
                data[col] = []
            
            # Parse data
            for line in lines[1:]:
                values = line.strip().split(',')
                for i, value in enumerate(values):
                    if i < len(header):
                        data[header[i]].append(value)
            
            return SimpleDataFrame(data)
    
    def _create_sample_data(self):
        """Create sample Walmart product data for testing."""
        sample_data = {
            'name': [
                'Adidas T-Shirt', 'Nike Running Shoes', 'Zara Party Dress', 
                'Levi\'s Classic Jeans', 'Puma Winter Jacket', 'Reebok Sports Shoes',
                'H&M Casual Shirt', 'Tommy Hilfiger Polo', 'Calvin Klein T-Shirt',
                'Under Armour Hoodie'
            ],
            'brand': [
                'Adidas', 'Nike', 'Zara', 'Levi\'s', 'Puma', 'Reebok',
                'H&M', 'Tommy Hilfiger', 'Calvin Klein', 'Under Armour'
            ],
            'category': [
                'Clothing', 'Footwear', 'Clothing', 'Clothing', 'Clothing', 'Footwear',
                'Clothing', 'Clothing', 'Clothing', 'Clothing'
            ],
            'price': [1200, 3500, 2800, 2200, 1800, 3200, 800, 1500, 1100, 2000],
            'discount': [15, 20, 10, 25, 30, 18, 40, 12, 22, 35],
            'description': [
                'Comfortable cotton t-shirt with Adidas logo',
                'Professional running shoes with advanced cushioning',
                'Elegant party dress perfect for special occasions',
                'Classic blue jeans with perfect fit',
                'Warm winter jacket with insulation',
                'Comfortable sports shoes for daily wear',
                'Casual shirt suitable for office and casual wear',
                'Premium polo shirt with classic design',
                'Comfortable t-shirt with modern fit',
                'Warm hoodie perfect for cold weather'
            ],
            'stock_quantity': [50, 30, 25, 40, 35, 45, 60, 20, 55, 30],
            'store_location': [
                'Jaipur', 'Mumbai', 'Jaipur', 'Delhi', 'Jaipur', 'Mumbai',
                'Delhi', 'Jaipur', 'Mumbai', 'Delhi'
            ]
        }
        
        if PANDAS_AVAILABLE:
            return pd.DataFrame(sample_data)
        else:
            return SimpleDataFrame(sample_data)
    
    def _clean_data(self, df):
        """Clean and validate the dataset."""
        # Fill missing values
        df = df.fillna({
            'brand': 'Unknown',
            'category': 'General',
            'price': 0,
            'discount': 0,
            'description': 'No description available',
            'stock_quantity': 0,
            'store_location': 'Unknown'
        })
        
        # Ensure numeric columns are numeric
        numeric_cols = ['price', 'discount', 'stock_quantity']
        for col in numeric_cols:
            if col in df.columns:
                if PANDAS_AVAILABLE:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                else:
                    # Manual numeric conversion
                    cleaned_values = []
                    for val in df[col]:
                        try:
                            cleaned_values.append(float(val) if val is not None else 0.0)
                        except (ValueError, TypeError):
                            cleaned_values.append(0.0)
                    df[col] = cleaned_values
        
        return df
    
    def create_documents_and_metadata(self) -> bool:
        """Create documents and metadata for vector database."""
        try:
            self.documents = []
            self.metadatas = []
            
            if self.df is None:
                print("✗ No data available")
                return False
            
            for idx, row in self.df.iterrows():
                # Create rich document text
                doc_text = (
                    f"Product: {row['name']}, Brand: {row['brand']}, Category: {row['category']}, "
                    f"Price: ₹{row['price']:.0f}, Discount: {row['discount']:.0f}%, "
                    f"Description: {row['description']}, Stock: {row['stock_quantity']:.0f} units, "
                    f"Store: {row['store_location']}"
                )
                
                self.documents.append(doc_text)
                
                # Create metadata for filtering
                metadata = {
                    "brand": str(row['brand']),
                    "category": str(row['category']),
                    "store": str(row['store_location']),
                    "price": float(row['price']),
                    "discount": float(row['discount']),
                    "stock": int(row['stock_quantity'])
                }
                self.metadatas.append(metadata)
            
            print(f"✓ Created {len(self.documents)} documents with metadata")
            return True
            
        except Exception as e:
            print(f"✗ Error creating documents: {e}")
            return False
    
    def initialize_embedding_model(self) -> bool:
        """Initialize the sentence transformer model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("✗ Sentence transformers not available")
            return False
        
        try:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            print("✓ Sentence transformer model loaded")
            return True
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return False
    
    def generate_embeddings(self, batch_size: int = 32) -> bool:
        """Generate embeddings with batching for memory efficiency."""
        if not self.model:
            print("✗ No embedding model available")
            return False
        
        try:
            self.embeddings = []
            for i in range(0, len(self.documents), batch_size):
                batch = self.documents[i:i + batch_size]
                batch_embeddings = self.model.encode(batch, show_progress_bar=True)
                self.embeddings.extend(batch_embeddings.tolist())
                
            print(f"✓ Generated {len(self.embeddings)} embeddings")
            return True
        except Exception as e:
            print(f"✗ Error generating embeddings: {e}")
            return False
    
    def initialize_chroma_collection(self, collection_name: str = "walmart_products") -> bool:
        """Initialize ChromaDB collection with error handling."""
        if not CHROMADB_AVAILABLE:
            print("✗ ChromaDB not available")
            return False
        
        try:
            # Disable telemetry to avoid warnings
            client = chromadb.Client(Settings(
                anonymized_telemetry=False,
                is_persistent=True
            ))
            
            # Get or create collection
            self.collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Walmart products for RAG system"}
            )
            
            print(f"✓ ChromaDB collection '{collection_name}' initialized")
            return True
        except Exception as e:
            print(f"✗ Error initializing ChromaDB: {e}")
            return False
    
    def add_documents_to_collection(self, batch_size: int = 100) -> bool:
        """Add documents to ChromaDB collection."""
        if not self.collection:
            print("✗ No ChromaDB collection available")
            return False
        
        try:
            # Generate unique IDs
            ids = [f"prod_{i}" for i in range(len(self.documents))]
            
            # Add documents in batches to avoid memory issues
            for i in range(0, len(self.documents), batch_size):
                end_idx = min(i + batch_size, len(self.documents))
                
                self.collection.add(
                    documents=self.documents[i:end_idx],
                    embeddings=self.embeddings[i:end_idx],
                    metadatas=self.metadatas[i:end_idx],
                    ids=ids[i:end_idx]
                )
            
            print(f"✓ Added {len(self.documents)} documents to collection")
            return True
        except Exception as e:
            print(f"✗ Error adding documents: {e}")
            return False
    
    def initialize_llm(self) -> bool:
        """Initialize Google Generative AI LLM."""
        if not LLM_AVAILABLE:
            print("⚠ LLM packages not available, using mock responses")
            return False
        
        try:
            if self.api_key:
                os.environ["GOOGLE_API_KEY"] = self.api_key
            
            # Check if API key is available
            if not os.environ.get("GOOGLE_API_KEY"):
                print("⚠ Warning: GOOGLE_API_KEY not found. Using mock responses.")
                return False
            
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash-latest",
                google_api_key=os.environ["GOOGLE_API_KEY"],
                temperature=0.3,
                max_output_tokens=1000
            )
            
            print("✓ LLM initialized successfully")
            return True
        except Exception as e:
            print(f"✗ Error initializing LLM: {e}")
            return False
    
    def create_rag_chain(self) -> bool:
        """Create the RAG chain with error handling."""
        if not LLM_AVAILABLE or not self.llm:
            print("⚠ No LLM available, skipping RAG chain creation")
            return False
        
        try:
            template = """
You are a helpful Walmart shopping assistant. Use the following product information to answer the customer's question accurately and helpfully.

Context Information:
{context}

Customer Question: {question}

Instructions:
1. Provide a friendly, helpful response
2. Mention specific products, prices, and deals when relevant
3. Include store location information if available
4. Keep the response concise but informative
5. If no relevant products are found, suggest alternative categories or stores

Response:
"""
            
            prompt = PromptTemplate(input_variables=["context", "question"], template=template)
            self.rag_chain = LLMChain(llm=self.llm, prompt=prompt)
            
            print("✓ RAG chain created successfully")
            return True
        except Exception as e:
            print(f"✗ Error creating RAG chain: {e}")
            return False
    
    def query_rag_system(self, query: str, n_results: int = 5, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query the RAG system with optimized search."""
        if not self.collection or not self.model:
            print("✗ No collection or model available")
            return {"context": "", "documents": [], "metadatas": [], "distances": []}
        
        try:
            # Generate query embedding
            query_embedding = self.model.encode([query])[0]
            
            # Prepare query parameters
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": n_results
            }
            
            # Add filters if provided
            if filters:
                query_params["where"] = filters
            
            # Execute query
            results = self.collection.query(**query_params)
            
            # Check if results is valid
            if not results or "documents" not in results:
                return {"context": "", "documents": [], "metadatas": [], "distances": []}
            
            # Extract and format results
            try:
                # Ensure results is a valid object
                if hasattr(results, 'get') and hasattr(results, '__getitem__'):
                    context_chunks = results["documents"][0]
                    context = "\n".join(context_chunks)
                    
                    return {
                        "context": context,
                        "documents": context_chunks,
                        "metadatas": results.get("metadatas", [[]])[0],
                        "distances": results.get("distances", [[]])[0]
                    }
                else:
                    return {"context": "", "documents": [], "metadatas": [], "distances": []}
            except (TypeError, KeyError, IndexError):
                return {"context": "", "documents": [], "metadatas": [], "distances": []}
            
        except Exception as e:
            print(f"✗ Error querying RAG system: {e}")
            return {"context": "", "documents": [], "metadatas": [], "distances": []}
    
    def get_rag_response(self, query: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get RAG response with comprehensive error handling."""
        try:
            print(f"🔍 Searching for: {query}")
            results = self.query_rag_system(query, filters=filters)
            
            if not results["documents"]:
                return {
                    "response": "I couldn't find any relevant products for your query. Please try a different search term or check other categories.",
                    "products": [],
                    "metadata": {"found": False, "count": 0}
                }
            
            # Generate response using LLM
            if self.rag_chain:
                print("🤖 Generating AI response...")
                response = self.rag_chain.run({
                    "context": results["context"],
                    "question": query
                })
            else:
                # Fallback response
                response = f"Based on the search results, here are some relevant products:\n{results['context'][:500]}..."
            
            # Extract product information
            products = []
            for i, doc in enumerate(results["documents"]):
                if i < len(results.get("metadatas", [])):
                    metadata = results["metadatas"][i]
                    products.append({
                        "name": doc.split(", Brand:")[0].replace("Product: ", ""),
                        "brand": metadata.get("brand", "Unknown"),
                        "category": metadata.get("category", "Unknown"),
                        "price": metadata.get("price", 0),
                        "discount": metadata.get("discount", 0),
                        "store": metadata.get("store", "Unknown"),
                        "description": doc
                    })
            
            return {
                "response": response,
                "products": products,
                "metadata": {
                    "found": True,
                    "count": len(products),
                    "query": query,
                    "filters": filters
                }
            }
            
        except Exception as e:
            print(f"✗ Error in RAG response: {e}")
            return {
                "response": f"Sorry, I encountered an error while processing your request: {str(e)}",
                "products": [],
                "metadata": {"found": False, "error": str(e)}
            }
    
    def initialize_system(self) -> bool:
        """Initialize the complete RAG system."""
        try:
            # Load data
            if not self.load_data():
                return False
            
            # Create documents
            if not self.create_documents_and_metadata():
                return False
            
            # Initialize embedding model
            if not self.initialize_embedding_model():
                return False
            
            # Generate embeddings
            if not self.generate_embeddings():
                return False
            
            # Initialize ChromaDB
            if not self.initialize_chroma_collection():
                return False
            
            # Add documents to collection
            if not self.add_documents_to_collection():
                return False
            
            # Initialize LLM (optional)
            self.initialize_llm()
            self.create_rag_chain()
            
            self.initialized = True
            print("✓ RAG system initialized successfully")
            return True
            
        except Exception as e:
            print(f"✗ Error initializing system: {e}")
            return False
    
    def system_health_check(self) -> Dict[str, Any]:
        """Check the health of all system components."""
        health_status = {
            "initialized": self.initialized,
            "data_loaded": self.df is not None and len(self.df) > 0,
            "embeddings_generated": len(self.embeddings) > 0,
            "collection_ready": self.collection is not None,
            "llm_available": self.llm is not None,
            "rag_chain_ready": self.rag_chain is not None,
            "pandas_available": PANDAS_AVAILABLE,
            "chromadb_available": CHROMADB_AVAILABLE,
            "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE,
            "llm_packages_available": LLM_AVAILABLE
        }
        
        return health_status

# Global RAG system instance
rag_system = None

def initialize_rag_system():
    """Initialize the RAG system globally."""
    global rag_system
    if rag_system is None:
        rag_system = WalmartRAGSystem()
        rag_system.initialize_system()
    return rag_system

# Create FastAPI app
if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Walmart RAG API",
        description="API for Walmart product search and recommendations using RAG",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure this properly for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize the RAG system on startup."""
        global rag_system
        print("🚀 Starting Walmart RAG API...")
        rag_system = initialize_rag_system()
        print("✅ API startup complete")
    
    @app.get("/", response_model=Dict[str, str])
    async def root():
        """Root endpoint."""
        return {
            "message": "Walmart RAG API is running!",
            "version": "1.0.0",
            "endpoints": {
                "health": "/health",
                "query": "/query",
                "docs": "/docs"
            }
        }
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        global rag_system
        if rag_system is None:
            rag_system = initialize_rag_system()
        
        health_status = rag_system.system_health_check()
        
        # Determine overall status
        if health_status["initialized"]:
            status = "healthy"
            message = "RAG system is fully operational"
        elif health_status["data_loaded"] and health_status["embeddings_generated"]:
            status = "partial"
            message = "RAG system is partially operational (no LLM)"
        else:
            status = "unhealthy"
            message = "RAG system is not properly initialized"
        
        return HealthResponse(
            status=status,
            components=health_status,
            message=message
        )
    
    @app.post("/query", response_model=QueryResponse)
    async def query_products(request: QueryRequest):
        """Query products using RAG system."""
        global rag_system
        if rag_system is None:
            rag_system = initialize_rag_system()
        
        if not rag_system.initialized:
            raise HTTPException(status_code=503, detail="RAG system not initialized")
        
        try:
            result = rag_system.get_rag_response(
                query=request.query,
                filters=request.filters
            )
            
            return QueryResponse(
                response=result["response"],
                products=result["products"],
                metadata=result["metadata"]
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
    
    @app.get("/products", response_model=Dict[str, Any])
    async def get_products(
        brand: Optional[str] = None,
        category: Optional[str] = None,
        store: Optional[str] = None,
        limit: int = 10
    ):
        """Get products with optional filtering."""
        global rag_system
        if rag_system is None:
            rag_system = initialize_rag_system()
        
        if not rag_system.initialized:
            raise HTTPException(status_code=503, detail="RAG system not initialized")
        
        try:
            # Build filters
            filters = {}
            if brand:
                filters["brand"] = brand
            if category:
                filters["category"] = category
            if store:
                filters["store"] = store
            
            # Query the system
            results = rag_system.query_rag_system(
                query="all products",
                n_results=limit,
                filters=filters if filters else None
            )
            
            # Extract product information
            products = []
            for i, doc in enumerate(results["documents"]):
                if i < len(results.get("metadatas", [])):
                    metadata = results["metadatas"][i]
                    products.append({
                        "name": doc.split(", Brand:")[0].replace("Product: ", ""),
                        "brand": metadata.get("brand", "Unknown"),
                        "category": metadata.get("category", "Unknown"),
                        "price": metadata.get("price", 0),
                        "discount": metadata.get("discount", 0),
                        "store": metadata.get("store", "Unknown"),
                        "description": doc
                    })
            
            return {
                "products": products,
                "count": len(products),
                "filters": filters
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get products: {str(e)}")

if __name__ == "__main__":
    if not FASTAPI_AVAILABLE:
        print("❌ FastAPI not available. Please install: pip install fastapi uvicorn")
        sys.exit(1)
    
    # Run the API
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port) 