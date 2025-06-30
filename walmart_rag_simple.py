#!/usr/bin/env python3
"""
Walmart RAG System - Simplified Version
A Retrieval-Augmented Generation system for Walmart product data with minimal dependencies.
"""

import os
import sys
import time
import logging
import warnings
import subprocess
from typing import List, Dict, Any, Optional, Tuple
from functools import wraps
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.ERROR)

# Set environment variables
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Try to import required libraries with fallbacks
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    print("⚠ pandas not available, using basic data structures")
    PANDAS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠ sentence-transformers not available")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    print("⚠ chromadb not available")
    CHROMADB_AVAILABLE = False

# Optional LLM imports
LLM_AVAILABLE = False
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import PromptTemplate
    from langchain.chains import LLMChain
    LLM_AVAILABLE = True
except ImportError:
    print("⚠ langchain packages not available, using mock responses")

def performance_monitor(func):
    """Decorator to monitor function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        print(f"⏱️ {func.__name__} took {duration:.2f} seconds")
        return result
    return wrapper

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
    """Simplified Walmart RAG System with comprehensive error handling."""
    
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
        
        print("🚀 Initializing Walmart RAG System (Simplified)...")
    
    def install_dependencies(self) -> bool:
        """Install required dependencies with error handling."""
        packages = [
            "pandas",
            "numpy",
            "chromadb",
            "sentence-transformers"
        ]
        
        print("📦 Installing dependencies...")
        success_count = 0
        for package in packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
                print(f"  ✓ {package}")
                success_count += 1
            except subprocess.CalledProcessError:
                print(f"  ✗ Failed to install {package}")
        
        print(f"✓ Installed {success_count}/{len(packages)} packages")
        return success_count > 0
    
    @performance_monitor
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
                        # Read CSV manually
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
    
    @performance_monitor
    def create_documents_and_metadata(self) -> bool:
        """Create documents and metadata for vector database."""
        try:
            self.documents = []
            self.metadatas = []
            
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
    
    @performance_monitor
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
    
    @performance_monitor
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
    
    @performance_monitor
    def initialize_chroma_collection(self, collection_name: str = "walmart_products") -> bool:
        """Initialize ChromaDB collection with error handling."""
        if not CHROMADB_AVAILABLE:
            print("✗ ChromaDB not available")
            return False
        
        try:
            # Disable telemetry to avoid warnings
            client = chromadb.Client(chromadb.config.Settings(
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
    
    @performance_monitor
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
            
            # Extract and format results
            context_chunks = results["documents"][0]
            context = "\n".join(context_chunks)
            
            return {
                "context": context,
                "documents": context_chunks,
                "metadatas": results.get("metadatas", [[]])[0],
                "distances": results.get("distances", [[]])[0]
            }
            
        except Exception as e:
            print(f"✗ Error querying RAG system: {e}")
            return {"context": "", "documents": [], "metadatas": [], "distances": []}
    
    def get_rag_response(self, query: str, filters: Optional[Dict[str, Any]] = None) -> str:
        """Get RAG response with comprehensive error handling."""
        try:
            print(f"🔍 Searching for: {query}")
            results = self.query_rag_system(query, filters=filters)
            
            if not results["documents"]:
                return "I couldn't find any relevant products for your query. Please try a different search term or check other categories."
            
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
            
            return response
            
        except Exception as e:
            print(f"✗ Error in RAG response: {e}")
            return f"Sorry, I encountered an error while processing your request: {str(e)}"
    
    def system_health_check(self) -> Dict[str, Any]:
        """Check the health of all system components."""
        health_status = {
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
        
        print("🏥 System Health Check:")
        for component, status in health_status.items():
            status_icon = "✓" if status else "✗"
            print(f"  {status_icon} {component}: {status}")
        
        return health_status
    
    def print_system_summary(self):
        """Print system summary and statistics."""
        if self.df is not None:
            print(f"\n📊 System Summary:")
            print(f"  • Total Products: {len(self.df)}")
            
            # Count unique values
            if PANDAS_AVAILABLE:
                print(f"  • Categories: {self.df['category'].nunique()}")
                print(f"  • Brands: {self.df['brand'].nunique()}")
                print(f"  • Stores: {self.df['store_location'].nunique()}")
                print(f"  • Average Price: ₹{self.df['price'].mean():.0f}")
                print(f"  • Average Discount: {self.df['discount'].mean():.1f}%")
            else:
                # Manual counting for SimpleDataFrame
                categories = set(self.df['category'])
                brands = set(self.df['brand'])
                stores = set(self.df['store_location'])
                avg_price = sum(self.df['price']) / len(self.df['price'])
                avg_discount = sum(self.df['discount']) / len(self.df['discount'])
                
                print(f"  • Categories: {len(categories)}")
                print(f"  • Brands: {len(brands)}")
                print(f"  • Stores: {len(stores)}")
                print(f"  • Average Price: ₹{avg_price:.0f}")
                print(f"  • Average Discount: {avg_discount:.1f}%")
    
    def run_interactive_mode(self):
        """Run the system in interactive mode."""
        print("\n🎯 Interactive Mode - Type 'quit' to exit")
        print("=" * 50)
        
        while True:
            try:
                query = input("\n🔍 Enter your query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not query:
                    print("Please enter a valid query.")
                    continue
                
                # Ask for filters
                use_filters = input("Apply filters? (y/n): ").strip().lower()
                filters = None
                
                if use_filters == 'y':
                    store = input("Store location (or press Enter to skip): ").strip()
                    category = input("Category (or press Enter to skip): ").strip()
                    brand = input("Brand (or press Enter to skip): ").strip()
                    
                    filters = {}
                    if store:
                        filters["store"] = store
                    if category:
                        filters["category"] = category
                    if brand:
                        filters["brand"] = brand
                
                # Get response
                response = self.get_rag_response(query, filters)
                print(f"\n🤖 Response: {response}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"✗ Error: {e}")

def main():
    """Main function to run the Walmart RAG system."""
    print("🛒 Walmart RAG System - Simplified Version")
    print("=" * 50)
    
    # Initialize system
    rag_system = WalmartRAGSystem()
    
    # Install dependencies
    if not rag_system.install_dependencies():
        print("⚠ Some dependencies failed to install, continuing with available packages")
    
    # Initialize components
    if not rag_system.load_data():
        print("✗ Failed to load data")
        return
    
    if not rag_system.create_documents_and_metadata():
        print("✗ Failed to create documents")
        return
    
    if not rag_system.initialize_embedding_model():
        print("⚠ No embedding model available, skipping vector search")
    
    if rag_system.model and not rag_system.generate_embeddings():
        print("✗ Failed to generate embeddings")
        return
    
    if not rag_system.initialize_chroma_collection():
        print("⚠ No ChromaDB available, skipping vector database")
    
    if rag_system.collection and not rag_system.add_documents_to_collection():
        print("✗ Failed to add documents to collection")
        return
    
    # Initialize LLM (optional)
    rag_system.initialize_llm()
    rag_system.create_rag_chain()
    
    # Health check
    rag_system.system_health_check()
    rag_system.print_system_summary()
    
    # Test queries
    print("\n🧪 Running Test Queries")
    print("=" * 30)
    
    test_queries = [
        "What are the best clothing deals in Jaipur?",
        "Show me Nike products under 3000 rupees",
        "What's available in the footwear category?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}: {query}")
        print("-" * 40)
        response = rag_system.get_rag_response(query)
        print(f"Response: {response}")
    
    # Interactive mode
    rag_system.run_interactive_mode()

if __name__ == "__main__":
    main() 