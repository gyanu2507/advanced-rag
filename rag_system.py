"""
RAG (Retrieval Augmented Generation) system implementation using FREE APIs.
"""
import os
from typing import List, Optional
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from advanced_chunking import AdvancedChunker

load_dotenv()


class RAGSystem:
    """RAG system for document question-answering using free models."""
    
    def __init__(self, persist_directory: str = "./chroma_db", use_api: bool = True):
        """Initialize the RAG system.
        
        Args:
            persist_directory: Directory to store vector database
            use_api: If True, use Hugging Face Inference API (free, requires token)
                    If False, use local models (completely free, slower)
        """
        self.persist_directory = persist_directory
        self.use_api = use_api
        
        # Initialize embeddings (using better free models)
        print("Loading embeddings model (this may take a minute on first run)...")
        try:
            # Try better models first, fallback to smaller ones
            embedding_models = [
                "BAAI/bge-small-en-v1.5",  # Better quality, still fast (384 dim)
                "sentence-transformers/all-mpnet-base-v2",  # High quality (768 dim)
                "sentence-transformers/all-MiniLM-L6-v2",  # Fast fallback (384 dim)
            ]
            
            for model_name in embedding_models:
                try:
                    print(f"  Trying {model_name}...")
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name=model_name,
                        model_kwargs={'device': 'cpu'},
                        encode_kwargs={'normalize_embeddings': True}  # Better similarity search
                    )
                    print(f"✓ Loaded embedding model: {model_name}")
                    break
                except Exception as e:
                    print(f"Failed to load {model_name}, trying next...")
                    if model_name == embedding_models[-1]:
                        raise
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            raise
        
        # Initialize LLM
        print("Loading language model...")
        if use_api:
            # Use Hugging Face Inference API (free tier available)
            hf_token = os.getenv("HUGGINGFACE_API_TOKEN", "")
            if hf_token:
                from langchain_community.llms import HuggingFaceEndpoint
                self.llm = HuggingFaceEndpoint(
                    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
                    huggingfacehub_api_token=hf_token,
                    task="text-generation",
                    model_kwargs={
                        "temperature": 0.7,
                        "max_new_tokens": 512,
                    }
                )
            else:
                # Fallback to local model if no token
                self._init_local_llm()
        else:
            # Use local model (completely free, no API needed)
            self._init_local_llm()
        
        self.vectorstore: Optional[Chroma] = None
        self.qa_chain: Optional[RetrievalQA] = None
        
        # Load existing vectorstore if it exists
        if os.path.exists(persist_directory):
            try:
                self.vectorstore = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embeddings
                )
                self._create_qa_chain()
            except Exception:
                pass
    
    def _init_local_llm(self):
        """Initialize a local LLM (completely free, no API needed)."""
        try:
            # Use Hugging Face's free inference API with a small model
            # This works without authentication for public models
            from langchain_community.llms import HuggingFaceHub
            
            # Try to use a free public model via API
            try:
                self.llm = HuggingFaceHub(
                    repo_id="google/flan-t5-base",
                    model_kwargs={"temperature": 0.7, "max_length": 512}
                )
                print("✓ Using Hugging Face free API (no token needed)")
                return
            except:
                pass
            
            # Fallback: Use local model if API doesn't work
            print("Loading local model (this may take a few minutes on first run)...")
            model_name = "google/flan-t5-small"  # Smaller model for faster loading
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="cpu"
            )
            
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512,
                temperature=0.7
            )
            
            self.llm = HuggingFacePipeline(pipeline=pipe)
            print("✓ Using local model (completely free, no internet needed)")
        except Exception as e:
            print(f"Warning: Could not load LLM model: {e}")
            print("Using retrieval-only mode (will show relevant document chunks)")
            # Fallback to a very simple approach
            self.llm = None
    
    def _create_qa_chain(self):
        """Create the QA chain with custom prompt."""
        if self.vectorstore is None:
            return
        
        # Only create QA chain if we have an LLM
        if self.llm is None:
            print("No LLM available, will use retrieval-only mode")
            self.qa_chain = None
            return
        
        prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Provide a detailed and accurate answer based on the context provided:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        try:
            # Enhanced retriever with better search
            try:
                retriever = self.vectorstore.as_retriever(
                    search_type="similarity_score_threshold",  # Better filtering
                    search_kwargs={
                        "k": 5,  # Get more candidates
                        "score_threshold": 0.3  # Quality threshold
                    }
                )
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=retriever,
                    chain_type_kwargs={"prompt": PROMPT},
                    return_source_documents=True
                )
            except:
                # Fallback to standard retriever
                retriever = self.vectorstore.as_retriever(
                    search_kwargs={"k": 5}  # Get top 5 most relevant chunks
                )
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=retriever,
                    chain_type_kwargs={"prompt": PROMPT},
                    return_source_documents=True
                )
            print("✓ QA chain created successfully")
        except Exception as e:
            print(f"Warning: Could not create QA chain: {e}")
            print("Will use retrieval-only mode")
            self.qa_chain = None
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        chunk_strategy: str = "smart"
    ):
        """Add documents to the vectorstore with advanced chunking.
        
        Args:
            texts: List of text documents to add
            metadatas: Optional metadata for each document
            chunk_strategy: Chunking strategy - "smart", "semantic", "sentence_aware", "paragraph"
        """
        # Use advanced chunking for better results
        chunker = AdvancedChunker(
            chunk_size=512,  # Optimal for embeddings
            chunk_overlap=100  # Good overlap for context
        )
        
        documents = []
        for i, text in enumerate(texts):
            # Use smart chunking strategy
            if chunk_strategy == "smart":
                chunk_docs = chunker.smart_chunk(text, metadata=metadatas[i] if metadatas and i < len(metadatas) else {})
            elif chunk_strategy == "semantic":
                chunk_docs = chunker.semantic_chunk(text, metadata=metadatas[i] if metadatas and i < len(metadatas) else {})
            elif chunk_strategy == "sentence_aware":
                chunk_docs = chunker.sentence_aware_chunk(text, metadata=metadatas[i] if metadatas and i < len(metadatas) else {})
            elif chunk_strategy == "paragraph":
                chunk_docs = chunker.paragraph_chunk(text, metadata=metadatas[i] if metadatas and i < len(metadatas) else {})
            else:
                chunk_docs = chunker.smart_chunk(text, metadata=metadatas[i] if metadatas and i < len(metadatas) else {})
            
            documents.extend(chunk_docs)
        
        if self.vectorstore is None:
            # Create new vectorstore
            print(f"Creating vectorstore with {len(documents)} document chunks...")
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            print("✓ Vectorstore created")
        else:
            # Add to existing vectorstore
            print(f"Adding {len(documents)} document chunks to existing vectorstore...")
            self.vectorstore.add_documents(documents)
            self.vectorstore.persist()
            print("✓ Documents added")
        
        self._create_qa_chain()
    
    def query(self, question: str, document_ids: Optional[List[str]] = None) -> dict:
        """Query the RAG system with a question.
        
        Args:
            question: The question to ask
            document_ids: Optional list of document IDs to filter results to specific files
        """
        # Check if we have documents loaded
        if self.vectorstore is None:
            return {
                "answer": "No documents have been loaded yet. Please upload a document first.",
                "sources": []
            }
        
        try:
            # Build filter if document_ids are provided
            filter_dict = None
            if document_ids and len(document_ids) > 0:
                # ChromaDB supports filtering with "$in" for multiple values
                # But we'll use a simpler approach: filter by document_id in list
                filter_dict = {"document_id": {"$in": document_ids}}
            
            # Try to get documents from vectorstore with optional filter
            if filter_dict:
                # For ChromaDB, we need to use where filter differently
                # Let's use a workaround: search all and filter results
                all_docs = self.vectorstore.similarity_search(question, k=20)
                docs = [doc for doc in all_docs if doc.metadata.get("document_id") in document_ids][:5]
            else:
                docs = self.vectorstore.similarity_search(question, k=5)
            
            if not docs:
                if document_ids:
                    return {
                        "answer": "No relevant information found in the selected document(s).",
                        "sources": []
                    }
                return {
                    "answer": "No relevant information found in the documents.",
                    "sources": []
                }
            
            # If we have an LLM and QA chain, use it for better answers
            if self.qa_chain is not None and self.llm is not None:
                try:
                    # Create filtered retriever if document_ids are provided
                    if document_ids and len(document_ids) > 0:
                        # Custom retriever that filters by document_ids
                        def filtered_retriever(query: str):
                            all_docs = self.vectorstore.similarity_search(query, k=20)
                            filtered = [doc for doc in all_docs if doc.metadata.get("document_id") in document_ids]
                            return filtered[:5]
                        
                        # Temporarily replace retriever
                        original_retriever = self.qa_chain.retriever
                        # Create a simple retriever wrapper
                        from langchain.schema import BaseRetriever
                        class FilteredRetriever(BaseRetriever):
                            def get_relevant_documents(self, query: str):
                                return filtered_retriever(query)
                        
                        self.qa_chain.retriever = FilteredRetriever()
                        result = self.qa_chain.invoke({"query": question})
                        self.qa_chain.retriever = original_retriever
                    else:
                        result = self.qa_chain.invoke({"query": question})
                    
                    # Enhanced source information
                    sources = []
                    for doc in result.get("source_documents", []):
                        source_info = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                        if doc.metadata:
                            source_info += f"\n[Chunk {doc.metadata.get('chunk_index', '?')}/{doc.metadata.get('total_chunks', '?')}]"
                        sources.append(source_info)
                    
                    return {
                        "answer": result["result"],
                        "sources": sources,
                        "num_sources": len(sources)
                    }
                except Exception as e:
                    print(f"QA chain error, falling back to retrieval: {e}")
                    # Fall through to retrieval-only mode
            
            # Enhanced retrieval-only mode with better formatting
            context_parts = []
            for i, doc in enumerate(docs, 1):
                chunk_info = f"[Source {i}]"
                if doc.metadata and 'chunk_index' in doc.metadata:
                    chunk_info += f" (Chunk {doc.metadata.get('chunk_index', '?')})"
                context_parts.append(f"{chunk_info}\n{doc.page_content}")
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Create a better formatted answer
            answer = f"Based on the uploaded documents:\n\n{context[:1500]}"
            if len(context) > 1500:
                answer += "\n\n[... more content available in sources ...]"
            
            sources = []
            for doc in docs:
                source_text = doc.page_content[:250] + "..." if len(doc.page_content) > 250 else doc.page_content
                if doc.metadata:
                    source_text += f"\n[Metadata: {doc.metadata.get('chunk_index', 'N/A')}/{doc.metadata.get('total_chunks', 'N/A')}]"
                sources.append(source_text)
            
            return {
                "answer": answer,
                "sources": sources,
                "num_sources": len(sources)
            }
            
        except Exception as e:
            return {
                "answer": f"Error processing query: {str(e)}",
                "sources": []
            }
    
    def clear_documents(self):
        """Clear all documents from the vectorstore."""
        import shutil
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
        self.vectorstore = None
        self.qa_chain = None

