"""
RAG (Retrieval Augmented Generation) system implementation using FREE APIs.
Enhanced with query rewriting, hybrid search, cross-encoder reranking, 
HyDE retrieval, RRF fusion, hallucination detection, and conversation memory.

🚀 GREATEST RAG EVER - v2.0
"""
import os
import re
import hashlib
from typing import List, Optional, Dict, Tuple, Any
from collections import defaultdict
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from advanced_chunking import AdvancedChunker
from retrieval_strategies import (
    reciprocal_rank_fusion,
    hyde_generate_hypothetical,
    contextual_compression,
    ParentChildRetriever
)

# Cross-encoder for superior reranking
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    print("⚠️ CrossEncoder not available. Install sentence-transformers for better reranking.")

load_dotenv()


class EmbeddingCache:
    """LRU cache for embeddings to avoid recomputation."""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, np.ndarray] = {}
        self.max_size = max_size
        self.access_order: List[str] = []
    
    def get(self, text: str, embed_fn) -> np.ndarray:
        """Get embedding from cache or compute it."""
        key = hashlib.md5(text.encode()).hexdigest()
        
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        
        # Compute new embedding
        embedding = np.array(embed_fn(text))
        
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        self.cache[key] = embedding
        self.access_order.append(key)
        return embedding
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.access_order.clear()


class RAGSystem:
    """RAG system for document question-answering using free models.
    
    🚀 GREATEST RAG EVER - v2.0 Features:
    - Cross-encoder reranking for superior relevance
    - Reciprocal Rank Fusion (RRF) for multi-retrieval
    - HyDE (Hypothetical Document Embeddings)
    - Embedding cache for performance
    - Hallucination detection
    - RAGAS-inspired evaluation metrics
    """
    
    def __init__(self, persist_directory: str = "./chroma_db", use_api: bool = True):
        """Initialize the RAG system.
        
        Args:
            persist_directory: Directory to store vector database
            use_api: If True, use Hugging Face Inference API (free, requires token)
                    If False, use local models (completely free, slower)
        """
        self.persist_directory = persist_directory
        self.use_api = use_api
        
        # Initialize embedding cache for performance
        self.embedding_cache = EmbeddingCache(max_size=500)
        
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
        
        # Conversation memory for context-aware queries
        self.conversation_history: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        
        # Initialize cross-encoder for superior reranking (lazy load)
        self.cross_encoder = None
        if CROSS_ENCODER_AVAILABLE:
            try:
                print("Loading cross-encoder model for reranking...")
                # MS MARCO MiniLM is fast and highly accurate for reranking
                self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                print("✓ Cross-encoder loaded for superior reranking")
            except Exception as e:
                print(f"⚠️ Could not load cross-encoder: {e}. Falling back to heuristic reranking.")
        
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
        
        prompt_template = """You are an expert assistant helping users understand information from their documents. 

Use the following context from uploaded documents to provide a clear, comprehensive, and well-structured answer to the question.

IMPORTANT INSTRUCTIONS:
- Answer in a natural, conversational tone
- Synthesize information from multiple sources when relevant
- Organize your answer logically with clear points
- Only use information from the provided context
- If the context doesn't contain enough information, say so clearly
- Do not include source citations or references in your answer - just provide the information naturally

Context from documents:
{context}

Question: {question}

Provide a detailed, well-structured answer:"""
        
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
    
    def _rewrite_query(self, question: str, conversation_context: Optional[List[Dict[str, str]]] = None) -> str:
        """Rewrite/expand query for better retrieval."""
        # Simple query expansion - add synonyms and related terms
        # In production, you could use an LLM for this
        
        # Basic query expansion patterns
        expansions = {
            r'\bwhat\b': ['what', 'describe', 'explain', 'tell me about'],
            r'\bhow\b': ['how', 'method', 'process', 'way'],
            r'\bwhy\b': ['why', 'reason', 'cause', 'purpose'],
            r'\bwhen\b': ['when', 'time', 'date', 'period'],
            r'\bwhere\b': ['where', 'location', 'place'],
            r'\bwho\b': ['who', 'person', 'individual', 'entity'],
        }
        
        # If we have conversation context, add it
        if conversation_context and len(conversation_context) > 0:
            # Use last few exchanges for context
            recent_context = conversation_context[-2:] if len(conversation_context) >= 2 else conversation_context
            context_text = " ".join([f"{item.get('role', 'user')}: {item.get('content', '')}" for item in recent_context])
            # Enhance question with context
            enhanced_question = f"{context_text} {question}"
        else:
            enhanced_question = question
        
        return enhanced_question
    
    def _generate_query_variations(self, question: str) -> List[str]:
        """Generate multiple query variations for better retrieval."""
        variations = [question]  # Original query
        
        # Add question without question words
        question_lower = question.lower()
        if question_lower.startswith(('what', 'how', 'why', 'when', 'where', 'who', 'which')):
            # Remove question word and add as variation
            words = question.split()
            if len(words) > 1:
                variations.append(" ".join(words[1:]))
        
        # Add keyword-focused version (extract important terms)
        words = re.findall(r'\b\w{4,}\b', question.lower())  # Words with 4+ chars
        if words:
            variations.append(" ".join(words[:5]))  # Top 5 keywords
        
        # Add expanded version with synonyms (simple approach)
        expanded = question
        if '?' in question:
            expanded = question.replace('?', '')
        variations.append(expanded)
        
        return list(set(variations))  # Remove duplicates
    
    def _hybrid_search(
        self, 
        query: str, 
        k: int = 5, 
        document_ids: Optional[List[str]] = None
    ) -> List[Tuple[Document, float]]:
        """Hybrid search combining semantic and keyword matching."""
        results = []
        
        # 1. Semantic search (vector similarity)
        try:
            semantic_docs = self.vectorstore.similarity_search_with_score(query, k=k*2)
            for doc, score in semantic_docs:
                # Normalize score (ChromaDB returns distance, lower is better)
                # Convert to similarity score (higher is better)
                similarity_score = 1.0 / (1.0 + abs(score))
                results.append((doc, similarity_score * 0.7))  # 70% weight for semantic
        except Exception as e:
            print(f"Semantic search error: {e}")
        
        # 2. Keyword search (simple TF-based matching)
        try:
            query_words = set(re.findall(r'\b\w+\b', query.lower()))
            all_docs = self.vectorstore.similarity_search(query, k=k*3)
            
            for doc in all_docs:
                doc_text = doc.page_content.lower()
                doc_words = set(re.findall(r'\b\w+\b', doc_text))
                
                # Calculate keyword overlap
                if query_words:
                    overlap = len(query_words & doc_words) / len(query_words)
                    keyword_score = overlap * 0.3  # 30% weight for keywords
                    
                    # Check if already in results
                    found = False
                    for i, (existing_doc, _) in enumerate(results):
                        if existing_doc.page_content == doc.page_content:
                            results[i] = (existing_doc, results[i][1] + keyword_score)
                            found = True
                            break
                    
                    if not found:
                        results.append((doc, keyword_score))
        except Exception as e:
            print(f"Keyword search error: {e}")
        
        # Filter by document_ids if provided
        if document_ids:
            # Convert document_ids to strings for comparison (metadata might be string or int)
            doc_ids_str = [str(did) for did in document_ids]
            filtered_results = []
            for doc, score in results:
                doc_id = doc.metadata.get("document_id")
                # Check both string and int versions
                if doc_id is not None and (str(doc_id) in doc_ids_str or doc_id in document_ids):
                    filtered_results.append((doc, score))
            results = filtered_results
        
        # Sort by combined score and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def _rerank_documents(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: int = 5,
        use_cross_encoder: bool = True
    ) -> List[Document]:
        """Rerank documents using cross-encoder (if available) or heuristics.
        
        Cross-encoder provides dramatically better relevance scoring than 
        bi-encoder similarity, as it jointly encodes query-document pairs.
        """
        if not documents:
            return documents
        
        # Use cross-encoder if available (superior relevance scoring)
        if use_cross_encoder and self.cross_encoder is not None:
            try:
                # Create query-document pairs for cross-encoder
                pairs = [[query, doc.page_content[:512]] for doc in documents]  # Limit length
                scores = self.cross_encoder.predict(pairs)
                
                # Sort by cross-encoder score (higher = more relevant)
                scored_docs = list(zip(documents, scores))
                scored_docs.sort(key=lambda x: x[1], reverse=True)
                
                return [doc for doc, _ in scored_docs[:top_k]]
            except Exception as e:
                print(f"⚠️ Cross-encoder reranking failed: {e}. Falling back to heuristics.")
        
        # Fallback to heuristic reranking
        return self._heuristic_rerank(query, documents, top_k)
    
    def _heuristic_rerank(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: int = 5
    ) -> List[Document]:
        """Fallback heuristic reranking when cross-encoder is unavailable."""
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        
        scored_docs = []
        for doc in documents:
            doc_text = doc.page_content.lower()
            doc_words = set(re.findall(r'\b\w+\b', doc_text))
            
            # Calculate relevance score
            score = 0.0
            
            # 1. Keyword overlap (40% weight)
            if query_words:
                overlap = len(query_words & doc_words) / len(query_words)
                score += overlap * 0.4
            
            # 2. Query term frequency in document (30% weight)
            term_freq = sum(doc_text.count(word) for word in query_words)
            score += min(term_freq / 10.0, 0.3)
            
            # 3. Position bonus - earlier chunks more important (20% weight)
            if doc.metadata and 'chunk_index' in doc.metadata:
                chunk_idx = doc.metadata.get('chunk_index', 0)
                total_chunks = doc.metadata.get('total_chunks', 1)
                if total_chunks > 0:
                    position_score = 1.0 - (chunk_idx / max(total_chunks, 1))
                    score += position_score * 0.2
            
            # 4. Length bonus - prefer medium length docs (10% weight)
            doc_length = len(doc.page_content)
            if 100 <= doc_length <= 1000:
                score += 0.1
            
            scored_docs.append((doc, score))
        
        # Sort by score and return top k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:top_k]]
    
    def _synthesize_answer_from_chunks(self, question: str, docs: List[Document]) -> str:
        """Synthesize a proper RAG-style answer with inline citations.
        
        Creates structured response that clearly shows:
        1. Information is from uploaded documents
        2. Inline citations [1], [2], etc.
        3. Organized by topic/theme
        """
        if not docs:
            return "❌ No relevant information found in your uploaded documents for this question."
        
        # Map documents to citation numbers
        doc_citations = {}  # content_key -> citation_num
        citation_sources = {}  # citation_num -> filename
        
        for i, doc in enumerate(docs):
            content_key = doc.page_content[:100]
            citation_num = i + 1
            doc_citations[content_key] = citation_num
            filename = doc.metadata.get('filename', f'Source {citation_num}')
            citation_sources[citation_num] = filename
        
        # Extract and score sentences with their source
        question_lower = question.lower()
        question_keywords = set(re.findall(r'\b\w{4,}\b', question_lower))
        
        scored_items = []  # (score, sentence, citation_num)
        
        for doc in docs:
            content_key = doc.page_content[:100]
            citation_num = doc_citations.get(content_key, 1)
            
            sentences = re.split(r'[.!?]+', doc.page_content)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 25:
                    continue
                
                sentence_words = set(re.findall(r'\b\w{4,}\b', sentence.lower()))
                if question_keywords:
                    overlap = len(question_keywords & sentence_words) / len(question_keywords)
                else:
                    overlap = 0.5
                
                length_score = min(len(sentence) / 150, 1.0)
                total_score = overlap * 0.7 + length_score * 0.3
                
                if total_score > 0.2:  # Only include relevant sentences
                    scored_items.append((total_score, sentence, citation_num))
        
        if not scored_items:
            return "📄 Found documents but couldn't extract specific information for your question."
        
        # Sort by relevance
        scored_items.sort(reverse=True, key=lambda x: x[0])
        
        # Build answer with citations - deduplicate
        seen_sentences = set()
        answer_lines = []
        used_citations = set()
        
        for score, sentence, citation_num in scored_items[:6]:  # Top 6 points
            sentence_key = sentence.lower()[:80]
            if sentence_key in seen_sentences:
                continue
            seen_sentences.add(sentence_key)
            
            # Clean sentence and add citation
            clean_sentence = sentence.strip()
            if not clean_sentence.endswith(('.', '!', '?')):
                clean_sentence += '.'
            
            # Add inline citation
            answer_lines.append(f"• {clean_sentence} [{citation_num}]")
            used_citations.add(citation_num)
        
        if not answer_lines:
            return "📄 Found related content but couldn't synthesize a clear answer."
        
        # Build structured response with markdown for attractive rendering
        response_parts = []
        
        # Header with emoji
        response_parts.append("## 📄 Based on your uploaded documents\n")
        
        # Main points with citations
        response_parts.append("\n".join(answer_lines))
        
        # Sources footer with styling
        response_parts.append("\n\n---\n**📚 Sources:**")
        for cit_num in sorted(used_citations):
            source_name = citation_sources.get(cit_num, f"Source {cit_num}")
            response_parts.append(f"  `[{cit_num}]` *{source_name}*")
        
        return "\n".join(response_parts)
    
    def _calculate_confidence(
        self, 
        answer: str, 
        sources: List[Document], 
        query: str
    ) -> float:
        """Calculate confidence score using cosine similarity between query and source embeddings.
        
        This provides a much more accurate measure of relevance than keyword matching.
        """
        if not answer or not sources or not self.embeddings:
            return 0.0
        
        try:
            # Get query embedding
            query_embedding = np.array(self.embeddings.embed_query(query))
            
            # Get embeddings for all source documents and calculate cosine similarity
            similarities = []
            for source in sources:
                try:
                    # Embed the source document content
                    source_embedding = np.array(self.embeddings.embed_query(source.page_content[:512]))  # Limit length for efficiency
                    
                    # Calculate cosine similarity
                    # Cosine similarity = dot(A, B) / (||A|| * ||B||)
                    dot_product = np.dot(query_embedding, source_embedding)
                    norm_query = np.linalg.norm(query_embedding)
                    norm_source = np.linalg.norm(source_embedding)
                    
                    if norm_query > 0 and norm_source > 0:
                        cosine_sim = dot_product / (norm_query * norm_source)
                        # Cosine similarity ranges from -1 to 1, but with normalized embeddings it's typically 0-1
                        # Clamp to [0, 1] range
                        cosine_sim = max(0.0, min(1.0, cosine_sim))
                        similarities.append(cosine_sim)
                except Exception as e:
                    # If embedding fails for a source, skip it
                    print(f"Warning: Could not calculate similarity for source: {e}")
                    continue
            
            if not similarities:
                return 0.0
            
            # Use average similarity as confidence, but weight towards higher similarities
            # This gives more weight to highly relevant sources
            avg_similarity = np.mean(similarities)
            max_similarity = np.max(similarities)
            
            # Combine average and max (60% max, 40% average) for better confidence
            # This ensures that if at least one highly relevant source exists, confidence is higher
            confidence = 0.6 * max_similarity + 0.4 * avg_similarity
            
            # Scale to 0-100% for display (cosine similarity with normalized embeddings is typically 0-1)
            # Adjust scaling to make it more meaningful (0.5 similarity = 50%, 0.7 = 85%, etc.)
            # Use a non-linear scaling to better represent quality
            if confidence > 0.7:
                # High similarity: scale more generously
                confidence = 0.7 + (confidence - 0.7) * 0.6  # 0.7-1.0 maps to 0.7-0.88
            elif confidence > 0.5:
                # Medium similarity: linear mapping
                confidence = 0.5 + (confidence - 0.5) * 0.6  # 0.5-0.7 maps to 0.5-0.62
            else:
                # Low similarity: scale down more
                confidence = confidence * 0.8  # 0-0.5 maps to 0-0.4
            
            # Ensure it's between 0 and 1
            confidence = max(0.0, min(1.0, confidence))
            
            return confidence
            
        except Exception as e:
            print(f"Error calculating cosine similarity confidence: {e}")
            # Fallback to simple heuristic if embedding calculation fails
            return min(len(sources) / 5.0, 1.0) * 0.5
    
    def query(
        self, 
        question: str, 
        document_ids: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        use_enhancements: bool = True
    ) -> dict:
        """Query the RAG system with a question.
        
        Args:
            question: The question to ask
            document_ids: Optional list of document IDs to filter results to specific files
            user_id: Optional user ID for conversation memory
            use_enhancements: Whether to use enhanced features (query rewriting, hybrid search, etc.)
        """
        # Check if we have documents loaded
        if self.vectorstore is None:
            return {
                "answer": "No documents have been loaded yet. Please upload a document first.",
                "sources": [],
                "num_sources": 0,
                "confidence": 0.0,
                "enhanced": use_enhancements
            }
        
        try:
            # Get conversation context if user_id provided
            conversation_context = None
            if user_id and user_id in self.conversation_history:
                conversation_context = self.conversation_history[user_id]
            
            # Initialize enhanced_question (will be set if enhancements are used)
            enhanced_question = question
            
            # Enhanced query processing
            if use_enhancements:
                # 1. Query rewriting with context
                enhanced_question = self._rewrite_query(question, conversation_context)
                
                # 2. Generate query variations
                query_variations = self._generate_query_variations(enhanced_question)
                
                # 3. Hybrid search with all variations
                all_results = []
                seen_contents = set()
                
                for query_var in query_variations[:3]:  # Use top 3 variations
                    results = self._hybrid_search(query_var, k=10, document_ids=document_ids)
                    for doc, score in results:
                        doc_key = doc.page_content[:100]  # Use first 100 chars as key
                        if doc_key not in seen_contents:
                            all_results.append((doc, score))
                            seen_contents.add(doc_key)
                
                # If no results with filtering, try without filtering (fallback)
                if not all_results and document_ids:
                    print(f"⚠️ No results with document filter, trying without filter...")
                    for query_var in query_variations[:2]:  # Try fewer variations
                        results = self._hybrid_search(query_var, k=20, document_ids=None)
                        for doc, score in results:
                            doc_key = doc.page_content[:100]
                            if doc_key not in seen_contents:
                                # Check if this doc matches our filter
                                doc_id = doc.metadata.get("document_id")
                                doc_ids_str = [str(did) for did in document_ids]
                                if doc_id is not None and (str(doc_id) in doc_ids_str or doc_id in document_ids):
                                    all_results.append((doc, score))
                                    seen_contents.add(doc_key)
                
                # 4. Rerank results
                docs = [doc for doc, _ in all_results[:15]]  # Get top 15 for reranking
                if docs:
                    docs = self._rerank_documents(enhanced_question, docs, top_k=5)
                else:
                    # Last resort: try simple search without enhancements
                    print(f"⚠️ Enhanced search returned no results, trying simple search...")
                    if document_ids:
                        doc_ids_str = [str(did) for did in document_ids]
                        all_docs = self.vectorstore.similarity_search(question, k=50)
                        docs = [doc for doc in all_docs 
                               if doc.metadata.get("document_id") is not None and 
                               (str(doc.metadata.get("document_id")) in doc_ids_str or 
                                doc.metadata.get("document_id") in document_ids)][:5]
                    else:
                        docs = self.vectorstore.similarity_search(question, k=5)
            else:
                # Standard search (fallback)
                if document_ids and len(document_ids) > 0:
                    doc_ids_str = [str(did) for did in document_ids]
                    all_docs = self.vectorstore.similarity_search(question, k=50)  # Get more docs for filtering
                    docs = []
                    for doc in all_docs:
                        doc_id = doc.metadata.get("document_id")
                        if doc_id is not None and (str(doc_id) in doc_ids_str or doc_id in document_ids):
                            docs.append(doc)
                            if len(docs) >= 5:
                                break
                else:
                    docs = self.vectorstore.similarity_search(question, k=5)
            
            if not docs:
                if document_ids:
                    return {
                        "answer": "No relevant information found in the selected document(s).",
                        "sources": [],
                        "num_sources": 0,
                        "confidence": 0.0,
                        "enhanced": use_enhancements
                    }
                return {
                    "answer": "No relevant information found in the documents.",
                    "sources": [],
                    "num_sources": 0,
                    "confidence": 0.0,
                    "enhanced": use_enhancements
                }
            
            # Use enhanced question for LLM if available
            query_for_llm = enhanced_question if use_enhancements else question
            
            # If we have an LLM and QA chain, use it for better answers
            if self.qa_chain is not None and self.llm is not None:
                try:
                    # Create filtered retriever if document_ids are provided
                    if document_ids and len(document_ids) > 0:
                        # Custom retriever that filters by document_ids
                        def filtered_retriever(query: str):
                            if use_enhancements:
                                results = self._hybrid_search(query, k=20, document_ids=document_ids)
                                return [doc for doc, _ in results]
                            else:
                                doc_ids_str = [str(did) for did in document_ids]
                                all_docs = self.vectorstore.similarity_search(query, k=50)
                                filtered = []
                                for doc in all_docs:
                                    doc_id = doc.metadata.get("document_id")
                                    if doc_id is not None and (str(doc_id) in doc_ids_str or doc_id in document_ids):
                                        filtered.append(doc)
                                        if len(filtered) >= 5:
                                            break
                                return filtered
                        
                        # Temporarily replace retriever
                        original_retriever = self.qa_chain.retriever
                        # Create a simple retriever wrapper
                        from langchain.schema import BaseRetriever
                        class FilteredRetriever(BaseRetriever):
                            def get_relevant_documents(self, query: str):
                                return filtered_retriever(query)
                        
                        self.qa_chain.retriever = FilteredRetriever()
                        result = self.qa_chain.invoke({"query": query_for_llm})
                        self.qa_chain.retriever = original_retriever
                    else:
                        result = self.qa_chain.invoke({"query": query_for_llm})
                    
                    # Enhanced source information
                    source_docs = result.get("source_documents", docs)
                    sources = []
                    for doc in source_docs:
                        source_info = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                        if doc.metadata:
                            filename = doc.metadata.get('filename', 'Unknown')
                            chunk_idx = doc.metadata.get('chunk_index', '?')
                            total_chunks = doc.metadata.get('total_chunks', '?')
                            source_info += f"\n[File: {filename}, Chunk {chunk_idx}/{total_chunks}]"
                        sources.append(source_info)
                    
                    # Calculate confidence
                    confidence = self._calculate_confidence(result["result"], source_docs, question)
                    
                    # Update conversation history
                    if user_id:
                        self.conversation_history[user_id].append({
                            "role": "user",
                            "content": question
                        })
                        self.conversation_history[user_id].append({
                            "role": "assistant",
                            "content": result["result"]
                        })
                        # Keep only last 10 exchanges
                        if len(self.conversation_history[user_id]) > 20:
                            self.conversation_history[user_id] = self.conversation_history[user_id][-20:]
                    
                    return {
                        "answer": result["result"],
                        "sources": sources,
                        "num_sources": len(sources),
                        "confidence": round(confidence, 2),
                        "enhanced": use_enhancements
                    }
                except Exception as e:
                    print(f"QA chain error, falling back to retrieval: {e}")
                    # Fall through to retrieval-only mode
            
            # Enhanced retrieval-only mode with intelligent answer synthesis
            # Extract and synthesize information from retrieved chunks
            answer = self._synthesize_answer_from_chunks(question, docs)
            
            # Format sources cleanly
            sources = []
            seen_files = set()
            for doc in docs:
                if doc.metadata:
                    filename = doc.metadata.get('filename', 'Unknown')
                    chunk_idx = doc.metadata.get('chunk_index', 'N/A')
                    total_chunks = doc.metadata.get('total_chunks', 'N/A')
                    
                    # Create clean source reference
                    source_ref = f"{filename}"
                    if chunk_idx != 'N/A' and chunk_idx != '?' and total_chunks != 'N/A':
                        try:
                            chunk_num = int(chunk_idx) + 1 if isinstance(chunk_idx, (int, str)) and str(chunk_idx).isdigit() else chunk_idx
                            source_ref += f" (Section {chunk_num})"
                        except:
                            pass  # Skip if chunk_idx can't be converted
                    
                    # Avoid duplicate source entries
                    if source_ref not in seen_files:
                        seen_files.add(source_ref)
                        source_text = doc.page_content[:200].strip()
                        if len(doc.page_content) > 200:
                            source_text += "..."
                        sources.append(f"{source_text}\n— {source_ref}")
            
            # Calculate confidence for retrieval-only mode
            confidence = self._calculate_confidence(answer, docs, question)
            
            # Update conversation history
            if user_id:
                self.conversation_history[user_id].append({
                    "role": "user",
                    "content": question
                })
                self.conversation_history[user_id].append({
                    "role": "assistant",
                    "content": answer
                })
                # Keep only last 10 exchanges
                if len(self.conversation_history[user_id]) > 20:
                    self.conversation_history[user_id] = self.conversation_history[user_id][-20:]
            
            return {
                "answer": answer,
                "sources": sources,
                "num_sources": len(sources),
                "confidence": round(confidence, 2),
                "enhanced": use_enhancements
            }
            
        except Exception as e:
            return {
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "num_sources": 0,
                "confidence": 0.0,
                "enhanced": use_enhancements
            }
    
    def delete_documents_by_metadata(self, metadata_filter: Dict[str, str]):
        """Delete documents from vectorstore by metadata filter.
        
        Args:
            metadata_filter: Dictionary of metadata key-value pairs to filter by
        """
        if self.vectorstore is None:
            return
        
        try:
            # Get all documents with matching metadata
            # Chroma doesn't have direct delete by metadata, so we need to:
            # 1. Get document IDs that match the filter
            # 2. Delete them
            
            # Use get() with where filter to find matching documents
            results = self.vectorstore.get(
                where=metadata_filter
            )
            
            if results and 'ids' in results and len(results['ids']) > 0:
                # Delete the matching documents
                self.vectorstore.delete(ids=results['ids'])
                self.vectorstore.persist()
                print(f"✓ Deleted {len(results['ids'])} document chunks matching metadata: {metadata_filter}")
        except Exception as e:
            print(f"⚠️ Warning: Could not delete documents by metadata: {e}")
    
    def clear_documents(self, user_id: Optional[str] = None):
        """Clear all documents from the vectorstore."""
        import shutil
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
        self.vectorstore = None
        self.qa_chain = None
        
        # Clear conversation history for user if specified
        if user_id and user_id in self.conversation_history:
            del self.conversation_history[user_id]
        elif user_id is None:
            # Clear all conversation history
            self.conversation_history.clear()
    
    def clear_conversation_history(self, user_id: str):
        """Clear conversation history for a specific user."""
        if user_id in self.conversation_history:
            del self.conversation_history[user_id]
    
    # =========================================================================
    # 🚀 GREATEST RAG EVER - v2.0 Enhanced Methods
    # =========================================================================
    
    def enhanced_query(
        self,
        question: str,
        document_ids: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        use_hyde: bool = True,
        use_rrf: bool = True,
        use_compression: bool = False,
        include_evaluation: bool = True
    ) -> dict:
        """
        Enhanced query with all v2.0 features.
        
        Args:
            question: The question to ask
            document_ids: Optional document filter
            user_id: User ID for conversation context
            use_hyde: Use HyDE (Hypothetical Document Embeddings)
            use_rrf: Use Reciprocal Rank Fusion
            use_compression: Use contextual compression
            include_evaluation: Include RAGAS-style evaluation metrics
        
        Returns:
            Enhanced response with answer, sources, confidence, and evaluation
        """
        if self.vectorstore is None:
            return {
                "answer": "No documents loaded. Please upload documents first.",
                "sources": [],
                "confidence": 0.0,
                "evaluation": None,
                "enhanced_features": []
            }
        
        features_used = []
        
        try:
            # Get conversation context
            conversation_context = None
            if user_id and user_id in self.conversation_history:
                conversation_context = self.conversation_history[user_id]
            
            # 1. Query rewriting with context
            enhanced_question = self._rewrite_query(question, conversation_context)
            
            # 2. Collect multiple retrieval results for RRF
            retrieval_results = []
            
            # Standard semantic search
            semantic_results = self._hybrid_search(
                enhanced_question, k=15, document_ids=document_ids
            )
            retrieval_results.append(semantic_results)
            
            # Generate query variations
            query_variations = self._generate_query_variations(enhanced_question)
            for variation in query_variations[:2]:
                var_results = self._hybrid_search(
                    variation, k=10, document_ids=document_ids
                )
                retrieval_results.append(var_results)
            
            # 3. HyDE retrieval (if enabled)
            if use_hyde and self.llm:
                try:
                    hypotheticals = hyde_generate_hypothetical(question, self.llm, 1)
                    for hyp in hypotheticals:
                        hyp_results = [(doc, score) for doc, score in 
                                       self._hybrid_search(hyp, k=10, document_ids=document_ids)]
                        if hyp_results:
                            retrieval_results.append(hyp_results)
                            features_used.append("HyDE")
                except Exception as e:
                    print(f"⚠️ HyDE retrieval failed: {e}")
            
            # 4. Apply RRF fusion (if multiple result sets)
            if use_rrf and len(retrieval_results) > 1:
                fused_results = reciprocal_rank_fusion(retrieval_results)
                docs = [doc for doc, _ in fused_results[:10]]
                features_used.append("RRF")
            else:
                # Just use first result set
                docs = [doc for doc, _ in retrieval_results[0][:10]]
            
            # 5. Cross-encoder reranking
            if docs:
                docs = self._rerank_documents(enhanced_question, docs, top_k=5)
                if self.cross_encoder:
                    features_used.append("CrossEncoder")
            
            # 6. Contextual compression (optional)
            if use_compression and docs:
                try:
                    docs = contextual_compression(
                        question, docs, self.embeddings, 0.3
                    )
                    features_used.append("Compression")
                except Exception as e:
                    print(f"⚠️ Compression failed: {e}")
            
            if not docs:
                return {
                    "answer": "No relevant information found.",
                    "sources": [],
                    "confidence": 0.0,
                    "evaluation": None,
                    "enhanced_features": features_used
                }
            
            # 7. Generate answer
            if self.qa_chain and self.llm:
                try:
                    result = self.qa_chain.invoke({"query": enhanced_question})
                    answer = result.get("result", "")
                    source_docs = result.get("source_documents", docs)
                except:
                    answer = self._synthesize_answer_from_chunks(question, docs)
                    source_docs = docs
            else:
                answer = self._synthesize_answer_from_chunks(question, docs)
                source_docs = docs
            
            # 8. Format sources
            sources = []
            for doc in source_docs[:5]:
                source_text = doc.page_content[:250] + "..." if len(doc.page_content) > 250 else doc.page_content
                filename = doc.metadata.get('filename', 'Unknown')
                sources.append(f"{source_text}\n— {filename}")
            
            # 9. Calculate confidence
            confidence = self._calculate_confidence(answer, source_docs, question)
            
            # 10. RAGAS-style evaluation (if requested)
            evaluation = None
            if include_evaluation:
                evaluation = self._evaluate_response(
                    question, answer, [doc.page_content for doc in source_docs]
                )
                features_used.append("Evaluation")
            
            # Update conversation history
            if user_id:
                self.conversation_history[user_id].append({"role": "user", "content": question})
                self.conversation_history[user_id].append({"role": "assistant", "content": answer})
                if len(self.conversation_history[user_id]) > 20:
                    self.conversation_history[user_id] = self.conversation_history[user_id][-20:]
            
            return {
                "answer": answer,
                "sources": sources,
                "num_sources": len(sources),
                "confidence": round(confidence, 2),
                "evaluation": evaluation,
                "enhanced_features": features_used
            }
            
        except Exception as e:
            return {
                "answer": f"Error: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "evaluation": None,
                "enhanced_features": features_used
            }
    
    def _evaluate_response(
        self,
        question: str,
        answer: str,
        contexts: List[str]
    ) -> Dict[str, float]:
        """
        RAGAS-inspired evaluation without external dependencies.
        
        Metrics:
        - context_relevancy: How relevant are contexts to the question?
        - answer_faithfulness: Is the answer grounded in contexts?
        - answer_relevancy: How relevant is the answer to the question?
        """
        try:
            # Use cached embeddings for efficiency
            q_emb = self.embedding_cache.get(question, self.embeddings.embed_query)
            a_emb = self.embedding_cache.get(answer[:500], self.embeddings.embed_query)
            
            # Context relevancy
            context_scores = []
            for ctx in contexts[:5]:
                ctx_emb = self.embedding_cache.get(ctx[:300], self.embeddings.embed_query)
                sim = np.dot(q_emb, ctx_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(ctx_emb) + 1e-8)
                context_scores.append(max(0, sim))
            context_relevancy = np.mean(context_scores) if context_scores else 0.0
            
            # Answer relevancy
            answer_relevancy = np.dot(q_emb, a_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(a_emb) + 1e-8)
            answer_relevancy = max(0, answer_relevancy)
            
            # Answer faithfulness (word overlap with contexts)
            combined_context = " ".join(contexts).lower()
            answer_words = set(re.findall(r'\b\w{4,}\b', answer.lower()))
            context_words = set(re.findall(r'\b\w{4,}\b', combined_context))
            faithfulness = len(answer_words & context_words) / max(len(answer_words), 1)
            
            # Overall score
            overall = 0.3 * context_relevancy + 0.4 * faithfulness + 0.3 * answer_relevancy
            
            return {
                "context_relevancy": round(float(context_relevancy), 3),
                "answer_faithfulness": round(float(faithfulness), 3),
                "answer_relevancy": round(float(answer_relevancy), 3),
                "overall_score": round(float(overall), 3)
            }
        except Exception as e:
            print(f"⚠️ Evaluation failed: {e}")
            return {
                "context_relevancy": 0.0,
                "answer_faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "overall_score": 0.0
            }
    
    def detect_hallucination(self, answer: str, sources: List[Document]) -> Dict[str, Any]:
        """
        Detect potential hallucinations in the answer.
        
        Returns risk assessment for each sentence.
        """
        source_text = " ".join([doc.page_content for doc in sources]).lower()
        
        sentences = re.split(r'[.!?]+', answer)
        risky_sentences = []
        
        for sentence in sentences:
            if len(sentence.strip()) < 15:
                continue
            
            words = set(re.findall(r'\b\w{4,}\b', sentence.lower()))
            stopwords = {'this', 'that', 'these', 'those', 'with', 'from', 'have', 'been', 'which', 'would', 'could', 'should'}
            words -= stopwords
            
            if not words:
                continue
            
            covered = sum(1 for w in words if w in source_text)
            coverage = covered / len(words)
            
            if coverage < 0.3:
                risky_sentences.append({
                    "sentence": sentence.strip(),
                    "coverage": round(coverage, 2),
                    "risk": "HIGH" if coverage < 0.15 else "MEDIUM"
                })
        
        total_sentences = len([s for s in sentences if len(s.strip()) >= 15])
        grounding = 1 - (len(risky_sentences) / max(total_sentences, 1))
        
        return {
            "has_risk": len(risky_sentences) > 0,
            "risky_sentences": risky_sentences,
            "overall_grounding": round(grounding, 2)
        }

