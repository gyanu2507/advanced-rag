"""
RAG (Retrieval Augmented Generation) system implementation using FREE APIs.
Enhanced with query rewriting, hybrid search, reranking, and conversation memory.
"""
import os
import re
from typing import List, Optional, Dict, Tuple
from collections import defaultdict
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
        
        # Conversation memory for context-aware queries
        self.conversation_history: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        
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
            results = [(doc, score) for doc, score in results 
                      if doc.metadata.get("document_id") in document_ids]
        
        # Sort by combined score and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def _rerank_documents(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: int = 5
    ) -> List[Document]:
        """Rerank documents using simple heuristics."""
        if not documents:
            return documents
        
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        
        scored_docs = []
        for doc in documents:
            doc_text = doc.page_content.lower()
            doc_words = set(re.findall(r'\b\w+\b', doc_text))
            
            # Calculate relevance score
            score = 0.0
            
            # 1. Keyword overlap
            if query_words:
                overlap = len(query_words & doc_words) / len(query_words)
                score += overlap * 0.4
            
            # 2. Query term frequency in document
            term_freq = sum(doc_text.count(word) for word in query_words)
            score += min(term_freq / 10.0, 0.3)  # Cap at 0.3
            
            # 3. Position bonus (earlier chunks might be more important)
            if doc.metadata and 'chunk_index' in doc.metadata:
                chunk_idx = doc.metadata.get('chunk_index', 0)
                total_chunks = doc.metadata.get('total_chunks', 1)
                if total_chunks > 0:
                    position_score = 1.0 - (chunk_idx / max(total_chunks, 1))
                    score += position_score * 0.2
            
            # 4. Length bonus (not too short, not too long)
            doc_length = len(doc.page_content)
            if 100 <= doc_length <= 1000:
                score += 0.1
            
            scored_docs.append((doc, score))
        
        # Sort by score and return top k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:top_k]]
    
    def _calculate_confidence(
        self, 
        answer: str, 
        sources: List[Document], 
        query: str
    ) -> float:
        """Calculate confidence score for the answer."""
        if not answer or not sources:
            return 0.0
        
        confidence = 0.0
        
        # 1. Source count (more sources = higher confidence)
        source_count_score = min(len(sources) / 5.0, 1.0) * 0.3
        confidence += source_count_score
        
        # 2. Answer length (reasonable length = higher confidence)
        answer_length = len(answer)
        if 50 <= answer_length <= 500:
            confidence += 0.2
        elif answer_length > 500:
            confidence += 0.1
        
        # 3. Query-answer relevance (simple keyword overlap)
        query_words = set(re.findall(r'\b\w{4,}\b', query.lower()))
        answer_words = set(re.findall(r'\b\w{4,}\b', answer.lower()))
        if query_words:
            relevance = len(query_words & answer_words) / len(query_words)
            confidence += relevance * 0.3
        
        # 4. Source quality (check if sources have good metadata)
        quality_score = 0.0
        for source in sources:
            if source.metadata:
                quality_score += 0.1
        confidence += min(quality_score, 0.2)
        
        return min(confidence, 1.0)
    
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
                "confidence": 0.0
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
                
                # 4. Rerank results
                docs = [doc for doc, _ in all_results[:15]]  # Get top 15 for reranking
                docs = self._rerank_documents(enhanced_question, docs, top_k=5)
            else:
                # Standard search (fallback)
                if document_ids and len(document_ids) > 0:
                all_docs = self.vectorstore.similarity_search(question, k=20)
                docs = [doc for doc in all_docs if doc.metadata.get("document_id") in document_ids][:5]
            else:
                docs = self.vectorstore.similarity_search(question, k=5)
            
            if not docs:
                if document_ids:
                    return {
                        "answer": "No relevant information found in the selected document(s).",
                        "sources": [],
                        "confidence": 0.0
                    }
                return {
                    "answer": "No relevant information found in the documents.",
                    "sources": [],
                    "confidence": 0.0
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
            
            # Enhanced retrieval-only mode with better formatting
            context_parts = []
            for i, doc in enumerate(docs, 1):
                chunk_info = f"[Source {i}]"
                if doc.metadata:
                    filename = doc.metadata.get('filename', 'Unknown')
                    chunk_idx = doc.metadata.get('chunk_index', '?')
                    total_chunks = doc.metadata.get('total_chunks', '?')
                    chunk_info += f" (File: {filename}, Chunk {chunk_idx}/{total_chunks})"
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
                    filename = doc.metadata.get('filename', 'Unknown')
                    chunk_idx = doc.metadata.get('chunk_index', 'N/A')
                    total_chunks = doc.metadata.get('total_chunks', 'N/A')
                    source_text += f"\n[File: {filename}, Chunk {chunk_idx}/{total_chunks}]"
                sources.append(source_text)
            
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
                "confidence": 0.0
            }
    
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

