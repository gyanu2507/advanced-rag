"""
Advanced Retrieval Strategies for RAG System.

🚀 GREATEST RAG EVER - v2.0

Includes:
- Reciprocal Rank Fusion (RRF): Mathematically optimal fusion of multiple rankings
- HyDE (Hypothetical Document Embeddings): Bridge query-document gap
- Parent-Child Hierarchy: Retrieve children, return parent context
"""
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from langchain.schema import Document
import numpy as np


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[Document, float]]],
    k: int = 60
) -> List[Tuple[Document, float]]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion.
    
    RRF is mathematically proven to be effective for combining multiple
    retrieval strategies. Formula: RRF(d) = Σ 1/(k + rank(d))
    
    Args:
        ranked_lists: List of ranked document lists (each is [(doc, score), ...])
        k: Constant to prevent high-ranked documents from dominating (default 60)
    
    Returns:
        Fused and re-ranked list of (Document, score) tuples
    
    Reference: Cormack, Clarke, Büttcher (2009) - "Reciprocal Rank Fusion 
               outperforms Condorcet and individual Rank Learning Methods"
    """
    doc_scores: Dict[str, float] = defaultdict(float)
    doc_map: Dict[str, Document] = {}
    
    for ranked_list in ranked_lists:
        for rank, (doc, _) in enumerate(ranked_list):
            # Use content prefix as unique identifier
            doc_id = doc.page_content[:150]
            doc_scores[doc_id] += 1.0 / (k + rank + 1)
            doc_map[doc_id] = doc
    
    # Sort by fused score
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_map[doc_id], score) for doc_id, score in sorted_docs]


def hyde_generate_hypothetical(
    query: str,
    llm=None,
    num_hypotheticals: int = 1
) -> List[str]:
    """
    Generate hypothetical document(s) that would answer the query.
    
    HyDE (Hypothetical Document Embeddings) bridges the semantic gap between
    queries and documents by generating a hypothetical answer, then embedding
    that instead of the query.
    
    Args:
        query: The user's question
        llm: Language model for generation (optional)
        num_hypotheticals: Number of hypothetical docs to generate
    
    Returns:
        List of hypothetical document texts
    """
    # Prompt template for generating hypothetical documents
    hyde_prompt = f"""Please write a detailed paragraph that thoroughly answers this question.
Write as if you are writing a section of a knowledge base document.
Be specific and include relevant details.

Question: {query}

Detailed Answer:"""
    
    hypotheticals = []
    
    if llm is not None:
        try:
            for _ in range(num_hypotheticals):
                response = llm.invoke(hyde_prompt)
                if response and len(response.strip()) > 50:
                    hypotheticals.append(response.strip())
        except Exception as e:
            print(f"⚠️ HyDE generation failed: {e}")
    
    # Fallback: enhance query with common answer patterns
    if not hypotheticals:
        # Create pseudo-hypothetical by expanding query
        expanded = _expand_query_as_statement(query)
        hypotheticals = [expanded]
    
    return hypotheticals


def _expand_query_as_statement(query: str) -> str:
    """Convert a question into a statement-like pseudo-document."""
    query = query.strip().rstrip('?')
    
    # Common question-to-statement patterns
    patterns = [
        ('what is', '{subject} is'),
        ('what are', '{subject} are'),
        ('how to', 'To {action}, you should'),
        ('how does', '{subject} works by'),
        ('why is', '{subject} is because'),
        ('why do', 'The reason is that'),
        ('when did', 'It happened when'),
        ('where is', '{subject} is located at'),
        ('who is', '{person} is'),
    ]
    
    query_lower = query.lower()
    for q_pattern, s_pattern in patterns:
        if query_lower.startswith(q_pattern):
            remainder = query[len(q_pattern):].strip()
            return f"{s_pattern.replace('{subject}', remainder).replace('{action}', remainder).replace('{person}', remainder)} {remainder}"
    
    # Default: just return expanded query as context
    return f"Information about: {query}. This relates to {query}."


class ParentChildRetriever:
    """
    Retrieve using child chunks, but return parent context.
    
    This allows for more precise matching (small chunks) while maintaining
    context in the response (large parent chunks).
    """
    
    def __init__(self, chunk_size_child: int = 300, chunk_size_parent: int = 1500):
        self.chunk_size_child = chunk_size_child
        self.chunk_size_parent = chunk_size_parent
        self.parent_store: Dict[str, Document] = {}  # parent_id -> parent_doc
        self.child_to_parent: Dict[str, str] = {}    # child_id -> parent_id
    
    def create_parent_child_chunks(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> Tuple[List[Document], List[Document]]:
        """
        Create parent and child chunks with linking metadata.
        
        Returns:
            Tuple of (parent_documents, child_documents)
        """
        from advanced_chunking import AdvancedChunker
        import hashlib
        
        # Create parent chunks
        parent_chunker = AdvancedChunker(
            chunk_size=self.chunk_size_parent,
            chunk_overlap=200
        )
        parents = parent_chunker.smart_chunk(text, metadata)
        
        # Create child chunks within each parent
        child_chunker = AdvancedChunker(
            chunk_size=self.chunk_size_child,
            chunk_overlap=50
        )
        
        parent_docs = []
        child_docs = []
        
        for parent_idx, parent in enumerate(parents):
            # Generate unique parent ID
            parent_id = hashlib.md5(
                f"{parent.page_content[:100]}_{parent_idx}".encode()
            ).hexdigest()
            
            # Store parent
            parent.metadata["parent_id"] = parent_id
            parent.metadata["is_parent"] = True
            parent_docs.append(parent)
            self.parent_store[parent_id] = parent
            
            # Create children
            children = child_chunker.smart_chunk(
                parent.page_content,
                {
                    **(metadata or {}),
                    "parent_id": parent_id,
                    "is_child": True,
                    "parent_chunk_index": parent_idx
                }
            )
            
            for child_idx, child in enumerate(children):
                child_id = hashlib.md5(
                    f"{child.page_content[:50]}_{parent_idx}_{child_idx}".encode()
                ).hexdigest()
                child.metadata["child_id"] = child_id
                self.child_to_parent[child_id] = parent_id
                child_docs.append(child)
        
        return parent_docs, child_docs
    
    def get_parent_from_child(self, child_doc: Document) -> Optional[Document]:
        """Get the parent document for a child chunk."""
        parent_id = child_doc.metadata.get("parent_id")
        if parent_id and parent_id in self.parent_store:
            return self.parent_store[parent_id]
        return None
    
    def expand_to_parents(self, child_docs: List[Document]) -> List[Document]:
        """Expand child documents to their parents, deduplicating."""
        seen_parents = set()
        parent_docs = []
        
        for child in child_docs:
            parent = self.get_parent_from_child(child)
            if parent:
                parent_id = parent.metadata.get("parent_id")
                if parent_id not in seen_parents:
                    seen_parents.add(parent_id)
                    parent_docs.append(parent)
            else:
                # Child without parent - include as is
                parent_docs.append(child)
        
        return parent_docs


def contextual_compression(
    query: str,
    documents: List[Document],
    embeddings,
    similarity_threshold: float = 0.3
) -> List[Document]:
    """
    Compress documents to only include query-relevant sentences.
    
    This reduces context length while keeping the most relevant information.
    
    Args:
        query: The search query
        documents: Documents to compress
        embeddings: Embedding model
        similarity_threshold: Minimum similarity for sentence inclusion
    
    Returns:
        List of compressed documents
    """
    import re
    
    query_embedding = np.array(embeddings.embed_query(query))
    compressed_docs = []
    
    for doc in documents:
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', doc.page_content)
        
        # Score each sentence
        relevant_sentences = []
        for sentence in sentences:
            if len(sentence.strip()) < 20:
                continue
            
            try:
                sent_embedding = np.array(embeddings.embed_query(sentence))
                similarity = np.dot(query_embedding, sent_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(sent_embedding)
                )
                
                if similarity >= similarity_threshold:
                    relevant_sentences.append(sentence)
            except:
                # Include sentence if embedding fails
                relevant_sentences.append(sentence)
        
        # Create compressed document
        if relevant_sentences:
            compressed_content = " ".join(relevant_sentences)
            compressed_doc = Document(
                page_content=compressed_content,
                metadata={
                    **doc.metadata,
                    "compressed": True,
                    "original_length": len(doc.page_content),
                    "compressed_length": len(compressed_content)
                }
            )
            compressed_docs.append(compressed_doc)
    
    return compressed_docs
