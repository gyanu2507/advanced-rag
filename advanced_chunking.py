"""
Advanced text chunking strategies for better RAG performance.
"""
import re
from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


class AdvancedChunker:
    """Advanced chunking with multiple strategies."""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None
    ):
        """Initialize advanced chunker.
        
        Args:
            chunk_size: Target size for chunks
            chunk_overlap: Overlap between chunks for context preservation
            separators: Custom separators for splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Default separators (ordered by priority)
        if separators is None:
            self.separators = [
                "\n\n",  # Paragraphs
                "\n",    # Lines
                ". ",    # Sentences
                "! ",    # Exclamations
                "? ",    # Questions
                "; ",    # Semicolons
                ", ",    # Commas
                " ",     # Words
                ""       # Characters
            ]
        else:
            self.separators = separators
    
    def semantic_chunk(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> List[Document]:
        """Create semantic chunks that preserve meaning.
        
        Uses recursive splitting with smart separators to maintain
        semantic coherence.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False
        )
        
        chunks = splitter.split_text(text)
        documents = []
        
        for i, chunk in enumerate(chunks):
            doc_metadata = metadata.copy() if metadata else {}
            doc_metadata.update({
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk)
            })
            documents.append(Document(page_content=chunk, metadata=doc_metadata))
        
        return documents
    
    def sentence_aware_chunk(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> List[Document]:
        """Chunk text while preserving sentence boundaries."""
        # Split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_size = len(sentence)
            
            # If adding this sentence would exceed chunk size
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-self._get_overlap_sentence_count():]
                current_chunk = overlap_sentences + [sentence]
                current_size = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        documents = []
        for i, chunk in enumerate(chunks):
            doc_metadata = metadata.copy() if metadata else {}
            doc_metadata.update({
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk),
                "chunking_strategy": "sentence_aware"
            })
            documents.append(Document(page_content=chunk, metadata=doc_metadata))
        
        return documents
    
    def _get_overlap_sentence_count(self) -> int:
        """Calculate how many sentences to include in overlap."""
        # Rough estimate: overlap should be about 20% of chunk
        estimated_sentences_per_chunk = self.chunk_size // 50  # ~50 chars per sentence
        return max(1, int(estimated_sentences_per_chunk * 0.2))
    
    def paragraph_chunk(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> List[Document]:
        """Chunk by paragraphs, combining small ones."""
        paragraphs = text.split("\n\n")
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_size = len(para)
            
            # If paragraph is larger than chunk size, split it
            if para_size > self.chunk_size:
                # Save current chunk if exists
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                # Split large paragraph using recursive method
                split_para = self.semantic_chunk(para, metadata)
                chunks.extend([doc.page_content for doc in split_para])
            elif current_size + para_size > self.chunk_size:
                # Save current chunk
                chunks.append("\n\n".join(current_chunk))
                
                # Start new chunk with overlap
                overlap_paras = current_chunk[-1:] if current_chunk else []
                current_chunk = overlap_paras + [para]
                current_size = sum(len(p) for p in current_chunk)
            else:
                current_chunk.append(para)
                current_size += para_size
        
        # Add remaining chunk
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        
        documents = []
        for i, chunk in enumerate(chunks):
            doc_metadata = metadata.copy() if metadata else {}
            doc_metadata.update({
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk),
                "chunking_strategy": "paragraph"
            })
            documents.append(Document(page_content=chunk, metadata=doc_metadata))
        
        return documents
    
    def smart_chunk(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> List[Document]:
        """Smart chunking that chooses the best strategy based on text structure."""
        # Analyze text structure
        has_paragraphs = "\n\n" in text
        avg_sentence_length = self._get_avg_sentence_length(text)
        
        # Choose strategy
        if has_paragraphs and len(text) > self.chunk_size * 2:
            # Use paragraph chunking for structured text
            return self.paragraph_chunk(text, metadata)
        elif avg_sentence_length < 100:
            # Use sentence-aware for short sentences
            return self.sentence_aware_chunk(text, metadata)
        else:
            # Default to semantic chunking
            return self.semantic_chunk(text, metadata)
    
    def _get_avg_sentence_length(self, text: str) -> float:
        """Calculate average sentence length."""
        sentences = re.split(r'[.!?]+\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0
        return sum(len(s) for s in sentences) / len(sentences)
