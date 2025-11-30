# RAG System Enhancements

This document describes the new enhancements added to the RAG (Retrieval Augmented Generation) system.

## 🚀 New Features

### 1. **Query Rewriting & Expansion**
- Automatically rewrites and expands user queries for better retrieval
- Incorporates conversation context for follow-up questions
- Generates query variations to improve search coverage

### 2. **Hybrid Search**
- Combines semantic (vector) search with keyword-based search
- 70% weight for semantic similarity, 30% for keyword matching
- Provides more comprehensive and accurate results

### 3. **Multi-Query Generation**
- Generates multiple query variations from the original question
- Searches with all variations and combines results
- Improves recall by finding relevant documents that might be missed with a single query

### 4. **Document Reranking**
- Reranks retrieved documents using multiple heuristics:
  - Keyword overlap with query
  - Term frequency in documents
  - Document position (earlier chunks may be more important)
  - Document length (optimal range preferred)
- Ensures the most relevant documents are prioritized

### 5. **Confidence Scoring**
- Calculates confidence scores for answers (0.0 to 1.0)
- Based on:
  - Number of supporting sources
  - Answer length and quality
  - Query-answer relevance
  - Source metadata quality
- Helps users understand answer reliability

### 6. **Conversation Memory**
- Maintains conversation history per user
- Uses context from previous exchanges to improve follow-up questions
- Automatically manages memory (keeps last 10 exchanges)

## 📊 Enhanced Response Format

The query response now includes additional fields:

```json
{
  "answer": "The answer text...",
  "sources": ["source1", "source2", ...],
  "num_sources": 3,
  "confidence": 0.85,
  "enhanced": true
}
```

## 🎯 Usage

All enhancements are enabled by default. The system automatically:
- Rewrites queries with conversation context
- Uses hybrid search for better retrieval
- Reranks results for relevance
- Calculates confidence scores
- Maintains conversation history

## 🔧 Technical Details

### Query Rewriting
- Analyzes question structure and adds context from conversation history
- Expands question words (what, how, why, etc.) with related terms

### Hybrid Search
- Semantic search: Uses embedding similarity (70% weight)
- Keyword search: Uses term overlap matching (30% weight)
- Combines scores for final ranking

### Reranking Algorithm
- Keyword overlap: 40% weight
- Term frequency: 30% weight
- Position bonus: 20% weight
- Length bonus: 10% weight

### Confidence Calculation
- Source count: 30% weight
- Answer length: 20% weight
- Query-answer relevance: 30% weight
- Source quality: 20% weight

## 💡 Benefits

1. **Better Retrieval**: Hybrid search finds more relevant documents
2. **Improved Accuracy**: Reranking ensures best documents are used
3. **Context Awareness**: Conversation memory enables natural follow-ups
4. **Transparency**: Confidence scores help users assess answer quality
5. **Robustness**: Multiple query variations improve recall

## 🎨 UI Enhancements

The frontend now displays:
- Confidence score with color coding (green/yellow/red)
- "Enhanced Search" badge when enhancements are active
- Updated feature badges showing new capabilities

## 🔄 Backward Compatibility

All enhancements are backward compatible. The system works with or without enhancements enabled. You can disable enhancements by setting `use_enhancements=False` in the query method.

