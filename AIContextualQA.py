# -*- coding: utf-8 -*-
"""
Real AI-Powered Persian QA System
Uses:
1. Sentence Transformers (Neural Network Embeddings) - REAL AI
2. Ollama (Local LLM) - REAL AI for answer generation
3. FAISS for efficient similarity search
"""

from hazm import Normalizer, sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import json
import re


class AIContextualQA:
    """
    Real AI-powered QA system using:
    - Sentence-BERT for semantic embeddings (110M parameters neural network)
    - Ollama for intelligent answer generation (7B+ parameters LLM)
    """
    
    def __init__(self, 
                 embedding_model='paraphrase-multilingual-MiniLM-L12-v2',
                 ollama_model='llama2',
                 ollama_url='http://localhost:11434',
                 context_window=2):
        """
        Args:
            embedding_model: Sentence transformer model (neural network)
            ollama_model: Ollama model name (llama2, mistral, etc.)
            ollama_url: Ollama API URL
            context_window: Sentences before/after to include
        """
        print("🚀 Initializing AI-Powered QA System...")
        print("=" * 60)
        
        # Hazm for Persian preprocessing
        self.normalizer = Normalizer()
        
        # REAL AI: Sentence Transformer (Neural Network with 110M parameters!)
        print(f"📥 Loading AI model: {embedding_model}")
        print("   This is a REAL neural network with millions of parameters!")
        self.embedding_model = SentenceTransformer(embedding_model)
        print("   ✓ Neural network loaded successfully")
        
        # Ollama settings
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.use_ollama = self._check_ollama_available()
        
        # Storage
        self.sentences = []
        self.sentence_metadata = []
        self.context_window = context_window
        
        # AI embeddings (vectors from neural network)
        self.embeddings = None
        self.faiss_index = None
        
        print("=" * 60)
        print("✓ AI System initialized")
        print(f"  - Embedding model: {embedding_model} (Neural Network)")
        print(f"  - Ollama LLM: {ollama_model} ({'✓ Available' if self.use_ollama else '✗ Not running'})")
        print(f"  - Context window: ±{context_window} sentences")
        print("=" * 60)
    
    def _check_ollama_available(self):
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            if response.status_code == 200:
                print("   ✓ Ollama LLM is available (Local AI)")
                return True
        except:
            pass
        
        print("   ⚠ Ollama not running (will use simple answers)")
        print("   Start Ollama with: ollama serve")
        return False
    
    def train(self, paragraphs, paragraph_names=None):
        """
        Train using REAL AI embeddings
        """
        print(f"\n🧠 Training with REAL AI on {len(paragraphs)} paragraphs...")
        print("=" * 60)
        
        if paragraph_names is None:
            paragraph_names = [f"Para_{i+1}" for i in range(len(paragraphs))]
        
        # Extract sentences with context
        all_sentences = []
        all_contexts = []
        
        for para_idx, (paragraph, para_name) in enumerate(zip(paragraphs, paragraph_names)):
            # Normalize
            normalized = self.normalizer.normalize(paragraph)
            sentences = sent_tokenize(normalized)
            
            # Process each sentence
            para_sentences = []
            for sent_idx, sentence in enumerate(sentences):
                if len(word_tokenize(sentence)) < 3:
                    continue
                
                para_sentences.append({
                    'text': sentence,
                    'para_name': para_name,
                    'para_idx': para_idx,
                    'sent_idx': sent_idx
                })
            
            # Build context for each sentence
            for i, sent_data in enumerate(para_sentences):
                # Get surrounding sentences
                start = max(0, i - self.context_window)
                end = min(len(para_sentences), i + self.context_window + 1)
                
                context_sentences = [para_sentences[j]['text'] for j in range(start, end)]
                context_text = ' '.join(context_sentences)
                
                # Store
                self.sentences.append(sent_data['text'])
                all_contexts.append(context_text)
                
                # Metadata
                self.sentence_metadata.append({
                    'paragraph_name': para_name,
                    'paragraph_idx': para_idx,
                    'sentence_idx': sent_idx,
                    'original': sent_data['text'],
                    'local_idx': i,
                    'paragraph_sentences': para_sentences
                })
        
        print(f"✓ Extracted {len(self.sentences)} sentences")
        
        # REAL AI: Generate embeddings using neural network
        print("\n🤖 Generating AI embeddings (neural network inference)...")
        print("   This uses a trained neural network with 110M parameters!")
        print("   Each sentence becomes a 384-dimensional semantic vector")
        
        # Generate embeddings for contexts (REAL AI!)
        self.embeddings = self.embedding_model.encode(
            all_contexts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # For cosine similarity
        )
        
        print(f"✓ Generated {self.embeddings.shape[0]} embeddings")
        print(f"  Shape: {self.embeddings.shape} (sentences × dimensions)")
        
        # Build FAISS index for efficient search
        print("\n📊 Building FAISS index for fast similarity search...")
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product (cosine)
        self.faiss_index.add(self.embeddings.astype('float32'))
        
        print("=" * 60)
        print("✓ Training complete!")
        print(f"  Total knowledge: {len(self.sentences)} sentences")
        print(f"  Vector dimension: {dimension}D")
        print("=" * 60)
    
    def find_answer(self, question, top_k=5, use_ollama_for_answer=True):
        """
        Find answer using REAL AI
        """
        if self.faiss_index is None:
            return {
                'answer': "سیستم هنوز آموزش نداده است!",
                'confidence': 0.0
            }
        
        print(f"\n" + "=" * 60)
        print(f"🔍 Question: {question}")
        print("=" * 60)
        
        # Normalize question
        normalized_question = self.normalizer.normalize(question)
        
        # REAL AI: Generate question embedding using neural network
        print("🤖 Generating question embedding (neural network)...")
        question_embedding = self.embedding_model.encode(
            [normalized_question],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Search with FAISS (efficient similarity search)
        print(f"🔎 Searching in {len(self.sentences)} sentences...")
        k = min(top_k, len(self.sentences))
        similarities, indices = self.faiss_index.search(
            question_embedding.astype('float32'),
            k
        )
        
        # Get top candidates
        candidates = []
        print(f"\n📋 Top {k} most relevant sentences:")
        for i, (idx, score) in enumerate(zip(indices[0], similarities[0]), 1):
            sentence = self.sentences[idx]
            candidates.append({
                'text': sentence,
                'score': float(score),
                'index': idx,
                'metadata': self.sentence_metadata[idx]
            })
            print(f"   {i}. [{score:.3f}] {sentence[:70]}...")
        
        # Build multi-sentence context
        answer_indices = self._build_multi_sentence_answer(
            candidates,
            question
        )
        
        # Collect answer sentences
        answer_sentences = [self.sentences[idx] for idx in answer_indices]
        context = ' '.join(answer_sentences)
        
        print(f"\n📝 Selected {len(answer_indices)} sentences for answer")
        
        # Generate intelligent answer
        if use_ollama_for_answer and self.use_ollama:
            print("🧠 Generating intelligent answer with Ollama LLM...")
            answer = self._generate_answer_with_ollama(
                question,
                context,
                answer_sentences
            )
        else:
            print("📄 Using context-based answer (no LLM)")
            answer = context
        
        print("=" * 60)
        
        return {
            'answer': answer,
            'confidence': float(similarities[0][0]),
            'context_sentences': answer_sentences,
            'sources': [
                {
                    'text': sent,
                    'paragraph': self.sentence_metadata[idx]['paragraph_name'],
                    'score': float(similarities[0][list(indices[0]).index(idx)])
                }
                for idx, sent in zip(answer_indices, answer_sentences)
            ]
        }
    
    def _build_multi_sentence_answer(self, candidates, question):
        """Build multi-sentence answer intelligently"""
        answer_indices = [candidates[0]['index']]  # Start with best match
        best_meta = candidates[0]['metadata']
        
        # Add adjacent sentences from same paragraph if relevant
        for candidate in candidates[1:]:
            idx = candidate['index']
            meta = candidate['metadata']
            
            # Same paragraph?
            if meta['paragraph_idx'] == best_meta['paragraph_idx']:
                # Adjacent?
                if abs(meta['local_idx'] - best_meta['local_idx']) <= 2:
                    # High enough score?
                    if candidate['score'] > 0.3:
                        answer_indices.append(idx)
        
        # Sort by document order
        answer_indices.sort(key=lambda idx: (
            self.sentence_metadata[idx]['paragraph_idx'],
            self.sentence_metadata[idx]['sentence_idx']
        ))
        
        return answer_indices
    
    def _generate_answer_with_ollama(self, question, context, sentences):
        """
        Use Ollama (REAL AI LLM) to generate intelligent answer
        """
        try:
            # Build prompt for LLM
            prompt = f"""You are a helpful assistant answering questions in Persian.

Question: {question}

Context from documents:
{context}

Instructions:
- Answer the question in Persian based ONLY on the provided context
- Be concise but complete
- If the context doesn't contain enough information, say so
- Do not make up information

Answer in Persian:"""

            # Call Ollama API (Local LLM with billions of parameters!)
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Lower = more factual
                        "num_predict": 200   # Max tokens
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '').strip()
                
                if answer:
                    print("   ✓ LLM generated intelligent answer")
                    return answer
            
        except Exception as e:
            print(f"   ⚠ Ollama error: {e}")
        
        # Fallback
        return context
    
    def compare_with_tfidf(self, question):
        """
        Compare AI embeddings vs TF-IDF (for demonstration)
        """
        print("\n" + "="*60)
        print("📊 COMPARISON: AI vs Classical ML")
        print("="*60)
        
        # AI approach
        print("\n1️⃣ AI Approach (Neural Network):")
        ai_result = self.find_answer(question, use_ollama_for_answer=False)
        print(f"   Answer: {ai_result['answer'][:100]}...")
        print(f"   Confidence: {ai_result['confidence']:.3f}")
        
        # For comparison, you'd need TF-IDF here
        print("\n2️⃣ Classical ML (TF-IDF) would:")
        print("   - Just count word frequencies")
        print("   - Miss semantic meaning")
        print("   - Can't understand 'similar' words")
        
        print("\n✅ AI is better because:")
        print("   - Understands meaning, not just keywords")
        print("   - Handles synonyms and paraphrasing")
        print("   - Captures semantic relationships")
        print("="*60)


def run_ai_qa_demo():
    """Demo of real AI-powered QA"""
    print("\n" + "="*70)
    print("🤖 REAL AI-POWERED PERSIAN QA SYSTEM")
    print("="*70)
    print("\nThis system uses ACTUAL neural networks:")
    print("  • Sentence Transformers: 110M parameter neural network")
    print("  • Ollama LLM: 7B+ parameter language model")
    print("  • FAISS: Efficient similarity search")
    print("="*70)
    
    # Initialize
    qa = AIContextualQA(
        embedding_model='paraphrase-multilingual-MiniLM-L12-v2',
        ollama_model='llama2',
        context_window=2
    )
    
    # Training data
    training_data = [
        """
        دانشگاه تهران در سال ۱۳۱۳ تاسیس شد. این دانشگاه اولین دانشگاه مدرن ایران است.
        دانشگاه تهران در شهر تهران واقع شده است. این دانشگاه در خیابان انقلاب قرار دارد.
        دانشگاه تهران دارای دانشکده های مهندسی، پزشکی و علوم انسانی است. 
        دلیل معروف بودن این دانشگاه کیفیت بالای آموزشی آن است. 
        تعداد دانشجویان این دانشگاه حدود ۵۰,۰۰۰ نفر است.
        """,
        """
        شرکت گوگل در سال ۱۹۹۸ توسط لری پیج و سرگی برین تاسیس شد.
        دفتر مرکزی این شرکت در Mountain View کالیفرنیا قرار دارد.
        گوگل به دلیل موتور جستجوی قدرتمند خود مشهور است.
        محصولات اصلی گوگل شامل اندروید، یوتیوب و جیمیل می باشد.
        """,
        """
        هوش مصنوعی مجموعه ای از تکنیک های کامپیوتری است که رفتار هوشمندانه را تقلید می کند.
        این سیستم ها می توانند از تجربیات یاد بگیرند و تصمیمات پیچیده بگیرند.
        کاربردهای هوش مصنوعی شامل تشخیص تصویر، پردازش زبان و ربات ها است.
        """
    ]
    
    paragraph_names = ["دانشگاه تهران", "گوگل", "هوش مصنوعی"]
    
    # Train
    qa.train(training_data, paragraph_names)
    
    # Test questions
    test_questions = [
        "دانشگاه تهران کجا واقع شده و چه دانشکده هایی دارد؟",
        "چرا دانشگاه تهران معروف است؟",
        "گوگل توسط چه کسانی ساخته شد؟",
        "محصولات گوگل کدامند؟",
        "هوش مصنوعی چیست و چه کاربردهایی دارد؟",
    ]
    
    # Answer questions
    for i, question in enumerate(test_questions, 1):
        result = qa.find_answer(question, top_k=5, use_ollama_for_answer=True)
        
        print(f"\n{'='*70}")
        print(f"✅ FINAL ANSWER:")
        print(f"   {result['answer']}")
        print(f"\n📍 Sources:")
        for j, source in enumerate(result['sources'], 1):
            print(f"   {j}. {source['paragraph']} (score: {source['score']:.3f})")
        print("="*70)
        
        if i < len(test_questions):
            input("\n⏸  Press Enter for next question...")


if __name__ == "__main__":
    run_ai_qa_demo()

