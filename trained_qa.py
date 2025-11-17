# -*- coding: utf-8 -*-
"""
Trained Persian QA System using Hazm + TF-IDF
This system actually learns from the data using machine learning
"""

from hazm import Normalizer, sent_tokenize, word_tokenize, Stemmer, stopwords_list
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re


class TrainedPersianQA:
    """
    A QA system that learns from Persian text using:
    - Hazm for preprocessing
    - TF-IDF for feature extraction
    - Cosine similarity for matching
    """
    
    def __init__(self):
        """Initialize components"""
        print("🚀 Initializing Trained Persian QA System...")
        
        # Hazm components
        self.normalizer = Normalizer()
        self.stemmer = Stemmer()
        
        # Get Persian stop words properly
        self.stop_words = set(stopwords_list())
        
        # Storage for training data
        self.sentences = []  # Original sentences
        self.processed_sentences = []  # Preprocessed sentences
        self.sentence_metadata = []  # Metadata (paragraph source, etc.)
        
        # TF-IDF vectorizer (will be trained)
        self.vectorizer = None
        self.sentence_vectors = None
        
        print("✓ System initialized")

    def preprocess_text(self, text, remove_stopwords=True):
        """
        Preprocess Persian text with Hazm
        
        Args:
            text: Raw Persian text
            remove_stopwords: Whether to remove stop words
            
        Returns:
            Preprocessed text string
        """
        # Normalize
        text = self.normalizer.normalize(text)
        
        # Tokenize
        words = word_tokenize(text)
        
        # Stem and filter
        processed_words = []
        for word in words:
            # Skip punctuation and numbers
            if re.match(r'^[\W\d]+$', word):
                continue
            
            # Stem the word
            stemmed = self.stemmer.stem(word)
            
            # Remove stop words if requested
            if remove_stopwords and stemmed in self.stop_words:
                continue
            
            processed_words.append(stemmed)
        
        return ' '.join(processed_words)
    
    def train(self, paragraphs, paragraph_names=None):
        """
        Train the QA system on a dataset of paragraphs
        
        Args:
            paragraphs: List of Persian text paragraphs
            paragraph_names: Optional names/IDs for paragraphs
        """
        print(f"\n📚 Training on {len(paragraphs)} paragraphs...")
        
        if paragraph_names is None:
            paragraph_names = [f"Paragraph_{i+1}" for i in range(len(paragraphs))]
        
        # Process each paragraph
        for para_idx, (paragraph, para_name) in enumerate(zip(paragraphs, paragraph_names)):
            # Normalize paragraph
            normalized = self.normalizer.normalize(paragraph)
            
            # Split into sentences
            sentences = sent_tokenize(normalized)
            
            # Process each sentence
            for sent_idx, sentence in enumerate(sentences):
                # Skip very short sentences
                if len(word_tokenize(sentence)) < 3:
                    continue
                
                # Store original sentence
                self.sentences.append(sentence)
                
                # Preprocess and store
                processed = self.preprocess_text(sentence, remove_stopwords=True)
                self.processed_sentences.append(processed)
                
                # Store metadata
                self.sentence_metadata.append({
                    'paragraph_name': para_name,
                    'paragraph_idx': para_idx,
                    'sentence_idx': sent_idx,
                    'original': sentence
                })
        
        print(f"✓ Extracted {len(self.sentences)} sentences")
        
        # Train TF-IDF vectorizer
        print("🧠 Training TF-IDF vectorizer...")
        self.vectorizer = TfidfVectorizer(
            max_features=1000,  # Keep top 1000 features
            ngram_range=(1, 2),  # Use unigrams and bigrams
            min_df=1,  # Minimum document frequency
        )
        
        # Fit and transform all sentences
        self.sentence_vectors = self.vectorizer.fit_transform(self.processed_sentences)
        
        print(f"✓ Trained on {len(self.vectorizer.get_feature_names_out())} features")
        print("✓ Training complete!")
        
        # Show learned vocabulary sample
        vocab = self.vectorizer.get_feature_names_out()
        print(f"   Sample vocabulary: {list(vocab[:10])}")
    
    def find_answer(self, question, top_k=4, threshold=0.1):
        """
        Find the best answer to a question
        
        Args:
            question: Persian question
            top_k: Number of candidate sentences to consider
            threshold: Minimum similarity threshold
            
        Returns:
            Dict with answer and metadata
        """
        if not self.vectorizer:
            return {
                'answer': "سیستم هنوز آموزش نداده است!",
                'confidence': 0.0,
                'source': None
            }
        
        print(f"\n🔍 Question: {question}")
        
        # Preprocess question
        processed_question = self.preprocess_text(question, remove_stopwords=True)
        print(f"   Processed: {processed_question}")
        
        # Convert question to vector
        question_vector = self.vectorizer.transform([processed_question])
        
        # Calculate similarity with all sentences
        similarities = cosine_similarity(question_vector, self.sentence_vectors)[0]
        
        # Get top-k most similar sentences
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_scores = similarities[top_indices]
        
        print(f"\n   Top {top_k} candidates:")
        for idx, score in zip(top_indices, top_scores):
            print(f"   [{score:.3f}] {self.sentences[idx][:60]}...")
        
        # Check if best match is above threshold
        best_idx = top_indices[0]
        best_score = top_scores[0]
        
        if best_score < threshold:
            return {
                'answer': "پاسخ دقیقی یافت نشد.",
                'confidence': best_score,
                'source': None
            }
        
        # Apply question-type specific logic
        answer_idx = self._select_best_answer(
            question, 
            top_indices, 
            top_scores,
            processed_question
        )
        
        return {
            'answer': self.sentences[answer_idx],
            'confidence': similarities[answer_idx],
            'source': self.sentence_metadata[answer_idx],
            'all_candidates': [
                {
                    'sentence': self.sentences[idx],
                    'score': float(similarities[idx])
                }
                for idx in top_indices
            ]
        }
    
    def _select_best_answer(self, question, candidate_indices, scores, processed_question):
        """
        Select the best answer using question-type heuristics
        
        Args:
            question: Original question
            candidate_indices: Indices of candidate sentences
            scores: Similarity scores
            processed_question: Preprocessed question
            
        Returns:
            Index of best answer
        """
        question_lower = question.lower()
        
        # Question type patterns
        why_patterns = ['چرا', 'به چه دلیل', 'علت', 'دلیل']
        when_patterns = ['کی', 'چه زمانی', 'چه سالی', 'کدام سال']
        where_patterns = ['کجا', 'کدام مکان', 'کدام شهر', 'در کجا']
        who_patterns = ['چه کسی', 'چه کسانی', 'بنیانگذار', 'موسس']
        what_patterns = ['چیست', 'چه چیزی', 'چگونه']
        
        question_type = None
        if any(p in question_lower for p in why_patterns):
            question_type = 'why'
        elif any(p in question_lower for p in when_patterns):
            question_type = 'when'
        elif any(p in question_lower for p in where_patterns):
            question_type = 'where'
        elif any(p in question_lower for p in who_patterns):
            question_type = 'who'
        elif any(p in question_lower for p in what_patterns):
            question_type = 'what'
        
        print(f"   Detected question type: {question_type}")
        
        # Boost scores based on question type
        boosted_scores = scores.copy()
        
        for i, idx in enumerate(candidate_indices):
            sentence = self.sentences[idx].lower()
            
            if question_type == 'why':
                # Look for reason indicators
                if any(word in sentence for word in ['دلیل', 'زیرا', 'چون', 'به دلیل', 'به خاطر']):
                    boosted_scores[i] += 0.2
                    print(f"   Boosted (why): {self.sentences[idx][:40]}...")
            
            elif question_type == 'when':
                # Look for years (4-digit numbers)
                if re.search(r'\d{4}', sentence):
                    boosted_scores[i] += 0.2
                    print(f"   Boosted (when): {self.sentences[idx][:40]}...")
                # Look for time words
                if any(word in sentence for word in ['سال', 'زمان', 'تاریخ']):
                    boosted_scores[i] += 0.1
            
            elif question_type == 'where':
                # Look for location words
                if any(word in sentence for word in ['شهر', 'مکان', 'واقع', 'قرار دارد']):
                    boosted_scores[i] += 0.2
                    print(f"   Boosted (where): {self.sentences[idx][:40]}...")
            
            elif question_type == 'who':
                # Look for person names (capitalized words)
                if re.search(r'[A-Z][a-z]+', sentence):
                    boosted_scores[i] += 0.15
                # Look for توسط
                if 'توسط' in sentence:
                    boosted_scores[i] += 0.2
                    print(f"   Boosted (who): {self.sentences[idx][:40]}...")
        
        # Return index with best boosted score
        best_idx_in_candidates = np.argmax(boosted_scores)
        return candidate_indices[best_idx_in_candidates]
    
    def get_stats(self):
        """Get statistics about the trained model"""
        return {
            'total_sentences': len(self.sentences),
            'vocabulary_size': len(self.vectorizer.get_feature_names_out()) if self.vectorizer else 0,
            'trained': self.vectorizer is not None,
            'stop_words_count': len(self.stop_words)
        }


def run_trained_qa_demo():
    """
    Demonstration of the trained QA system
    """
    print("=" * 80)
    print("🎓 TRAINED PERSIAN QA SYSTEM DEMO")
    print("=" * 80)
    
    # Initialize system
    qa = TrainedPersianQA()
    
    # Training dataset
    training_data = [
        """
        دانشگاه تهران در سال ۱۳۱۳ تاسیس شد. این دانشگاه در شهر تهران واقع شده است. 
        دانشگاه تهران دارای دانشکده های مهندسی، پزشکی و علوم انسانی است. 
        دلیل معروف بودن این دانشگاه کیفیت بالای آموزشی آن است. 
        تعداد دانشجویان این دانشگاه حدود ۵۰,۰۰۰ نفر است. 
        رشته کامپیوتر در دانشکده مهندسی این دانشگاه تدریس می شود.
        """,
        """
        شرکت گوگل در سال ۱۹۹۸ توسط لری پیج و سرگی برین تاسیس شد.
        دفتر مرکزی این شرکت در Mountain View کالیفرنیا قرار دارد.
        گوگل به دلیل موتور جستجوی قدرتمند خود مشهور است.
        این شرکت در سال ۲۰۰۴ به صورت عمومی عرضه شد.
        محصولات اصلی گوگل شامل اندروید، یوتیوب و جیمیل می باشد.
        """,
        """
        هوش مصنوعی به مجموعه ای از تکنیک ها و سیستم های کامپیوتری گفته می شود که هدف آن ها تقلید و شبیه سازی رفتارهای هوشمندانه انسان ها است.
        این سیستم ها می توانند اطلاعات را پردازش کنند، از تجربیات خود یاد بگیرند و حتا تصمیمات پیچیده اتخاذ کنند.
        """
    ]
    
    paragraph_names = ["دانشگاه تهران", "گوگل", "هوش مصنوعي"]
    
    # Train the system
    qa.train(training_data, paragraph_names)
    
    # Test questions
    test_questions = [
        # About Tehran University
        "دانشگاه تهران در چه سالی تاسیس شد؟",
        "چرا دانشگاه تهران معروف است؟",
        "دانشگاه تهران کجا واقع شده است؟",
        "تعداد دانشجویان دانشگاه تهران چقدر است؟",
        
        # About Google
        "گوگل در چه سالی تاسیس شد؟",
        "بنیانگذاران گوگل چه کسانی هستند؟",
        "دفتر مرکزی گوگل کجاست؟",
        "چرا گوگل مشهور است؟",
        "گوگل چه محصولاتی دارد؟",
        
        # About AI
        "هوش مصنوعی چیست؟",
        "هوش مصنوعی در چه سالی معرفی شد؟",
        "کاربردهای هوش مصنوعی چیست؟",
    ]
    
    # Get stats
    stats = qa.get_stats()
    print(f"\n📊 System Statistics:")
    print(f"   - Sentences in knowledge base: {stats['total_sentences']}")
    print(f"   - Vocabulary size: {stats['vocabulary_size']}")
    print(f"   - Stop words: {stats['stop_words_count']}")
    
    # Answer questions
    print("\n" + "=" * 80)
    print("❓ ANSWERING QUESTIONS")
    print("=" * 80)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*80}")
        print(f"Q{i}: {question}")
        
        result = qa.find_answer(question, top_k=4, threshold=0.05)
        
        print(f"\n✅ Answer (confidence: {result['confidence']:.3f}):")
        print(f"   {result['answer']}")
        
        if result['source']:
            print(f"\n   Source: {result['source']['paragraph_name']}")


if __name__ == "__main__":
    run_trained_qa_demo()

