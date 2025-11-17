# -*- coding: utf-8 -*-

"""
Persian Text Processing with Hazm Library
This module provides advanced Persian text processing for the PDF QA Bot
"""

from hazm import Normalizer, WordTokenizer, Stemmer, Lemmatizer, POSTagger
from hazm import sent_tokenize, word_tokenize
import logging

logger = logging.getLogger(__name__)


class PersianTextProcessor:
    """
    Advanced Persian text processor using Hazm library
    Provides normalization, tokenization, stemming, and lemmatization
    """
    
    def __init__(self):
        """Initialize Hazm components"""
        logger.info("Initializing Persian text processor with Hazm...")
        
        try:
            # Core Hazm components
            self.normalizer = Normalizer()
            self.stemmer = Stemmer()
            self.lemmatizer = Lemmatizer()
            self.word_tokenizer = WordTokenizer()
            self.pos_tagger = POSTagger(model='resources/postagger.model')
            
            # Storage for learned vocabulary
            self.vocabulary = set()
            self.word_stems = {}
            self.word_lemmas = {}
            self.processed_paragraphs = []
            
            logger.info("✓ Hazm components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Hazm: {e}")
            raise
    
    def learn_paragraph(self, paragraph):
        """
        Process and learn from a Persian paragraph
        
        Args:
            paragraph: Persian text paragraph to process
            
        Returns:
            Dict with processed information
        """
        try:
            # Step 1: Normalize the text
            normalized_text = self.normalizer.normalize(paragraph)
            logger.info(f"Normalized text: {normalized_text[:100]}...")
            
            # Step 2: Sentence tokenization
            sentences = sent_tokenize(normalized_text)
            logger.info(f"Found {len(sentences)} sentences")
            
            # Step 3: Process each sentence
            processed_sentences = []
            all_words = []
            all_stems = []
            all_lemmas = []
            all_pos_tags = []
            
            for sentence in sentences:
                # Word tokenization
                words = self.word_tokenizer.tokenize(sentence)
                all_words.extend(words)
                
                # Add to vocabulary
                self.vocabulary.update(words)
                
                # Stemming (finding root of words)
                stems = [self.stemmer.stem(word) for word in words]
                all_stems.extend(stems)
                
                # Store stem mappings
                for word, stem in zip(words, stems):
                    if word not in self.word_stems:
                        self.word_stems[word] = stem
                
                # Lemmatization (finding base form)
                lemmas = [self.lemmatizer.lemmatize(word) for word in words]
                all_lemmas.extend(lemmas)
                
                # Store lemma mappings
                for word, lemma in zip(words, lemmas):
                    if word not in self.word_lemmas:
                        self.word_lemmas[word] = lemma
                
                # POS tagging (finding parts of speech)
                try:
                    pos_tags = self.pos_tagger.tag(words)
                    all_pos_tags.extend(pos_tags)
                except:
                    # If POS tagging fails, continue without it
                    pos_tags = [(word, 'UNKNOWN') for word in words]
                
                processed_sentences.append({
                    'original': sentence,
                    'words': words,
                    'stems': stems,
                    'lemmas': lemmas,
                    'pos_tags': pos_tags
                })
            
            # Store processed paragraph
            paragraph_data = {
                'original': paragraph,
                'normalized': normalized_text,
                'sentences': processed_sentences,
                'total_words': len(all_words),
                'unique_words': len(set(all_words)),
                'unique_stems': len(set(all_stems)),
                'unique_lemmas': len(set(all_lemmas))
            }
            
            self.processed_paragraphs.append(paragraph_data)
            
            logger.info(f"Learned paragraph with {len(all_words)} words")
            
            return paragraph_data
            
        except Exception as e:
            logger.error(f"Error processing paragraph: {e}")
            raise
    
    def get_word_info(self, word):
        """
        Get detailed information about a Persian word
        
        Args:
            word: Persian word to analyze
            
        Returns:
            Dict with word information
        """
        normalized = self.normalizer.normalize(word)
        
        return {
            'original': word,
            'normalized': normalized,
            'stem': self.stemmer.stem(normalized),
            'lemma': self.lemmatizer.lemmatize(normalized),
            'in_vocabulary': normalized in self.vocabulary
        }
    
    def find_similar_words(self, word):
        """
        Find words with the same stem (similar meaning)
        
        Args:
            word: Persian word to find similar words for
            
        Returns:
            List of similar words
        """
        normalized = self.normalizer.normalize(word)
        target_stem = self.stemmer.stem(normalized)
        
        similar = [
            w for w, stem in self.word_stems.items()
            if stem == target_stem and w != normalized
        ]
        
        return similar
    
    def get_statistics(self):
        """
        Get statistics about learned text
        
        Returns:
            Dict with statistics
        """
        return {
            'total_paragraphs': len(self.processed_paragraphs),
            'vocabulary_size': len(self.vocabulary),
            'unique_stems': len(set(self.word_stems.values())),
            'unique_lemmas': len(set(self.word_lemmas.values())),
            'total_words_processed': sum(
                p['total_words'] for p in self.processed_paragraphs
            )
        }
    
    def search_in_learned_text(self, query):
        """
        Search for a word or phrase in learned paragraphs
        Uses stem matching for better Persian search
        
        Args:
            query: Search query in Persian
            
        Returns:
            List of matching paragraphs with scores
        """
        query_normalized = self.normalizer.normalize(query)
        query_words = self.word_tokenizer.tokenize(query_normalized)
        query_stems = [self.stemmer.stem(w) for w in query_words]
        
        results = []
        
        for para in self.processed_paragraphs:
            # Get all stems from this paragraph
            para_stems = []
            for sent in para['sentences']:
                para_stems.extend(sent['stems'])
            
            # Calculate match score
            matches = sum(1 for stem in query_stems if stem in para_stems)
            score = matches / len(query_stems) if query_stems else 0
            
            if score > 0:
                results.append({
                    'paragraph': para['original'],
                    'score': score,
                    'matches': matches,
                    'total_query_words': len(query_stems)
                })
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results


# ===== SIMPLE EXAMPLE FUNCTION =====

def process_and_learn_persian_text(text):
    """
    Simple function to process and learn from Persian text
    
    Args:
        text: Persian paragraph or text
        
    Returns:
        Processed information dict
    """
    processor = PersianTextProcessor()
    result = processor.learn_paragraph(text)
    
    print(f"\n{'='*50}")
    print(f"📝 Original Text:")
    print(f"{text[:200]}...")
    print(f"\n✓ Normalized Text:")
    print(f"{result['normalized'][:200]}...")
    print(f"\n📊 Statistics:")
    print(f"   - Total words: {result['total_words']}")
    print(f"   - Unique words: {result['unique_words']}")
    print(f"   - Unique stems: {result['unique_stems']}")
    print(f"   - Sentences: {len(result['sentences'])}")
    print(f"\n🔍 First sentence analysis:")
    first_sent = result['sentences'][0]
    print(f"   Words: {first_sent['words'][:5]}")
    print(f"   Stems: {first_sent['stems'][:5]}")
    print(f"   Lemmas: {first_sent['lemmas'][:5]}")
    print(f"{'='*50}\n")
    
    return result


# ===== USAGE EXAMPLES =====

if __name__ == "__main__":
    # Example 1: Simple usage
    text = """
    دانشگاه تهران یکی از قدیمی�ترین و معتبرترین دانشگاه�های ایران است.
    این دانشگاه در سال ۱۳۱۳ تأسیس شد و دارای دانشکده�های متعددی است.
    دانشجویان این دانشگاه در رشته�های مختلف تحصیل می�کنند.
    """
    
    result = process_and_learn_persian_text(text)
    
    # Example 2: Advanced usage
    print("\n" + "="*50)
    print("Advanced Usage Example:")
    print("="*50 + "\n")
    
    processor = PersianTextProcessor()
    
    # Learn multiple paragraphs
    paragraphs = [
        "شرایط ثبت�نام در دانشگاه شامل داشتن مدرک دیپلم و قبولی در آزمون ورودی است.",
        "دانشجویان باید در ابتدای هر ترم واحدهای درسی خود را انتخاب کنند.",
        "برای انتقالی دانشجو باید حداقل یک ترم را در دانشگاه فعلی گذرانده باشد."
    ]
    
    for para in paragraphs:
        processor.learn_paragraph(para)
    
    # Get statistics
    stats = processor.get_statistics()
    print(f"📊 Total Statistics:")
    print(f"   - Paragraphs learned: {stats['total_paragraphs']}")
    print(f"   - Vocabulary size: {stats['vocabulary_size']}")
    print(f"   - Unique stems: {stats['unique_stems']}")
    
    # Search in learned text
    print(f"\n🔍 Searching for 'ثبت�نام دانشجو':")
    results = processor.search_in_learned_text("ثبت�نام دانشجو")
    for i, result in enumerate(results[:3], 1):
        print(f"\n   Result {i} (score: {result['score']:.2f}):")
        print(f"   {result['paragraph'][:100]}...")
    
    # Analyze a specific word
    print(f"\n🔤 Word Analysis for 'دانشجویان':")
    word_info = processor.get_word_info("دانشجویان")
    print(f"   - Normalized: {word_info['normalized']}")
    print(f"   - Stem: {word_info['stem']}")
    print(f"   - Lemma: {word_info['lemma']}")
    
    # Find similar words
    similar = processor.find_similar_words("دانشجویان")
    if similar:
        print(f"   - Similar words: {', '.join(similar)}")