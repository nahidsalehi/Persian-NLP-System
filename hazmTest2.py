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
            self.pos_tagger = POSTagger(model='models/postagger.model')
            
            # Storage for learned vocabulary
            self.vocabulary = set()
            self.word_stems = {}
            self.word_lemmas = {}
            self.processed_paragraphs = []
            
            logger.info("Hazm components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Hazm: {e}")
            raise
    
    def learn_paragraph(self, paragraph):
        """
        Process and learn from a Persian paragraph
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
                
                # Stemming
                stems = [self.stemmer.stem(word) for word in words]
                all_stems.extend(stems)
                
                # Store stem mappings
                for word, stem in zip(words, stems):
                    if word not in self.word_stems:
                        self.word_stems[word] = stem
                
                # Lemmatization
                lemmas = [self.lemmatizer.lemmatize(word) for word in words]
                all_lemmas.extend(lemmas)
                
                # Store lemma mappings
                for word, lemma in zip(words, lemmas):
                    if word not in self.word_lemmas:
                        self.word_lemmas[word] = lemma
                
                # POS tagging
                try:
                    pos_tags = self.pos_tagger.tag(words)
                    all_pos_tags.extend(pos_tags)
                except:
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
        """Get detailed information about a Persian word"""
        normalized = self.normalizer.normalize(word)
        
        return {
            'original': word,
            'normalized': normalized,
            'stem': self.stemmer.stem(normalized),
            'lemma': self.lemmatizer.lemmatize(normalized),
            'in_vocabulary': normalized in self.vocabulary
        }
    
    def find_similar_words(self, word):
        """Find words with the same stem"""
        normalized = self.normalizer.normalize(word)
        target_stem = self.stemmer.stem(normalized)
        
        similar = [
            w for w, stem in self.word_stems.items()
            if stem == target_stem and w != normalized
        ]
        
        return similar
    
    def get_statistics(self):
        """Get statistics about learned text"""
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
        """Search for a word or phrase in learned paragraphs"""
        query_normalized = self.normalizer.normalize(query)
        query_words = self.word_tokenizer.tokenize(query_normalized)
        query_stems = [self.stemmer.stem(w) for w in query_words]
        
        results = []
        
        for para in self.processed_paragraphs:
            para_stems = []
            for sent in para['sentences']:
                para_stems.extend(sent['stems'])
            
            matches = sum(1 for stem in query_stems if stem in para_stems)
            score = matches / len(query_stems) if query_stems else 0
            
            if score > 0:
                results.append({
                    'paragraph': para['original'],
                    'score': score,
                    'matches': matches,
                    'total_query_words': len(query_stems)
                })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results


def process_and_learn_persian_text(text):
    """Simple function to process and learn from Persian text"""
    processor = PersianTextProcessor()
    result = processor.learn_paragraph(text)
    
    print(f"\n{'='*50}")
    print(f"Original Text:")
    print(f"{text[:200]}...")
    print(f"\nNormalized Text:")
    print(f"{result['normalized'][:200]}...")
    print(f"\nStatistics:")
    print(f"   - Total words: {result['total_words']}")
    print(f"   - Unique words: {result['unique_words']}")
    print(f"   - Unique stems: {result['unique_stems']}")
    print(f"   - Sentences: {len(result['sentences'])}")
    
    if result['sentences']:
        first_sent = result['sentences'][0]
        print(f"\nFirst sentence analysis:")
        print(f"   Words: {first_sent['words'][:5]}")
        print(f"   Stems: {first_sent['stems'][:5]}")
        print(f"   Lemmas: {first_sent['lemmas'][:5]}")
    
    print(f"{'='*50}\n")
    return result

def find_similar_words(self, word):
        """Find words with the same stem"""
        normalized = self.normalizer.normalize(word)
        target_stem = self.stemmer.stem(normalized)
        
        similar = [
            w for w, stem in self.word_stems.items()
            if stem == target_stem and w != normalized
        ]
        
        return similar
    
def search_in_learned_text(self, query):
        """Search for a word or phrase in learned paragraphs"""
        query_normalized = self.normalizer.normalize(query)
        query_words = self.word_tokenizer.tokenize(query_normalized)
        query_stems = [self.stemmer.stem(w) for w in query_words]
        
        results = []
        
        for para in self.processed_paragraphs:
            para_stems = []
            for sent in para['sentences']:
                para_stems.extend(sent['stems'])
            
            matches = sum(1 for stem in query_stems if stem in para_stems)
            score = matches / len(query_stems) if query_stems else 0
            
            if score > 0:
                results.append({
                    'paragraph': para['original'],
                    'score': score,
                    'matches': matches
                })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results


if __name__ == "__main__":
    print("Testing Persian Text Processor...")
    
# ----------------------------------------------------Test 1: Simple Persian sentence
    print("\n1. Testing with simple Persian text:")
    print("-" * 40)
    
    simple_text = "دانشگاه  تهران از بزرگترين دانشگاه هاميباشد."
    
    try:
        processor = PersianTextProcessor()
        result = processor.learn_paragraph(simple_text)
        
        print(f"\n✓ Successfully processed Persian text!")
        print(f"Original: {simple_text}")
        print(f"Normalized: {result['normalized']}")
        
    except Exception as e:
        print(f"✗ Error with simple Persian: {e}")
        



# ----------------------------------------------------Test 2: More complex Persian text
    print("\n\n2. Testing with more complex Persian text:")
    print("-" * 40)
    
    complex_text = """
    دانشجویان دانشگاه تهران در کلاس های درس حاضر می شوند.
    آن ها مطالب علمی را یاد می گیرند.
    """
    
    try:
        result2 = processor.learn_paragraph(complex_text)
        print(f"✓ Successfully processed complex Persian text!")
        
        # Show statistics
        stats = processor.get_statistics()
        print(f"\n📊 Statistics:")
        print(f"   Total paragraphs: {stats['total_paragraphs']}")
        print(f"   Vocabulary size: {stats['vocabulary_size']}")
        print(f"   Unique stems: {stats['unique_stems']}")
        
    except Exception as e:
        print(f"✗ Error with complex Persian: {e}")
        
    
    # Test 3: Word analysis
    print("\n\n3. Testing word analysis:")
    print("-" * 40)
    
    test_words = ["دانشگاهيان", "دانشجویان", "کلاسها"]
    
    for word in test_words:
        try:
            word_info = processor.get_word_info(word)
            print(f"\nWord: '{word}'")
            print(f"  Normalized: '{word_info['normalized']}'")
            print(f"  Stem: '{word_info['stem']}'")
            print(f"  Lemma: '{word_info['lemma']}'")
            print(f"  In vocabulary: {word_info['in_vocabulary']}")
        except Exception as e:
            print(f"✗ Error analyzing '{word}': {e}")
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY! 🎉")
    print("=" * 60)



#----------------------------------------------------------- Learn multiple paragraphs
    paragraphs = [
        "دانشگاه تهران در سال ۱۳۱۳ تاسیس شد.",
        "این دانشگاه دارای دانشکده های متعددی است.",
        "دانشجویان در رشته های مختلف تحصیل می کنند.",
        "شرایط ثبت نام شامل مدرک دیپلم و آزمون ورودی است."
    ]
    
    print("Learning paragraphs...")
    for i, para in enumerate(paragraphs, 1):
        print(f"Paragraph {i}: {para}")
        processor.learn_paragraph(para)
    
    # Show statistics
    stats = processor.get_statistics()
    print(f"\n📊 FINAL STATISTICS:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Search examples
    print(f"\n🔍 SEARCH EXAMPLES:")
    searches = ["دانشگاه", "ثبت نام", "دانشجویان"]
    
    for search_term in searches:
        print(f"\nSearching for: '{search_term}'")
        results = processor.search_in_learned_text(search_term)
        for j, result in enumerate(results, 1):
            print(f"   {j}. Score: {result['score']:.2f} - {result['paragraph'][:50]}...")
    
    # Word analysis
    print(f"\n🔤 WORD ANALYSIS:")
    words_to_analyze = ["دانشگاه", "دانشجویان", "ثبت"]
    
    for word in words_to_analyze:
        info = processor.get_word_info(word)
        print(f"\n'{word}':")
        print(f"   Stem: '{info['stem']}'")
        print(f"   Lemma: '{info['lemma']}'")
    
    # Similar words
    print(f"\n🔄 SIMILAR WORDS:")
    test_word = "دانشگاه"
    similar = processor.find_similar_words(test_word)
    if similar:
        print(f"Words similar to '{test_word}': {similar}")
    else:
        print(f"No similar words found for '{test_word}'")

    # Test with simple English first
    # processor = PersianTextProcessor()
    # print("Processor created successfully!")