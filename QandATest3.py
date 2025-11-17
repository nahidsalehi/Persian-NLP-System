# -*- coding: utf-8 -*-
"""
Simple QA System using HAZM Library
Finds answers to questions in Persian paragraphs
"""

from hazm import Normalizer, sent_tokenize, word_tokenize, Stemmer

class SimplePersianQA:
    def __init__(self):
        """Initialize HAZM components"""
        self.normalizer = Normalizer()
        self.stemmer = Stemmer()
        self.processed_data = {}
    
    def process_paragraph(self, paragraph):
        """
        Process and analyze a Persian paragraph
        """
        # Normalize the text
        normalized = self.normalizer.normalize(paragraph)
        
        # Tokenize into sentences
        sentences = sent_tokenize(normalized)
        
        # Process each sentence
        processed_sentences = []
        all_words = []
        all_stems = []
        
        for sentence in sentences:
            # Tokenize words
            words = word_tokenize(sentence)
            
            # Get stems
            stems = [self.stemmer.stem(word) for word in words]
            
            processed_sentences.append({
                'original': sentence,
                'words': words,
                'stems': stems
            })
            
            all_words.extend(words)
            all_stems.extend(stems)
        
        # Store processed data
        self.processed_data = {
            'original': paragraph,
            'normalized': normalized,
            'sentences': processed_sentences,
            'all_words': all_words,
            'all_stems': all_stems,
            'unique_stems': set(all_stems)
        }
        
        return self.processed_data
    
    def find_answer(self, question):
        """
        Find the answer to a question in the processed paragraph
        Returns the most relevant whole sentence
        """
        if not self.processed_data:
            return "No paragraph processed yet!"
        
        # Normalize and process the question
        question_norm = self.normalizer.normalize(question)
        question_words = word_tokenize(question_norm)
        question_stems = [self.stemmer.stem(word) for word in question_words]
        
        print(f"\n🔍 Question: {question}")
        print(f"Question stems: {question_stems}")
        
        # Find the best matching sentence
        best_sentence = None
        best_score = 0
        
        for sentence_data in self.processed_data['sentences']:
            sentence = sentence_data['original']
            sentence_stems = sentence_data['stems']
            
            # Calculate match score based on common stems
            common_stems = set(question_stems) & set(sentence_stems)
            score = len(common_stems)
            
            print(f"  Sentence: {sentence[:50]}... | Score: {score}")
            
            if score > best_score:
                best_score = score
                best_sentence = sentence
        
        if best_sentence and best_score > 0:
            return best_sentence  # Return the whole sentence
        else:
            return "پاسخ یافت نشد."


def run_qa_test():
    """
    Run a complete QA test with sample Persian text
    """
    print("=" * 70)
    print("🧠 PERSIAN QUESTION ANSWERING SYSTEM")
    print("=" * 70)
    
    # Sample Persian paragraph
    paragraph = """
    دانشگاه تهران در سال ۱۳۱۳ تاسیس شد. این دانشگاه در شهر تهران واقع شده است. 
    دانشگاه تهران دارای دانشکده های مهندسی، پزشکی و علوم انسانی است. 
    دلیل معروف بودن این دانشگاه کیفیت بالای آموزشی آن است. 
    تعداد دانشجویان این دانشگاه حدود ۵۰,۰۰۰ نفر است. 
    رشته کامپیوتر در دانشکده مهندسی این دانشگاه تدریس می شود.
    """
    
    # Initialize QA system
    qa_system = SimplePersianQA()
    
    # Process the paragraph
    print("\n📖 Processing paragraph...")
    processed = qa_system.process_paragraph(paragraph)
    
    print(f"✓ Normalized text: {processed['normalized'][:100]}...")
    print(f"✓ Number of sentences: {len(processed['sentences'])}")
    print(f"✓ Total words: {len(processed['all_words'])}")
    print(f"✓ Unique stems: {len(processed['unique_stems'])}")
    
    # Test questions
    test_questions = [
        "دانشگاه تهران در چه سالی تاسیس شد؟",
        "دانشگاه تهران در کجا واقع شده است؟",
        "دانشگاه تهران چه دانشکده هایی دارد؟",
        "چرا دانشگاه تهران معروف است؟",
        "تعداد دانشجویان دانشگاه تهران چقدر است؟",
        "رشته کامپیوتر در کدام دانشکده تدریس می شود؟",
        "رنگ دانشگاه تهران چیست؟"  # Question that might not have answer
    ]
    
    print("\n" + "=" * 70)
    print("❓ QUESTION ANSWERING")
    print("=" * 70)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. {question}")
        answer = qa_system.find_answer(question)
        print(f"   🤖 پاسخ: {answer}")


def advanced_qa_example():
    """
    More advanced QA example with different types of questions
    """
    print("\n" + "=" * 70)
    print("🚀 ADVANCED QA EXAMPLE")
    print("=" * 70)
    
    paragraph = """
    شرکت گوگل در سال ۱۹۹۸ توسط لری پیج و سرگی برین تاسیس شد. 
    دفتر مرکزی این شرکت در mountain view کالیفرنیا قرار دارد. 
    گوگل به دلیل موتور جستجوی قدرتمند خود مشهور است. 
    این شرکت در سال ۲۰۰۴ به صورت عمومی عرضه شد. 
    محصولات اصلی گوگل شامل اندروید، یوتیوب و جیمیل می باشد.
    """
    
    qa_system = SimplePersianQA()
    qa_system.process_paragraph(paragraph)
    
    questions = [
        "گوگل در چه سالی تاسیس شد؟",
        "بنیانگذاران گوگل چه کسانی هستند؟",
        "دفتر مرکزی گوگل کجاست؟",
        "چرا گوگل مشهور است؟",
        "گوگل چه محصولاتی دارد؟"
    ]
    
    for question in questions:
        print(f"\n❓ {question}")
        answer = qa_system.find_answer(question)
        print(f"   📝 پاسخ: {answer}")


if __name__ == "__main__":
    # Run basic QA test
    run_qa_test()
    
    # Run advanced example
    advanced_qa_example()
    
    print("\n" + "=" * 70)
    print("✅ TEST COMPLETED SUCCESSFULLY!")
    print("=" * 70)