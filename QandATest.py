# -*- coding: utf-8 -*-
"""
Improved QA System with Tie-Breaking Rules
"""

from hazm import Normalizer, sent_tokenize, word_tokenize, Stemmer

class ImprovedPersianQA:
    def __init__(self, min_score_threshold=0.3):
        """Initialize HAZM components"""
        self.normalizer = Normalizer()
        self.stemmer = Stemmer()
        self.processed_data = {}
        self.min_score_threshold = min_score_threshold
    
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
        
        # Store processed data
        self.processed_data = {
            'original': paragraph,
            'normalized': normalized,
            'sentences': processed_sentences
        }
        
        return self.processed_data
    
    def calculate_score(self, question_stems, sentence_stems, question):
        """
        Calculate a better matching score with semantic tie-breaking
        """
        # Remove common stop words from scoring
        stop_stems = {'اس', 'است', 'شد', 'شدن', 'باشد', 'های', 'ها', '؟', '!', 'در', 'به', 'از'}
        filtered_question_stems = [s for s in question_stems if s not in stop_stems]
        filtered_sentence_stems = [s for s in sentence_stems if s not in stop_stems]
        
        if not filtered_question_stems:
            return 0
        
        # Calculate intersection of important stems
        common_stems = set(filtered_question_stems) & set(filtered_sentence_stems)
        
        # Calculate base score as percentage of matching question stems
        base_score = len(common_stems) / len(filtered_question_stems)
        
        # === TIE-BREAKING RULES ===
        
        # Rule 1: Bonus for matching question type indicators
        question_lower = question.lower()
        bonus = 0
        
        # WHY questions - look for reason indicators
        if any(word in question_lower for word in ['چرا', 'به چه دلیل', 'علت', 'دلیل']):
            reason_indicators = ['دلیل', 'زیرا', 'چون', 'به دلیل', 'علت']
            if any(indicator in sentence_stems for indicator in reason_indicators):
                bonus += 0.3
                print("    + Bonus for reason indicator")
        
        # WHEN questions - look for time indicators
        elif any(word in question_lower for word in ['کی', 'چه زمانی', 'کدام سال', 'چه سالی']):
            time_indicators = ['سال', 'زمان', 'تاریخ']
            if any(indicator in sentence_stems for indicator in time_indicators):
                bonus += 0.3
                print("    + Bonus for time indicator")
        
        # WHERE questions - look for location indicators
        elif any(word in question_lower for word in ['کجا', 'کدام مکان', 'کدام شهر', 'کجاست']):
            location_indicators = ['شهر', 'مکان', 'موقعیت', 'واقع']
            if any(indicator in sentence_stems for indicator in location_indicators):
                bonus += 0.3
                print("    + Bonus for location indicator")
        
        # Rule 2: Bonus for matching the actual question word
        question_word_bonus = 0
        important_question_words = ['چرا', 'کی', 'کجا', 'چگونه', 'چه', 'کدام']
        for q_word in important_question_words:
            if q_word in question_lower and q_word in sentence_stems:
                question_word_bonus += 0.2
                print(f"    + Bonus for question word '{q_word}'")
        
        # Rule 3: Penalty for sentences that are too generic
        generic_indicators = ['این', 'آن', 'دارای', 'می باشد']
        generic_count = sum(1 for word in sentence_stems if word in generic_indicators)
        penalty = generic_count * 0.05
        
        final_score = base_score + bonus + question_word_bonus - penalty
        
        return min(final_score, 1.0)  # Cap at 1.0
    
    def find_answer(self, question):
        """
        Find the answer to a question in the processed paragraph
        """
        if not self.processed_data:
            return "No paragraph processed yet!"
        
        # Normalize and process the question
        question_norm = self.normalizer.normalize(question)
        question_words = word_tokenize(question_norm)
        question_stems = [self.stemmer.stem(word) for word in question_words]
        
        print(f"\n🔍 Question: {question}")
        print(f"Question stems (filtered): {[s for s in question_stems if s not in {'اس', '؟'}]}")
        
        # Find the best matching sentence
        best_sentence = None
        best_score = 0
        candidates = []  # Store all candidates for tie-breaking
        
        for sentence_data in self.processed_data['sentences']:
            sentence = sentence_data['original']
            sentence_stems = sentence_data['stems']
            
            # Calculate improved score with tie-breaking
            score = self.calculate_score(question_stems, sentence_stems, question)
            
            candidates.append({
                'sentence': sentence,
                'score': score,
                'stems': sentence_stems
            })
            
            print(f"  '{sentence[:40]}...' | Score: {score:.2f}")
            
            if score > best_score:
                best_score = score
                best_sentence = sentence
        
        # Final tie-breaking: if multiple have same score, use additional rules
        if best_score >= self.min_score_threshold:
            top_candidates = [c for c in candidates if c['score'] == best_score]
            
            if len(top_candidates) > 1:
                print(f"  Tie detected between {len(top_candidates)} candidates with score {best_score:.2f}")
                best_sentence = self._break_tie(question, top_candidates)
            
            print(f"✓ Best match selected with score: {best_score:.2f}")
            return best_sentence
        else:
            print(f"✗ No good match found (best score: {best_score:.2f}, threshold: {self.min_score_threshold})")
            return "پاسخ دقیقی یافت نشد."
    
    def _break_tie(self, question, candidates):
        """
        Break ties between multiple candidates with the same score
        """
        question_lower = question.lower()
        
        # Rule 1: Prefer sentences that contain the actual question word
        for candidate in candidates:
            sentence_stems = candidate['stems']
            
            # For WHY questions, prefer sentences with reason words
            if any(word in question_lower for word in ['چرا', 'دلیل', 'علت']):
                reason_words = ['دلیل', 'زیرا', 'چون', 'علت']
                if any(word in sentence_stems for word in reason_words):
                    print(f"  Tie broken: selected sentence with reason word")
                    return candidate['sentence']
            
            # For WHEN questions, prefer sentences with numbers (years)
            elif any(word in question_lower for word in ['کی', 'زمان', 'سال']):
                # Look for 4-digit numbers (years)
                import re
                if re.search(r'\d{4}', candidate['sentence']):
                    print(f"  Tie broken: selected sentence with year")
                    return candidate['sentence']
            
            # For WHERE questions, prefer sentences with location words
            elif any(word in question_lower for word in ['کجا', 'مکان', 'شهر']):
                location_words = ['شهر', 'مکان', 'واقع', 'موقعیت']
                if any(word in sentence_stems for word in location_words):
                    print(f"  Tie broken: selected sentence with location word")
                    return candidate['sentence']
        
        # Rule 2: If still tied, prefer longer sentences (more informative)
        longest_candidate = max(candidates, key=lambda x: len(x['sentence']))
        print(f"  Tie broken: selected longest sentence")
        return longest_candidate['sentence']


def run_improved_qa_test():
    """
    Run QA test with improved scoring and tie-breaking
    """
    print("=" * 70)
    print("🧠 IMPROVED QA SYSTEM WITH TIE-BREAKING")
    print("=" * 70)
    
    # Sample Persian paragraph
    paragraph = """
    دانشگاه تهران در سال ۱۳۱۳ تاسیس شد. این دانشگاه در شهر تهران واقع شده است. 
    دانشگاه تهران دارای دانشکده های مهندسی، پزشکی و علوم انسانی است. 
    دلیل معروف بودن این دانشگاه کیفیت بالای آموزشی آن است. 
    تعداد دانشجویان این دانشگاه حدود ۵۰,۰۰۰ نفر است. 
    رشته کامپیوتر در دانشکده مهندسی این دانشگاه تدریس می شود.
    """
    
    # Initialize improved QA system
    qa_system = ImprovedPersianQA(min_score_threshold=0.3)
    
    # Process the paragraph
    print("\n📖 Processing paragraph...")
    qa_system.process_paragraph(paragraph)
    
    # Test questions
    test_questions = [
        "چرا دانشگاه تهران معروف است؟",  # Should match the "دلیل معروف بودن" sentence
        "دانشگاه تهران در چه سالی تاسیس شد؟",  # Should match the first sentence
        "تعداد دانشجویان دانشگاه تهران چقدر است؟",  # Should match the numbers sentence
        "دانشگاه تهران کجا واقع شده است؟",  # Should match location sentence
    ]
    
    print("\n" + "=" * 70)
    print("❓ IMPROVED QUESTION ANSWERING 1")
    print("=" * 70)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. {question}")
        answer = qa_system.find_answer(question)
        print(f"   🤖 پاسخ: {answer}")



    paragraph2 = """
    شرکت گوگل در سال ۱۹۹۸ توسط لری پیج و سرگی برین تاسیس شد..
    دفتر مرکزی این شرکت در mountain view کالیفرنیا قرار دارد..
    گوگل به دلیل موتور جستجوی قدرتمند خود مشهور است..
    این شرکت در سال ۲۰۰۴ به صورت عمومی عرضه شد..
    محصولات اصلی گوگل شامل اندروید، یوتیوب و جیمیل می باشد.
    """
    # Initialize improved QA system
    qa_system = ImprovedPersianQA(min_score_threshold=0.3)

    # Process the paragraph
    print("\n📖 Processing paragraph...")
    qa_system.process_paragraph(paragraph2)


    questions = [
        "گوگل در چه سالی تاسیس شد؟",
        "بنیانگذاران گوگل چه کسانی هستند؟",
        "دفتر مرکزی گوگل کجاست؟",
        "چرا گوگل مشهور است؟",
        "گوگل چه محصولاتی دارد؟"
    ]

    print("\n" + "=" * 70)
    print("❓ IMPROVED QUESTION ANSWERING 2")
    print("=" * 70)

    for i, question in enumerate(questions, 1):
        print(f"\n{i}. {question}")
        answer = qa_system.find_answer(question)
        print(f"   🤖 پاسخ: {answer}")



if __name__ == "__main__":
    run_improved_qa_test()