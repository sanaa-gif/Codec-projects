from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Predefined questions and answers
faq = {
    "what are your working hours": "Our working hours are 9 AM to 5 PM, Monday to Friday.",
    "how can i contact support": "You can contact us at support@example.com or call +1-234-567-890.",
    "where is my order": "Please provide your order ID and we will check the status for you.",
    "i want to return a product": "To return a product, please visit our returns page and follow the instructions."
}

questions = list(faq.keys())
answers = list(faq.values())

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

def chatbot_nlp(user_input):
    user_input = user_input.lower()
    user_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vec, X)
    idx = similarities.argmax()
    if similarities[0, idx] < 0.3:
        return "I'm sorry, I didn't understand that. Could you please rephrase?"
    return answers[idx]

while True:
    user = input("You: ")
    if user.lower() in ['exit', 'quit']:
        print("Bot: Goodbye!")
        break
    print("Bot:", chatbot_nlp(user))
