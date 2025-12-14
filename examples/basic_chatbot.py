"""Example: Basic chatbot usage."""

from chatbot.bot import StressGuardBot


def main():
    """Run basic chatbot example."""
    print("Creating StressGuard chatbot...")
    
    # Initialize chatbot
    bot = StressGuardBot(
        model_name="microsoft/DialoGPT-medium",
        enable_ml_tools=True,
        enable_dl_tools=True,
    )
    
    print("Chatbot initialized!\n")
    
    # Example conversations
    messages = [
        "Hello! How are you?",
        "What can you help me with?",
        "Tell me about machine learning.",
    ]
    
    for message in messages:
        print(f"User: {message}")
        response = bot.chat(message)
        print(f"Bot: {response}\n")
    
    # Get chatbot info
    info = bot.get_info()
    print("\nChatbot Info:")
    print(f"Model: {info['model_info']['model_name']}")
    print(f"Conversation length: {info['conversation_length']} messages")


if __name__ == '__main__':
    main()
