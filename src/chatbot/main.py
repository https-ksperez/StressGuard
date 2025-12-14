"""Main entry point for StressGuard chatbot."""

import argparse
import sys
from pathlib import Path

from chatbot.bot import StressGuardBot
from utils.config import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)


def interactive_mode(bot: StressGuardBot):
    """
    Run chatbot in interactive mode.
    
    Args:
        bot: StressGuardBot instance
    """
    print("\n" + "="*60)
    print("StressGuard AI Chatbot")
    print("Samsung Innovation Campus Project")
    print("="*60)
    print("\nCommands:")
    print("  - Type your message to chat")
    print("  - Type 'info' to see chatbot information")
    print("  - Type 'reset' to clear conversation history")
    print("  - Type 'quit' or 'exit' to exit")
    print("="*60 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
                
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! Thank you for using StressGuard.")
                break
                
            elif user_input.lower() == 'info':
                info = bot.get_info()
                print("\nChatbot Information:")
                print(f"  Model: {info['model_info']['model_name']}")
                print(f"  Device: {info['model_info']['device']}")
                print(f"  Parameters: {info['model_info']['parameters']:,}")
                print(f"  ML Tools: {'Enabled' if info['ml_tools_enabled'] else 'Disabled'}")
                print(f"  DL Tools: {'Enabled' if info['dl_tools_enabled'] else 'Disabled'}")
                print(f"  Conversation Length: {info['conversation_length']} messages\n")
                continue
                
            elif user_input.lower() == 'reset':
                bot.reset_conversation()
                print("\nConversation history cleared.\n")
                continue
                
            # Get chatbot response
            response = bot.chat(user_input)
            print(f"Bot: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! Thank you for using StressGuard.")
            break
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}")
            print(f"\nError: {e}\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="StressGuard AI Chatbot - Samsung Innovation Campus"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='microsoft/DialoGPT-medium',
        help='LLM model to use (default: microsoft/DialoGPT-medium)'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--no-ml',
        action='store_true',
        help='Disable ML tools'
    )
    parser.add_argument(
        '--no-dl',
        action='store_true',
        help='Disable DL tools'
    )
    parser.add_argument(
        '--message',
        type=str,
        help='Single message to send to chatbot (non-interactive mode)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config) if args.config else Config()
    
    # Get model name from args or config
    model_name = args.model or config.get('model.name', 'microsoft/DialoGPT-medium')
    
    try:
        # Initialize chatbot
        print("Initializing StressGuard chatbot...")
        bot = StressGuardBot(
            model_name=model_name,
            enable_ml_tools=not args.no_ml,
            enable_dl_tools=not args.no_dl,
        )
        print("Chatbot initialized successfully!\n")
        
        # Single message mode or interactive mode
        if args.message:
            response = bot.chat(args.message)
            print(f"You: {args.message}")
            print(f"Bot: {response}")
        else:
            interactive_mode(bot)
            
    except Exception as e:
        logger.error(f"Error initializing chatbot: {e}")
        print(f"\nError: {e}")
        print("\nTip: If you encounter memory issues, try using a smaller model.")
        sys.exit(1)


if __name__ == '__main__':
    main()
