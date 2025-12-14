"""Local LLM model manager for StressGuard chatbot."""

import os
import logging
from typing import Optional, Dict, Any
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalLLMModel:
    """Manages local LLM model for the chatbot."""
    
    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        device: Optional[str] = None,
        max_length: int = 1000,
    ):
        """
        Initialize the local LLM model.
        
        Args:
            model_name: HuggingFace model name or path to local model
            device: Device to run model on ('cuda', 'cpu', or None for auto)
            max_length: Maximum length for generated responses
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing LLM model: {model_name} on {self.device}")
        
        self.tokenizer = None
        self.model = None
        self.generator = None
        self._load_model()
        
    def _load_model(self):
        """Load the tokenizer and model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Move model to device
            self.model.to(self.device)
            
            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """
        Generate a response to the given prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        try:
            # Generate response
            outputs = self.generator(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
            
            # Extract generated text
            response = outputs[0]["generated_text"]
            
            # Remove prompt from response if present
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
                
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error processing your request."
            
    def chat(self, message: str, context: Optional[str] = None) -> str:
        """
        Generate a chat response with optional context.
        
        Args:
            message: User message
            context: Optional conversation context
            
        Returns:
            Chatbot response
        """
        # Build prompt with context if provided
        if context:
            prompt = f"{context}\nUser: {message}\nBot:"
        else:
            prompt = f"User: {message}\nBot:"
            
        return self.generate_response(prompt)
        
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
            "parameters": sum(p.numel() for p in self.model.parameters()),
        }
