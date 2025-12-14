"""Main chatbot implementation with LLM and ML/DL tool integration."""

import logging
from typing import Optional, Dict, Any, List
from chatbot.llm_model import LocalLLMModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StressGuardBot:
    """
    Main chatbot class integrating LLM with ML/DL tools.
    
    This chatbot can:
    - Generate responses using a local LLM
    - Use ML tools for data analysis and predictions
    - Use DL tools for advanced AI tasks
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        enable_ml_tools: bool = True,
        enable_dl_tools: bool = True,
    ):
        """
        Initialize the StressGuard chatbot.
        
        Args:
            model_name: LLM model to use
            enable_ml_tools: Whether to enable ML tools
            enable_dl_tools: Whether to enable DL tools
        """
        logger.info("Initializing StressGuard chatbot...")
        
        # Initialize LLM
        self.llm = LocalLLMModel(model_name=model_name)
        
        # Track conversation history
        self.conversation_history: List[Dict[str, str]] = []
        
        # Tool availability flags
        self.ml_tools_enabled = enable_ml_tools
        self.dl_tools_enabled = enable_dl_tools
        
        # Lazy load tools
        self._ml_toolkit = None
        self._dl_toolkit = None
        
        logger.info("StressGuard chatbot initialized successfully")
        
    @property
    def ml_toolkit(self):
        """Lazy load ML toolkit."""
        if self._ml_toolkit is None and self.ml_tools_enabled:
            from ml_tools.ml_toolkit import MLToolkit
            self._ml_toolkit = MLToolkit()
        return self._ml_toolkit
        
    @property
    def dl_toolkit(self):
        """Lazy load DL toolkit."""
        if self._dl_toolkit is None and self.dl_tools_enabled:
            from dl_tools.dl_toolkit import DLToolkit
            self._dl_toolkit = DLToolkit()
        return self._dl_toolkit
        
    def chat(self, message: str, use_context: bool = True) -> str:
        """
        Process a user message and generate a response.
        
        Args:
            message: User's message
            use_context: Whether to use conversation history
            
        Returns:
            Chatbot's response
        """
        # Build context from conversation history
        context = None
        if use_context and self.conversation_history:
            context = self._build_context()
            
        # Generate response
        response = self.llm.chat(message, context=context)
        
        # Store in conversation history
        self.conversation_history.append({
            "role": "user",
            "content": message
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        # Keep only last 10 exchanges to prevent context from growing too large
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
            
        return response
        
    def _build_context(self, max_turns: int = 5) -> str:
        """
        Build conversation context from history.
        
        Args:
            max_turns: Maximum number of conversation turns to include
            
        Returns:
            Formatted context string
        """
        recent_history = self.conversation_history[-(max_turns * 2):]
        context_parts = []
        
        for entry in recent_history:
            if entry["role"] == "user":
                context_parts.append(f"User: {entry['content']}")
            else:
                context_parts.append(f"Bot: {entry['content']}")
                
        return "\n".join(context_parts)
        
    def analyze_with_ml(self, data: Any, task: str = "classification") -> Dict[str, Any]:
        """
        Use ML tools to analyze data.
        
        Args:
            data: Input data for analysis
            task: ML task type (classification, regression, clustering, etc.)
            
        Returns:
            Analysis results
        """
        if not self.ml_tools_enabled:
            return {"error": "ML tools are not enabled"}
            
        return self.ml_toolkit.analyze(data, task=task)
        
    def process_with_dl(self, data: Any, task: str = "inference") -> Dict[str, Any]:
        """
        Use DL tools to process data.
        
        Args:
            data: Input data for processing
            task: DL task type (inference, training, etc.)
            
        Returns:
            Processing results
        """
        if not self.dl_tools_enabled:
            return {"error": "DL tools are not enabled"}
            
        return self.dl_toolkit.process(data, task=task)
        
    def reset_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")
        
    def get_info(self) -> Dict[str, Any]:
        """
        Get chatbot information.
        
        Returns:
            Dictionary with chatbot info
        """
        return {
            "model_info": self.llm.get_model_info(),
            "ml_tools_enabled": self.ml_tools_enabled,
            "dl_tools_enabled": self.dl_tools_enabled,
            "conversation_length": len(self.conversation_history),
        }
