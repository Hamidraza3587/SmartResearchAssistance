import os
import logging
from typing import Optional, List, Dict, Any, Union
from dotenv import load_dotenv
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
    AutoConfig
)
from huggingface_hub import login, hf_hub_download
import warnings

# Disable unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [2]  # End of sequence token
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

class LLMClient:
    """Client for interacting with Hugging Face language models."""
    
    def __init__(self, model_name: str = "gpt2", use_4bit: bool = False):
        """
        Initialize the LLM client with a Hugging Face model.
        
        Args:
            model_name: Name of the Hugging Face model to use (default: 'gpt2' for testing)
            use_4bit: Whether to use 4-bit quantization (disabled by default for stability)
        """
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize with None
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer with simplified configuration."""
        try:
            logger.info(f"Loading model {self.model_name} on {self.device}...")
            
            # First, try to load the config to check if the model exists
            try:
                config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
                logger.info(f"Successfully loaded config for {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load model config: {str(e)}")
                raise
            
            # Load tokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    use_fast=True,
                    trust_remote_code=True
                )
                logger.info("Tokenizer loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load tokenizer: {str(e)}")
                raise
            
            # Load model with simplified configuration
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    device_map="auto" if self.device == "cuda" else None,
                    torch_dtype=torch.float32  # Use float32 for stability
                )
                logger.info("Model loaded successfully")
                
                # Create text generation pipeline
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if self.device == "cuda" else -1,
                    framework="pt"
                )
                logger.info("Text generation pipeline created")
                
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                raise
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            # Fall back to CPU if CUDA is not available
            if "CUDA" in str(e) and self.device == "cuda":
                logger.warning("Falling back to CPU...")
                self.device = "cpu"
                self._load_model()  # Try again with CPU
            else:
                raise
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _truncate_text(self, text: str, max_tokens: int = 4000) -> str:
        """Truncate text to a maximum number of tokens."""
        try:
            tokens = self.tokenizer.encode(text, truncation=True, max_length=max_tokens)
            return self.tokenizer.decode(tokens, skip_special_tokens=True)
        except Exception as e:
            logger.warning(f"Error during token truncation: {str(e)}")
            # Fallback to simple character-based truncation if tokenization fails
            return text[:max_tokens * 4]  # Rough estimate: 4 chars per token

    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 500, 
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate text using the language model.
        
        Args:
            prompt: The prompt to generate text from
            max_tokens: Maximum number of tokens to generate
            temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
            **kwargs: Additional generation parameters
            
        Returns:
            The generated text
            
        Raises:
            Exception: If text generation fails
        """
        try:
            # Validate input
            if not prompt or not isinstance(prompt, str):
                raise ValueError("Prompt must be a non-empty string")
                
            # Truncate the prompt if it's too long
            max_context_length = 4000  # Conservative limit for most models
            if len(prompt) > max_context_length * 4:  # Rough estimate: 4 chars per token
                logger.warning("Prompt is very long, truncating...")
                prompt = self._truncate_text(prompt, max_context_length - max_tokens)
            
            # Prepare the prompt with instruction format if needed
            try:
                if "instruct" in self.model_name.lower():
                    messages = [{"role": "user", "content": prompt}]
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                else:
                    formatted_prompt = prompt
            except Exception as e:
                logger.warning(f"Error formatting prompt, using as-is: {str(e)}")
                formatted_prompt = prompt
            
            # Prepare generation parameters
            generation_params = {
                'max_new_tokens': min(max_tokens, 1000),  # Cap max tokens
                'temperature': max(0.1, min(1.0, temperature)),  # Ensure valid range
                'do_sample': True,
                'pad_token_id': self.tokenizer.eos_token_id or 0,
                'stopping_criteria': StoppingCriteriaList([StopOnTokens()])
            }
            
            # Update with any additional parameters, but don't override existing ones
            for key, value in kwargs.items():
                if key not in generation_params:
                    generation_params[key] = value
            
            try:
                # Generate text with error handling
                output = self.pipeline(formatted_prompt, **generation_params)
                
                if not output or not isinstance(output, list) or not output[0].get("generated_text"):
                    raise ValueError("Unexpected model output format")
                
                # Extract and clean the generated text
                generated_text = output[0]["generated_text"]
                
                # Remove the input prompt from the output if present
                if generated_text.startswith(formatted_prompt):
                    generated_text = generated_text[len(formatted_prompt):].strip()
                
                return generated_text
                
            except Exception as gen_error:
                # Try with a simpler prompt if the first attempt fails
                logger.warning(f"Generation failed, retrying with simplified prompt: {str(gen_error)}")
                simplified_prompt = prompt[:1000]  # Try with just the first 1000 chars
                output = self.pipeline(simplified_prompt, **generation_params)
                return output[0]["generated_text"].replace(simplified_prompt, "").strip()
            
        except Exception as e:
            error_msg = f"Error generating text: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg) from e
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts using the model's hidden states.
        
        Args:
            texts: List of text strings to get embeddings for
            
        Returns:
            List of embeddings (list of floats)
        """
        try:
            # Use the model's encoder if available, otherwise use the full model
            if hasattr(self.model, 'get_encoder'):
                encoder = self.model.get_encoder()
            else:
                encoder = self.model
            
            # Tokenize the input texts
            inputs = self.tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                return_tensors="pt",
                max_length=512
            ).to(self.device)
            
            # Get hidden states
            with torch.no_grad():
                outputs = encoder(**inputs, output_hidden_states=True)
                # Use the last hidden state
                last_hidden_states = outputs.hidden_states[-1]
                # Mean pooling
                attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden_states.size()).float()
                sum_embeddings = torch.sum(last_hidden_states * attention_mask, 1)
                sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
                embeddings = (sum_embeddings / sum_mask).cpu().numpy().tolist()
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise Exception(f"Error generating embeddings: {str(e)}")
