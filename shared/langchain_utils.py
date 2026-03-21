"""
LangChain utilities for pattern implementations.
Provides helpers for creating chains, agents, and workflows.
"""

import os
from typing import Optional, Any, Dict
from loguru import logger

try:
    from langchain_community.llms import Ollama
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain.chains import LLMChain
except ImportError:
    logger.warning("LangChain not installed. Some utilities will not be available.")
    Ollama = None
    ChatPromptTemplate = None
    PromptTemplate = None
    StrOutputParser = None
    RunnablePassthrough = None
    LLMChain = None


def get_langchain_llm(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs
) -> Any:
    """
    Get a LangChain Ollama LLM instance.
    
    Args:
        model: Model name (default from env)
        base_url: Ollama base URL
        temperature: Sampling temperature
        **kwargs: Additional LLM parameters
        
    Returns:
        LangChain Ollama LLM instance
    """
    if Ollama is None:
        raise ImportError("LangChain is not installed. Install with: pip install langchain langchain-community")
    
    model = model or os.getenv("OLLAMA_DEFAULT_MODEL", "llama3")
    base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    return Ollama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        **kwargs
    )


def create_ollama_chain(
    prompt_template: str,
    model: Optional[str] = None,
    input_variables: Optional[list] = None,
    **llm_kwargs
) -> Any:
    """
    Create a simple LangChain chain with Ollama.
    
    Args:
        prompt_template: Prompt template string
        model: Model name
        input_variables: List of input variable names
        **llm_kwargs: Additional LLM parameters
        
    Returns:
        LangChain chain
    """
    if PromptTemplate is None or LLMChain is None:
        raise ImportError("LangChain is not installed.")
    
    llm = get_langchain_llm(model=model, **llm_kwargs)
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=input_variables or []
    )
    
    return LLMChain(llm=llm, prompt=prompt)


def create_chat_chain(
    system_message: Optional[str] = None,
    model: Optional[str] = None,
    **llm_kwargs
) -> Any:
    """
    Create a chat chain with system message support.
    
    Args:
        system_message: System message/instruction
        model: Model name
        **llm_kwargs: Additional LLM parameters
        
    Returns:
        LangChain chat chain
    """
    if ChatPromptTemplate is None or StrOutputParser is None:
        raise ImportError("LangChain is not installed.")
    
    llm = get_langchain_llm(model=model, **llm_kwargs)
    
    messages = []
    if system_message:
        messages.append(("system", system_message))
    messages.append(("human", "{input}"))
    
    prompt = ChatPromptTemplate.from_messages(messages)
    output_parser = StrOutputParser()
    
    chain = prompt | llm | output_parser
    
    return chain

