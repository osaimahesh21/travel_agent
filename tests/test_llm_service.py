import pytest
from travel_assistant.services.llm_service import LLMService



def test_llm_service_generate():
    # Create an instance of LLMService with dummy parameters
    llm_service = LLMService(base_url="http://localhost:11434", model_name="mistral")

    # Test the generate method with a sample prompt
    prompt = "What are the top attractions in Paris?"
    response = llm_service.generate(prompt)

    # Check that the response is a string and contains the expected text
    assert isinstance(response, str)
    assert "Generated response for prompt" in response
    assert prompt[:50] in response


