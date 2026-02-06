import pytest
from travel_assistant.models.schemas import UserQuery
from travel_assistant.models.schemas import AssistantResponse
from travel_assistant.models.schemas import RetrievedDocument

def test_user_query_validation():
    # Valid input
    valid_input = {
        "query": "Plan a trip to Paris",
        "session_id": "abc123",
        "metadata": {"destination": "Paris", "days": 5}
    }
    user_query = UserQuery(**valid_input)
    assert user_query.query == valid_input["query"]
    assert user_query.session_id == valid_input["session_id"]
    assert user_query.metadata == valid_input["metadata"]

    # Missing required field 'query'
    with pytest.raises(ValueError):
        UserQuery(session_id="abc123")

    # Invalid type for 'metadata'
    with pytest.raises(ValueError):
        UserQuery(query="Plan a trip", metadata="not a dict")



def test_assistant_response_validation():
    # Valid input
    valid_input = {
        "response": "I recommend visiting the Eiffel Tower.",
        "metadata": {"attractions": ["Eiffel Tower"]}
    }
    assistant_response = AssistantResponse(**valid_input)
    assert assistant_response.response == valid_input["response"]
    assert assistant_response.metadata == valid_input["metadata"]

    # Missing required field 'response'
    with pytest.raises(ValueError):
        AssistantResponse(metadata={"attractions": ["Eiffel Tower"]})

    # Invalid type for 'metadata'
    with pytest.raises(ValueError):
        AssistantResponse(response="I recommend visiting the Eiffel Tower.", metadata="not a dict")

def test_retrieved_document_validation():
    # Valid input
    valid_input = {
        "content": "The Eiffel Tower is a famous landmark in Paris.",
        "source": {"source": "travel_guides", "id": "doc123"},
        "similarity_score": 0.95
    }
    retrieved_doc = RetrievedDocument(**valid_input)
    
    assert retrieved_doc.content == valid_input["content"]
    assert retrieved_doc.source == valid_input["source"]

    # Missing required field 'content'
    with pytest.raises(ValueError):
        RetrievedDocument(source={"source": "travel_guides", "id": "doc123"})

    # Invalid type for 'source'
    with pytest.raises(ValueError):
        RetrievedDocument(content="The Eiffel Tower is a famous landmark in Paris.", source="not a dict")