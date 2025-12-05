# tests/test_tools.py
import pytest
from unittest.mock import Mock, patch
from module2 import llm1_outils, secure_input, agent_tools,State,should_continue

class TestSecureInput:
    """Tests pour la validation toxicité"""
    
    def test_toxic_content_blocked(self):
        """Vérifie que le contenu toxique est remplacé"""
        with patch('module2.Detoxify') as mock_detox:
            mock_detox.return_value.predict.return_value = {"toxicity": 0.8}
            result = secure_input("insulte ici")
            assert result == "How to talk politely to others ?"
    
    def test_clean_content_passes(self):
        """Vérifie que le contenu propre passe"""
        with patch('module2.Detoxify') as mock_detox:
            mock_detox.return_value.predict.return_value = {"toxicity": 0.2}
            result = secure_input("hello world")
            assert result == "hello world"
    
    def test_threshold_boundary(self):
        """Teste la valeur limite (0.5)"""
        with patch('module2.Detoxify') as mock_detox:
            mock_detox.return_value.predict.return_value = {"toxicity": 0.5}
            result = secure_input("borderline")
            assert result == "How to talk politely to others ?"


class TestToolsCreation:
    """Tests pour la creation des outils"""
    
    def test_llm1_outils_returns_three_tools(self):
        """Verifie qu'on a bien 3 outils"""
        tools = llm1_outils()
        assert len(tools) == 3
        assert all(hasattr(tool, 'name') for tool in tools)
    
    def test_agent_tools_returns_structured_tools(self):
        """Verifie que agent_tools retourne des StructuredTools"""
        tools = agent_tools()
        assert len(tools) == 3
        assert all(tool.name in [
            "Academic_web_recents_requests",
            "Anything_about_python", 
            "Document_summarizer"
        ] for tool in tools)


# tests/test_state_management.py
class TestStateManagement:
    """Tests pour la gestion du state"""
    
    def test_state_initialization(self):
        """Vérifie l'initialisation correcte du State"""
        state = State(messages=[{"role": "user", "content": "test"}])
        assert len(state['messages']) == 1
        assert state['messages'][0]['role'] == "user"
    
    def test_should_continue_with_tool_calls(self):
        """Vérifie le routing quand il y a des tool_calls"""
        mock_message = Mock()
        mock_message.tool_calls = [{"name": "test"}]
        state = State(messages=[mock_message])
        
        result = should_continue(state)
        assert result == 'tools'
    
    def test_should_continue_without_tool_calls(self):
        """Vérifie le routing quand il n'y a pas de tool_calls"""
        mock_message = Mock()
        mock_message.tool_calls = []
        state = State(messages=[mock_message])
        
        result = should_continue(state)
        assert result == 'end'