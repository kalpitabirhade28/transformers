import unittest
from transformers.agents.agent_types import AgentImage
from transformers.agents.agents import AgentError, ReactCodeAgent, ReactJsonAgent
from transformers.agents.monitoring import stream_to_gradio


class MonitoringTester(unittest.TestCase):
    class FakeLLMEngine:
        def __init__(self, input_tokens=10, output_tokens=20):
            self.last_input_token_count = input_tokens
            self.last_output_token_count = output_tokens

        def __call__(self, prompt, **kwargs):
            return kwargs.get("response", "")

    def test_code_agent_metrics(self):
        agent = ReactCodeAgent(tools=[], llm_engine=self.FakeLLMEngine(), max_iterations=1)
        agent.run("Task")
        self.assertEqual(agent.monitor.total_input_token_count, 10)
        self.assertEqual(agent.monitor.total_output_token_count, 20)

    def test_json_agent_metrics(self):
        engine = self.FakeLLMEngine()
        engine.__call__ = lambda prompt, **kwargs: '{"action": "final_answer", "action_input": {"answer": "image"}}'
        agent = ReactJsonAgent(tools=[], llm_engine=engine, max_iterations=1)
        agent.run("Task")
        self.assertEqual(agent.monitor.total_input_token_count, 10)
        self.assertEqual(agent.monitor.total_output_token_count, 20)

    def test_streaming_text_output(self):
        engine = self.FakeLLMEngine()
        engine.__call__ = lambda prompt, **kwargs: """
Code:
```py
final_answer('Answer.')
```"""
        agent = ReactCodeAgent(tools=[], llm_engine=engine, max_iterations=1)
        outputs = list(stream_to_gradio(agent, task="Task", test_mode=True))
        self.assertIn("Answer.", outputs[-1].content)

    def test_streaming_image_output(self):
        engine = self.FakeLLMEngine()
        engine.__call__ = lambda prompt, **kwargs: '{"action": "final_answer", "action_input": {"answer": "image"}}'
        agent = ReactJsonAgent(tools=[], llm_engine=engine, max_iterations=1)
        outputs = list(stream_to_gradio(agent, task="Task", image=AgentImage(value="path.png"), test_mode=True))
        self.assertEqual(outputs[-1].content["path"], "path.png")

    def test_streaming_with_error(self):
        engine = self.FakeLLMEngine()
        engine.__call__ = lambda prompt, **kwargs: (_ for _ in ()).throw(AgentError("Error"))
        agent = ReactCodeAgent(tools=[], llm_engine=engine, max_iterations=1)
        outputs = list(stream_to_gradio(agent, task="Task", test_mode=True))
        self.assertIn("Error", outputs[-1].content)
