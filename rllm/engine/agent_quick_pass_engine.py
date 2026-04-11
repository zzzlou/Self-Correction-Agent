from rllm.engine.agent_execution_engine import AgentExecutionEngine


class QuickPassEngine(AgentExecutionEngine):
    async def get_model_response(self, prompt, application_id, **kwargs) -> str:
        return ""