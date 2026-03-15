import abc


class BaseAgent(abc.ABC):
    """
       Abstract base class for agents.
       Subclasses must implement the act method.
       """
    
    @abc.abstractmethod
    def invoke(self, messages):
        pass
