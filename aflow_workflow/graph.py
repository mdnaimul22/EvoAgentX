
class SimpleWorkflow:
    def __init__(self):
        self.name = "Simple Workflow"
        self.description = "A simple workflow for testing"
    
    def execute(self, input_data):
        return f"Processed: {input_data}"
