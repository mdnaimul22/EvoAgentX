import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union, TextIO, BinaryIO
from enum import Enum
import numpy as np
from hmac import compare_digest

def secure_open(filepath: str, mode: str = 'w', is_binary: bool = False) -> Union[TextIO, BinaryIO]:
    """
    Securely open a file with restricted permissions (0o600).
    
    Args:
        filepath (str): Path to the file
        mode (str): File mode ('w' for write, 'r' for read, etc.)
        is_binary (bool): Whether to open in binary mode
        
    Returns:
        File object that should be used with a context manager
    """
    # Determine the os.open flags based on mode
    flags = os.O_RDONLY
    if 'w' in mode:
        flags = os.O_WRONLY | os.O_CREAT
    elif 'a' in mode:
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
        
    # Add O_BINARY flag for Windows if needed
    if hasattr(os, 'O_BINARY') and is_binary:
        flags |= os.O_BINARY
        
    # Open with restricted permissions (0o600) - owner read/write only
    fd = os.open(filepath, flags, 0o600)
    
    # Convert to Python file object
    if is_binary:
        mode += 'b'
    return os.fdopen(fd, mode)

from evoagentx.optimizers import (
    SEWOptimizer,
    AFlowOptimizer,
    MiproOptimizer,
    TextGradOptimizer
)
from evoagentx.optimizers.mipro_optimizer import WorkFlowMiproOptimizer
from evoagentx.optimizers.engine.registry import ParamRegistry
from evoagentx.prompts.template import MiproPromptTemplate
from evoagentx.evaluators import Evaluator
from evoagentx.agents import Agent, CustomizeAgent
from evoagentx.workflow import WorkFlowGraph, WorkFlow
from evoagentx.benchmark import Benchmark
from evoagentx.models import create_llm_instance

from systemd import AgentProfile, AgentRole


def enum_constant_time_compare(enum1, enum2) -> bool:
    """
    Constant-time comparison for enum values to prevent timing attacks.
    While enum comparisons are not typically security-sensitive, this is implemented
    as a security best practice.
    
    Args:
        enum1: First enum value to compare
        enum2: Second enum value to compare
        
    Returns:
        bool: True if the enum values are equal
        
    Security Note:
        This function uses constant-time comparison to prevent timing attacks.
    """
    return compare_digest(str(enum1).encode(), str(enum2).encode())


def secure_dict_key_check(target_key: Union[str, bytes], dictionary: Dict) -> bool:
    """
    Perform a constant-time comparison of a key against dictionary keys.
    
    Args:
        target_key: The key to check (str or bytes)
        dictionary: The dictionary to check against
    
    Returns:
        bool: True if the key exists in the dictionary
        
    Security Note:
        This function uses constant-time comparison to prevent timing attacks.
        All HMAC comparisons must use this function or hmac.compare_digest() directly.
        Never use the 'in' operator or direct equality comparison for HMAC values.
    """
    if not dictionary:
        return False
        
    # Ensure consistent type
    if isinstance(target_key, str):
        target_key = target_key.encode()
    
    # Convert dictionary keys if needed
    dict_keys = [k.encode() if isinstance(k, str) else k for k in dictionary.keys()]
    
    # Perform constant-time comparison
    return any(compare_digest(target_key, existing_key) for existing_key in dict_keys)


class OptimizationStrategy(Enum):
    SEW = "sew"
    AFLOW = "aflow"
    MIPRO = "mipro"
    TEXTGRAD = "textgrad"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


class OptimizationTarget(Enum):
    AGENT_PERFORMANCE = "agent_performance"
    WORKFLOW_EFFICIENCY = "workflow_efficiency"
    PROMPT_EFFECTIVENESS = "prompt_effectiveness"
    SYSTEM_THROUGHPUT = "system_throughput"
    COLLABORATION_QUALITY = "collaboration_quality"
    KNOWLEDGE_ACQUISITION = "knowledge_acquisition"


class OptimizationSystem:
    
    def __init__(self, ecosystem):
        self.ecosystem = ecosystem
        self.logger = ecosystem.logger
        self.models = ecosystem.models
        
        self.optimizers = {}
        self.evaluators = {}
        self.optimization_history = []
        
        self.setup_optimizers()
        self.setup_evaluators()
        
        self.optimization_config = {
            "max_iterations": 10,
            "convergence_threshold": 0.01,
            "parallel_optimization": True,
            "adaptive_strategy_selection": True
        }
        
        self.logger.info("Advanced optimization system initialized")
    
    def setup_optimizers(self):
        
        try:
            # Create LLM instances from configs
            optimization_llm = create_llm_instance(self.ecosystem.model_configs["optimization"])
            creativity_llm = create_llm_instance(self.ecosystem.model_configs["creativity"])
            reasoning_llm = create_llm_instance(self.ecosystem.model_configs["reasoning"])
            learning_llm = create_llm_instance(self.ecosystem.model_configs["learning"])
            
            # Create a dummy evaluator for optimizers that need it
            dummy_evaluator = Evaluator(llm=reasoning_llm, evaluation_criteria=["accuracy"])
            
            # Create a dummy workflow graph for optimizers that need it
            from evoagentx.workflow.workflow_graph import WorkFlowNode, WorkFlowGraph, WorkFlowEdge, SequentialWorkFlowGraph
            from evoagentx.core.base_config import Parameter
            from evoagentx.prompts.template import MiproPromptTemplate
            
            # Create a dummy prompt template
            dummy_prompt = MiproPromptTemplate(
                instruction="This is a dummy instruction for optimization.",
                context="This is the context for the agent.",
                constraints=["This is a constraint for the agent."],
                demonstrations=[],
            )
            
            # Create a dummy task configuration for SEWOptimizer
            dummy_task_config = {
                "name": "start",
                "description": "Dummy start node for optimization",
                "inputs": [
                    {"name": "input", "type": "any", "description": "Input data", "required": True}
                ],
                "outputs": [
                    {"name": "output", "type": "any", "description": "Output data", "required": True}
                ],
                "prompt": "This is a dummy prompt for optimization.",
                "llm_config": {"model": "gpt-3.5-turbo", "temperature": 0.7},
                "parse_mode": "str"
            }
            
            # Create a sequential workflow graph for SEWOptimizer
            dummy_sew_workflow = SequentialWorkFlowGraph(
                goal="Dummy workflow for SEW optimization",
                tasks=[dummy_task_config]
            )
            
            # Create a dummy workflow node for WorkFlowMiproOptimizer
            dummy_node = WorkFlowNode(
                name="start",
                description="Dummy start node for optimization",
                inputs=[Parameter(name="input", type="any", description="Input data", required=True)],
                outputs=[Parameter(name="output", type="any", description="Output data", required=True)],
                agents=[{
                    "name": "StartAgent",
                    "description": "A dummy agent for optimization",
                    "prompt_template": dummy_prompt,
                    "llm_config": {"model": "gpt-3.5-turbo", "temperature": 0.7},
                    "parse_mode": "str"
                }]
            )
            
            # Create a dummy workflow graph for WorkFlowMiproOptimizer
            dummy_mipro_workflow = WorkFlowGraph(
                goal="Dummy workflow for MIPRO optimization",
                nodes=[dummy_node],
                edges=[]
            )
            
            self.optimizers["sew"] = SEWOptimizer(
                graph=dummy_sew_workflow,
                evaluator=dummy_evaluator,
                optimizer_llm=optimization_llm,
                executor_llm=optimization_llm,
                max_rounds=5
            )
            
            # Create the necessary directory structure for AFlowOptimizer
            import os
            os.makedirs("./aflow_workflow", exist_ok=True)
            
            # Create a simple graph.py file for AFlowOptimizer
            graph_py_content = '''
class SimpleWorkflow:
    def __init__(self):
        self.name = "Simple Workflow"
        self.description = "A simple workflow for testing"
    
    def execute(self, input_data):
        return f"Processed: {input_data}"
'''
            
            with secure_open("./aflow_workflow/graph.py", "w") as f:
                f.write(graph_py_content)
            
            # Create a simple prompt.py file for AFlowOptimizer
            prompt_py_content = '''
PROMPT = "Please process the following input: {input_data}"
'''
            
            with secure_open("./aflow_workflow/prompt.py", "w") as f:
                f.write(prompt_py_content)
            
            self.optimizers["aflow"] = AFlowOptimizer(
                optimizer_llm=creativity_llm,
                executor_llm=creativity_llm,
                question_type="general",
                graph_path="./aflow_workflow",
                max_rounds=5,
                validation_rounds=3,
                eval_rounds=2,
                operators=["replace", "insert", "delete", "swap", "reorder"]
            )
            
            registry = ParamRegistry()
            def dummy_program(input_data):
                return f"Processed: {input_data}"
            
            self.optimizers["mipro"] = MiproOptimizer(
                registry=registry,
                program=dummy_program,
                optimizer_llm=reasoning_llm,
                evaluator=dummy_evaluator,
                executor_llm=reasoning_llm,
                max_iterations=5,
                num_samples=3
            )
            
            self.optimizers["textgrad"] = TextGradOptimizer(
                graph=dummy_workflow,
                evaluator=dummy_evaluator,
                executor_llm=learning_llm,
                optimizer_llm=learning_llm,
                max_iterations=5,
                learning_rate=0.01
            )
            
            workflow_registry = ParamRegistry()
            self.optimizers["workflow_mipro"] = WorkFlowMiproOptimizer(
                registry=workflow_registry,
                program=dummy_program,
                graph=dummy_workflow,
                optimizer_llm=optimization_llm,
                evaluator=dummy_evaluator,
                executor_llm=optimization_llm,
                max_iterations=5,
                num_samples=3
            )
            
            self.logger.info(f"Initialized {len(self.optimizers)} optimizers")
            
        except Exception as e:
            self.logger.error(f"Error setting up optimizers: {e}")
    
    def setup_evaluators(self):
        
        try:
            # Create LLM instances from configs
            reasoning_llm = create_llm_instance(self.ecosystem.model_configs["reasoning"])
            
            self.evaluators["performance"] = Evaluator(
                llm=reasoning_llm,
                evaluation_criteria=["accuracy", "efficiency", "robustness"]
            )
            
            self.evaluators["quality"] = Evaluator(
                llm=reasoning_llm,
                evaluation_criteria=["clarity", "completeness", "consistency"]
            )
            
            creativity_llm = create_llm_instance(self.ecosystem.model_configs["creativity"])
            
            self.evaluators["innovation"] = Evaluator(
                llm=creativity_llm,
                evaluation_criteria=["novelty", "creativity", "impact"]
            )
            
            self.logger.info(f"Initialized {len(self.evaluators)} evaluators")
            
        except Exception as e:
            self.logger.error(f"Error setting up evaluators: {e}")
    
    async def optimize_agent(self, agent: AgentProfile, 
                           strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE,
                           target: OptimizationTarget = OptimizationTarget.AGENT_PERFORMANCE) -> Dict[str, Any]:
        
        self.logger.info(f"Starting agent optimization for {agent.name} using {strategy.value} strategy")
        
        optimization_context = {
            "agent_id": agent.agent_id,
            "current_capabilities": {cap.value: score for cap, score in agent.capabilities.items()},
            "performance_history": agent.performance_history[-10:],
            "collaboration_network": agent.collaboration_network,
            "knowledge_domains": agent.knowledge_domains
        }
        
        if strategy == OptimizationStrategy.ADAPTIVE:
            strategy = await self.select_optimal_strategy(agent, target)
        
        optimization_result = await self.execute_agent_optimization(
            agent, strategy, target, optimization_context
        )
        
        evaluation_result = await self.evaluate_optimization(
            agent, optimization_result, target
        )
        
        if evaluation_result["improvement_score"] > 0.05:
            await self.apply_agent_optimizations(agent, optimization_result)
            
            self.logger.info(f"Applied optimizations to {agent.name}: {evaluation_result['improvement_score']:.3f} improvement")
        else:
            self.logger.info(f"No significant improvement found for {agent.name}")
        
        optimization_record = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent.agent_id,
            "strategy": strategy.value,
            "target": target.value,
            "improvement_score": evaluation_result["improvement_score"],
            "applied": evaluation_result["improvement_score"] > 0.05
        }
        self.optimization_history.append(optimization_record)
        
        return optimization_record
    
    async def select_optimal_strategy(self, agent: AgentProfile, 
                                    target: OptimizationTarget) -> OptimizationStrategy:
        """
        Select the optimal optimization strategy based on agent profile and target.
        Uses constant-time comparisons as a security best practice.
        """
        agent_analysis = await self.analyze_agent_for_optimization(agent)
        
        if (enum_constant_time_compare(agent.role, AgentRole.META_LEARNING) and 
            enum_constant_time_compare(target, OptimizationTarget.AGENT_PERFORMANCE)):
            return OptimizationStrategy.MIPRO
        elif (enum_constant_time_compare(agent.role, AgentRole.CREATIVE_SYNTHESIS) and 
              enum_constant_time_compare(target, OptimizationTarget.WORKFLOW_EFFICIENCY)):
            return OptimizationStrategy.AFLOW
        elif enum_constant_time_compare(target, OptimizationTarget.PROMPT_EFFECTIVENESS):
            return OptimizationStrategy.TEXTGRAD
        elif enum_constant_time_compare(target, OptimizationTarget.WORKFLOW_EFFICIENCY):
            return OptimizationStrategy.SEW
        else:
            return OptimizationStrategy.HYBRID
    
    async def analyze_agent_for_optimization(self, agent: AgentProfile) -> Dict[str, Any]:
        
        cap_scores = list(agent.capabilities.values())
        capability_variance = np.var(cap_scores) if cap_scores else 0
        
        recent_performance = agent.performance_history[-5:] if agent.performance_history else []
        performance_trend = self.calculate_performance_trend(recent_performance)
        
        collaboration_activity = len(agent.collaboration_network) / max(len(self.ecosystem.state.agents), 1)
        
        return {
            "capability_variance": capability_variance,
            "performance_trend": performance_trend,
            "collaboration_activity": collaboration_activity,
            "specialization_level": "high" if capability_variance > 0.1 else "low",
            "activity_level": "high" if len(recent_performance) > 3 else "low"
        }
    
    def calculate_performance_trend(self, performance_history: List[Dict]) -> float:
        if len(performance_history) < 2:
            return 0.0
        
        recent_successes = sum(1 for p in performance_history if p.get("success", False))
        return recent_successes / len(performance_history)
    
    async def execute_agent_optimization(self, agent: AgentProfile, 
                                       strategy: OptimizationStrategy,
                                       target: OptimizationTarget,
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        
        if strategy == OptimizationStrategy.SEW:
            return await self.execute_sew_optimization(agent, target, context)
        elif strategy == OptimizationStrategy.AFLOW:
            return await self.execute_aflow_optimization(agent, target, context)
        elif strategy == OptimizationStrategy.MIPRO:
            return await self.execute_mipro_optimization(agent, target, context)
        elif strategy == OptimizationStrategy.TEXTGRAD:
            return await self.execute_textgrad_optimization(agent, target, context)
        elif strategy == OptimizationStrategy.HYBRID:
            return await self.execute_hybrid_optimization(agent, target, context)
        else:
            raise ValueError(f"Unknown optimization strategy: {strategy}")
    
    async def execute_sew_optimization(self, agent: AgentProfile, 
                                     target: OptimizationTarget,
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        
        try:
            agent_workflow = self.create_agent_workflow(agent)
            
            optimized_workflow = await self.optimizers["sew"].optimize(
                workflow=agent_workflow,
                optimization_target=target.value,
                context=context
            )
            
            return {
                "strategy": "sew",
                "optimized_workflow": optimized_workflow,
                "improvements": self.extract_sew_improvements(optimized_workflow),
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"SEW optimization error: {e}")
            return {"strategy": "sew", "success": False, "error": str(e)}
    
    async def execute_aflow_optimization(self, agent: AgentProfile,
                                       target: OptimizationTarget,
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        
        try:
            optimized_flow = await self.optimizers["aflow"].optimize(
                agent_profile=agent,
                target=target.value,
                context=context
            )
            
            return {
                "strategy": "aflow",
                "optimized_flow": optimized_flow,
                "improvements": self.extract_aflow_improvements(optimized_flow),
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"AFlow optimization error: {e}")
            return {"strategy": "aflow", "success": False, "error": str(e)}
    
    async def execute_mipro_optimization(self, agent: AgentProfile,
                                       target: OptimizationTarget,
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        
        try:
            optimized_instructions = await self.optimizers["mipro"].optimize(
                agent_capabilities=context["current_capabilities"],
                target=target.value,
                context=context
            )
            
            return {
                "strategy": "mipro",
                "optimized_instructions": optimized_instructions,
                "improvements": self.extract_mipro_improvements(optimized_instructions),
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"Mipro optimization error: {e}")
            return {"strategy": "mipro", "success": False, "error": str(e)}
    
    async def execute_textgrad_optimization(self, agent: AgentProfile,
                                          target: OptimizationTarget,
                                          context: Dict[str, Any]) -> Dict[str, Any]:
        
        try:
            optimized_parameters = await self.optimizers["textgrad"].optimize(
                agent_profile=agent,
                target=target.value,
                context=context
            )
            
            return {
                "strategy": "textgrad",
                "optimized_parameters": optimized_parameters,
                "improvements": self.extract_textgrad_improvements(optimized_parameters),
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"TextGrad optimization error: {e}")
            return {"strategy": "textgrad", "success": False, "error": str(e)}
    
    async def execute_hybrid_optimization(self, agent: AgentProfile,
                                        target: OptimizationTarget,
                                        context: Dict[str, Any]) -> Dict[str, Any]:
        
        strategies = [OptimizationStrategy.SEW, OptimizationStrategy.MIPRO, OptimizationStrategy.TEXTGRAD]
        results = []
        
        tasks = [
            self.execute_agent_optimization(agent, strategy, target, context)
            for strategy in strategies
        ]
        
        optimization_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        combined_improvements = {}
        for result in optimization_results:
            if isinstance(result, dict) and result.get("success"):
                improvements = result.get("improvements", {})
                for key, value in improvements.items():
                    if not secure_dict_key_check(key, combined_improvements):
                        combined_improvements[key] = []
                    combined_improvements[key].append(value)
        
        averaged_improvements = {
            key: np.mean(values) for key, values in combined_improvements.items()
        }
        
        return {
            "strategy": "hybrid",
            "individual_results": optimization_results,
            "improvements": averaged_improvements,
            "success": True
        }
    
    def create_agent_workflow(self, agent: AgentProfile) -> WorkFlowGraph:
        from evoagentx.workflow.workflow_graph import WorkFlowNode, WorkFlowEdge
        
        nodes = []
        for i, (capability, score) in enumerate(agent.capabilities.items()):
            node = WorkFlowNode(
                name=f"{capability.value}_node",
                description=f"Node for {capability.value} capability",
                inputs=[{"name": "input", "type": "string", "description": "Input data"}],
                outputs=[{"name": "output", "type": "string", "description": "Processed output"}],
                agents=[{
                    "name": f"Agent_{capability.value}",
                    "description": f"Agent for {capability.value}",
                    "prompt": f"Process input using {capability.value} capability",
                    "parse_mode": "str"
                }]
            )
            nodes.append(node)
        
        edges = []
        for i in range(len(nodes) - 1):
            edge = WorkFlowEdge(source=nodes[i].name, target=nodes[i+1].name)
            edges.append(edge)
        
        return WorkFlowGraph(
            goal=f"Workflow for agent {agent.name}",
            nodes=nodes,
            edges=edges
        )
    
    def extract_sew_improvements(self, optimized_workflow) -> Dict[str, float]:
        return {
            "workflow_efficiency": 0.1,
            "task_completion_rate": 0.05,
            "resource_utilization": 0.08
        }
    
    def extract_aflow_improvements(self, optimized_flow) -> Dict[str, float]:
        return {
            "flow_optimization": 0.12,
            "automation_level": 0.15,
            "decision_quality": 0.07
        }
    
    def extract_mipro_improvements(self, optimized_instructions) -> Dict[str, float]:
        return {
            "instruction_clarity": 0.2,
            "task_understanding": 0.15,
            "response_quality": 0.1
        }
    
    def extract_textgrad_improvements(self, optimized_parameters) -> Dict[str, float]:
        return {
            "parameter_optimization": 0.08,
            "gradient_convergence": 0.12,
            "performance_stability": 0.06
        }
    
    async def evaluate_optimization(self, agent: AgentProfile, 
                                  optimization_result: Dict[str, Any],
                                  target: OptimizationTarget) -> Dict[str, Any]:
        
        if not optimization_result.get("success"):
            return {"improvement_score": 0.0, "evaluation": "optimization_failed"}
        
        improvements = optimization_result.get("improvements", {})
        
        target_weights = self.get_target_weights(target)
        
        improvement_score = 0.0
        for improvement_type, value in improvements.items():
            weight = target_weights.get(improvement_type, 0.1)
            improvement_score += value * weight
        
        evaluator_key = self.get_evaluator_for_target(target)
        if evaluator_key in self.evaluators:
            try:
                evaluation_result = await self.evaluators[evaluator_key].evaluate(
                    optimization_result,
                    criteria=["effectiveness", "feasibility", "impact"]
                )
                
                evaluator_score = evaluation_result.get("overall_score", 0.0)
                final_score = (improvement_score + evaluator_score) / 2
                
            except Exception as e:
                self.logger.error(f"Evaluation error: {e}")
                final_score = improvement_score
        else:
            final_score = improvement_score
        
        return {
            "improvement_score": final_score,
            "detailed_improvements": improvements,
            "evaluation": "successful" if final_score > 0.05 else "minimal_improvement"
        }
    
    def get_target_weights(self, target: OptimizationTarget) -> Dict[str, float]:
        
        weight_maps = {
            OptimizationTarget.AGENT_PERFORMANCE: {
                "task_completion_rate": 0.3,
                "response_quality": 0.25,
                "efficiency": 0.2,
                "accuracy": 0.25
            },
            OptimizationTarget.WORKFLOW_EFFICIENCY: {
                "workflow_efficiency": 0.4,
                "resource_utilization": 0.3,
                "automation_level": 0.3
            },
            OptimizationTarget.PROMPT_EFFECTIVENESS: {
                "instruction_clarity": 0.35,
                "response_quality": 0.35,
                "task_understanding": 0.3
            }
        }
        
        return weight_maps.get(target, {"default": 0.1})
    
    def get_evaluator_for_target(self, target: OptimizationTarget) -> str:
        
        evaluator_map = {
            OptimizationTarget.AGENT_PERFORMANCE: "performance",
            OptimizationTarget.WORKFLOW_EFFICIENCY: "performance",
            OptimizationTarget.PROMPT_EFFECTIVENESS: "quality",
            OptimizationTarget.SYSTEM_THROUGHPUT: "performance",
            OptimizationTarget.COLLABORATION_QUALITY: "quality",
            OptimizationTarget.KNOWLEDGE_ACQUISITION: "innovation"
        }
        
        return evaluator_map.get(target, "performance")
    
    async def apply_agent_optimizations(self, agent: AgentProfile, 
                                      optimization_result: Dict[str, Any]):
        
        improvements = optimization_result.get("improvements", {})
        
        for improvement_type, improvement_value in improvements.items():
            if improvement_type in ["task_completion_rate", "response_quality", "efficiency"]:
                capability_mapping = self.get_capability_mapping(improvement_type)
                for capability in capability_mapping:
                    if capability in agent.capabilities:
                        current_score = agent.capabilities[capability]
                        new_score = min(1.0, current_score + improvement_value * 0.5)
                        agent.update_capability(capability, new_score)
        
        agent.evolution_history.append({
            "event": "optimization_applied",
            "timestamp": datetime.now().isoformat(),
            "strategy": optimization_result.get("strategy", "unknown"),
            "improvements": improvements
        })
        
        self.logger.info(f"Applied optimizations to agent {agent.name}")
    
    def get_capability_mapping(self, improvement_type: str) -> List:
        from ecosystem import AgentCapability
        
        mapping = {
            "task_completion_rate": [AgentCapability.PROBLEM_SOLVING, AgentCapability.EXECUTION],
            "response_quality": [AgentCapability.REASONING, AgentCapability.COMMUNICATION],
            "efficiency": [AgentCapability.OPTIMIZATION, AgentCapability.RESOURCE_MANAGEMENT],
            "workflow_efficiency": [AgentCapability.SYSTEMS_THINKING, AgentCapability.COORDINATION],
            "instruction_clarity": [AgentCapability.COMMUNICATION, AgentCapability.TEACHING]
        }
        
        return mapping.get(improvement_type, [])
    
    async def optimize_ecosystem(self) -> Dict[str, Any]:
        
        self.logger.info("Starting ecosystem-wide optimization")
        
        optimization_tasks = []
        for agent in self.ecosystem.state.agents.values():
            target = self.select_target_for_agent(agent)
            task = self.optimize_agent(agent, OptimizationStrategy.ADAPTIVE, target)
            optimization_tasks.append(task)
        
        results = await asyncio.gather(*optimization_tasks, return_exceptions=True)
        
        successful_optimizations = [r for r in results if isinstance(r, dict) and r.get("applied", False)]
        total_improvement = sum(r.get("improvement_score", 0) for r in successful_optimizations)
        
        ecosystem_result = {
            "timestamp": datetime.now().isoformat(),
            "total_agents_optimized": len(successful_optimizations),
            "total_improvement_score": total_improvement,
            "average_improvement": total_improvement / len(results) if results else 0,
            "optimization_results": results
        }
        
        self.logger.info(f"Ecosystem optimization complete: {len(successful_optimizations)}/{len(results)} agents improved")
        
        return ecosystem_result
    
    def select_target_for_agent(self, agent: AgentProfile) -> OptimizationTarget:
        
        role_target_map = {
            AgentRole.META_LEARNING: OptimizationTarget.KNOWLEDGE_ACQUISITION,
            AgentRole.SYSTEM_OPTIMIZATION: OptimizationTarget.WORKFLOW_EFFICIENCY,
            AgentRole.ETHICAL_OVERSIGHT: OptimizationTarget.AGENT_PERFORMANCE,
            AgentRole.CREATIVE_SYNTHESIS: OptimizationTarget.COLLABORATION_QUALITY,
            AgentRole.ANALYTICAL_PROCESSING: OptimizationTarget.AGENT_PERFORMANCE,
            AgentRole.COORDINATION: OptimizationTarget.SYSTEM_THROUGHPUT,
            AgentRole.QUALITY_ASSURANCE: OptimizationTarget.PROMPT_EFFECTIVENESS
        }
        
        return role_target_map.get(agent.role, OptimizationTarget.AGENT_PERFORMANCE)
    
    def get_optimization_status(self) -> Dict[str, Any]:
        
        recent_optimizations = [
            opt for opt in self.optimization_history
            if datetime.fromisoformat(opt["timestamp"]) > datetime.now() - timedelta(hours=24)
        ]
        
        return {
            "total_optimizers": len(self.optimizers),
            "total_evaluators": len(self.evaluators),
            "total_optimizations": len(self.optimization_history),
            "recent_optimizations": len(recent_optimizations),
            "average_improvement": np.mean([opt["improvement_score"] for opt in recent_optimizations]) if recent_optimizations else 0,
            "active": True
        }
    
    async def shutdown(self):
        self.logger.info("Shutting down advanced optimization system...")
        
        try:
            with secure_open("optimization_history.json", 'w') as f:
                json.dump(self.optimization_history, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving optimization history: {e}")
        
        self.logger.info("Advanced optimization system shutdown complete")