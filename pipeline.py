import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np

from evoagentx.evaluators import Evaluator
from evoagentx.benchmark import (
    HotPotQA, HumanEval, AFlowHumanEval, MATH, MBPP, AFlowMBPP, Benchmark
)
from evoagentx.agents import Agent, CustomizeAgent
from evoagentx.workflow import WorkFlowGraph, WorkFlow
from evoagentx.models import create_llm_instance
from systemd import AgentProfile, AgentRole, AgentCapability


class EvaluationType(Enum):
    PERFORMANCE = "performance"
    QUALITY = "quality"
    EFFICIENCY = "efficiency"
    ACCURACY = "accuracy"
    CREATIVITY = "creativity"
    COLLABORATION = "collaboration"
    LEARNING = "learning"


class EvaluationScope(Enum):
    AGENT = "agent"
    WORKFLOW = "workflow"
    SYSTEM = "system"
    ECOSYSTEM = "ecosystem"


class EvaluationPipeline:
    
    def __init__(self, ecosystem):
        self.ecosystem = ecosystem
        self.logger = ecosystem.logger
        self.models = ecosystem.models
        
        self.evaluators = {}
        self.benchmarks = {}
        self.evaluation_history = []
        self.evaluation_metrics = {}
        
        self.setup_evaluators()
        self.setup_benchmarks()
        
        self.logger.info("Evaluation pipeline system initialized")
    
    def setup_evaluators(self):
        
        # Create LLM instances from configs
        reasoning_llm = create_llm_instance(self.ecosystem.model_configs["reasoning"])
        creativity_llm = create_llm_instance(self.ecosystem.model_configs["creativity"])
        learning_llm = create_llm_instance(self.ecosystem.model_configs["learning"])
        
        self.evaluators["performance"] = Evaluator(
            llm=reasoning_llm,
            evaluation_criteria=[
                "task_completion_rate",
                "response_time",
                "resource_efficiency",
                "error_rate"
            ]
        )
        
        self.evaluators["quality"] = Evaluator(
            llm=reasoning_llm,
            evaluation_criteria=[
                "accuracy",
                "completeness",
                "clarity",
                "consistency"
            ]
        )
        
        self.evaluators["creativity"] = Evaluator(
            llm=creativity_llm,
            evaluation_criteria=[
                "novelty",
                "originality",
                "usefulness",
                "feasibility"
            ]
        )
        
        self.evaluators["collaboration"] = Evaluator(
            llm=learning_llm,
            evaluation_criteria=[
                "communication_effectiveness",
                "knowledge_sharing",
                "team_coordination",
                "conflict_resolution"
            ]
        )
        
        self.logger.info(f"Initialized {len(self.evaluators)} evaluators")
    
    def setup_benchmarks(self):
        
        try:
            self.benchmarks["humaneval"] = HumanEval()
            self.benchmarks["aflow_humaneval"] = AFlowHumanEval()
            self.benchmarks["mbpp"] = MBPP()
            self.benchmarks["aflow_mbpp"] = AFlowMBPP()
            
            self.benchmarks["math"] = MATH()
            
            self.benchmarks["hotpotqa"] = HotPotQA()
            
            self.logger.info(f"Initialized {len(self.benchmarks)} benchmarks")
            
        except Exception as e:
            self.logger.error(f"Error setting up benchmarks: {e}")
    
    async def evaluate_agent(self, agent: AgentProfile, 
                           evaluation_types: List[EvaluationType] = None,
                           use_benchmarks: bool = True) -> Dict[str, Any]:
        
        if evaluation_types is None:
            evaluation_types = [EvaluationType.PERFORMANCE, EvaluationType.QUALITY]
        
        self.logger.info(f"Starting agent evaluation for {agent.name}")
        
        evaluation_results = {}
        
        for eval_type in evaluation_types:
            try:
                result = await self.run_agent_evaluation(agent, eval_type)
                evaluation_results[eval_type.value] = result
            except Exception as e:
                self.logger.error(f"Error in {eval_type.value} evaluation for {agent.name}: {e}")
                evaluation_results[eval_type.value] = {"error": str(e), "score": 0.0}
        
        if use_benchmarks:
            benchmark_results = await self.run_agent_benchmarks(agent)
            evaluation_results["benchmarks"] = benchmark_results
        
        overall_score = self.calculate_overall_score(evaluation_results)
        evaluation_results["overall_score"] = overall_score
        
        evaluation_record = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent.agent_id,
            "agent_name": agent.name,
            "evaluation_types": [et.value for et in evaluation_types],
            "overall_score": overall_score,
            "results": evaluation_results
        }
        self.evaluation_history.append(evaluation_record)
        
        self.logger.info(f"Agent evaluation complete for {agent.name}: {overall_score:.3f}")
        
        return evaluation_results
    
    async def run_agent_evaluation(self, agent: AgentProfile, 
                                 eval_type: EvaluationType) -> Dict[str, Any]:
        
        evaluator_key = self.get_evaluator_for_type(eval_type)
        if evaluator_key not in self.evaluators:
            raise ValueError(f"No evaluator available for {eval_type.value}")
        
        evaluator = self.evaluators[evaluator_key]
        
        evaluation_context = {
            "agent_id": agent.agent_id,
            "agent_name": agent.name,
            "agent_role": agent.role.value,
            "capabilities": {cap.value: score for cap, score in agent.capabilities.items()},
            "performance_history": agent.performance_history[-10:],
            "knowledge_domains": agent.knowledge_domains,
            "collaboration_network": agent.collaboration_network
        }
        
        if eval_type == EvaluationType.PERFORMANCE:
            return await self.evaluate_performance(agent, evaluator, evaluation_context)
        elif eval_type == EvaluationType.QUALITY:
            return await self.evaluate_quality(agent, evaluator, evaluation_context)
        elif eval_type == EvaluationType.CREATIVITY:
            return await self.evaluate_creativity(agent, evaluator, evaluation_context)
        elif eval_type == EvaluationType.COLLABORATION:
            return await self.evaluate_collaboration(agent, evaluator, evaluation_context)
        else:
            return await self.evaluate_generic(agent, evaluator, evaluation_context, eval_type)
    
    async def evaluate_performance(self, agent: AgentProfile, evaluator: Evaluator, 
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        
        recent_performance = agent.performance_history[-20:] if agent.performance_history else []
        
        success_rate = sum(1 for p in recent_performance if p.get("success", False)) / max(len(recent_performance), 1)
        
        response_times = [p.get("response_time", 1.0) for p in recent_performance if "response_time" in p]
        avg_response_time = np.mean(response_times) if response_times else 1.0
        
        capability_scores = list(agent.capabilities.values())
        resource_efficiency = np.mean(capability_scores) if capability_scores else 0.5
        
        evaluation_data = {
            "success_rate": success_rate,
            "avg_response_time": avg_response_time,
            "resource_efficiency": resource_efficiency,
            "context": context
        }
        
        try:
            evaluator_result = await evaluator.evaluate(
                evaluation_data,
                criteria=["task_completion_rate", "response_time", "resource_efficiency"]
            )
            
            return {
                "success_rate": success_rate,
                "avg_response_time": avg_response_time,
                "resource_efficiency": resource_efficiency,
                "evaluator_score": evaluator_result.get("overall_score", 0.0),
                "detailed_scores": evaluator_result.get("detailed_scores", {}),
                "score": (success_rate + resource_efficiency + (1.0 / max(avg_response_time, 0.1))) / 3.0
            }
            
        except Exception as e:
            self.logger.error(f"Evaluator error in performance evaluation: {e}")
            return {
                "success_rate": success_rate,
                "avg_response_time": avg_response_time,
                "resource_efficiency": resource_efficiency,
                "score": (success_rate + resource_efficiency) / 2.0
            }
    
    async def evaluate_quality(self, agent: AgentProfile, evaluator: Evaluator,
                             context: Dict[str, Any]) -> Dict[str, Any]:
        
        recent_performance = agent.performance_history[-10:]
        
        quality_indicators = {
            "consistency": self.calculate_consistency_score(recent_performance),
            "accuracy": self.calculate_accuracy_score(recent_performance),
            "completeness": self.calculate_completeness_score(recent_performance)
        }
        
        try:
            evaluator_result = await evaluator.evaluate(
                {"quality_indicators": quality_indicators, "context": context},
                criteria=["accuracy", "completeness", "clarity", "consistency"]
            )
            
            return {
                "quality_indicators": quality_indicators,
                "evaluator_score": evaluator_result.get("overall_score", 0.0),
                "detailed_scores": evaluator_result.get("detailed_scores", {}),
                "score": np.mean(list(quality_indicators.values()))
            }
            
        except Exception as e:
            self.logger.error(f"Evaluator error in quality evaluation: {e}")
            return {
                "quality_indicators": quality_indicators,
                "score": np.mean(list(quality_indicators.values()))
            }
    
    async def evaluate_creativity(self, agent: AgentProfile, evaluator: Evaluator,
                                context: Dict[str, Any]) -> Dict[str, Any]:
        
        creativity_score = agent.capabilities.get(AgentCapability.CREATIVITY, 0.5)
        innovation_score = len(set(agent.knowledge_domains)) / 10.0
        
        creative_outputs = [
            p for p in agent.performance_history 
            if "creative" in str(p).lower() or "innovative" in str(p).lower()
        ]
        
        creativity_metrics = {
            "base_creativity": creativity_score,
            "domain_diversity": innovation_score,
            "creative_output_frequency": len(creative_outputs) / max(len(agent.performance_history), 1)
        }
        
        try:
            evaluator_result = await evaluator.evaluate(
                {"creativity_metrics": creativity_metrics, "context": context},
                criteria=["novelty", "originality", "usefulness", "feasibility"]
            )
            
            return {
                "creativity_metrics": creativity_metrics,
                "evaluator_score": evaluator_result.get("overall_score", 0.0),
                "score": np.mean(list(creativity_metrics.values()))
            }
            
        except Exception as e:
            self.logger.error(f"Evaluator error in creativity evaluation: {e}")
            return {
                "creativity_metrics": creativity_metrics,
                "score": np.mean(list(creativity_metrics.values()))
            }
    
    async def evaluate_collaboration(self, agent: AgentProfile, evaluator: Evaluator,
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        
        collaboration_score = agent.capabilities.get(AgentCapability.COLLABORATION, 0.5)
        network_size = len(agent.collaboration_network)
        communication_score = agent.capabilities.get(AgentCapability.COMMUNICATION, 0.5)
        
        collaboration_metrics = {
            "collaboration_capability": collaboration_score,
            "network_size": min(1.0, network_size / 10.0),
            "communication_effectiveness": communication_score
        }
        
        try:
            evaluator_result = await evaluator.evaluate(
                {"collaboration_metrics": collaboration_metrics, "context": context},
                criteria=["communication_effectiveness", "knowledge_sharing", "team_coordination"]
            )
            
            return {
                "collaboration_metrics": collaboration_metrics,
                "evaluator_score": evaluator_result.get("overall_score", 0.0),
                "score": np.mean(list(collaboration_metrics.values()))
            }
            
        except Exception as e:
            self.logger.error(f"Evaluator error in collaboration evaluation: {e}")
            return {
                "collaboration_metrics": collaboration_metrics,
                "score": np.mean(list(collaboration_metrics.values()))
            }
    
    async def evaluate_generic(self, agent: AgentProfile, evaluator: Evaluator,
                             context: Dict[str, Any], eval_type: EvaluationType) -> Dict[str, Any]:
        
        relevant_capabilities = self.get_relevant_capabilities(eval_type)
        capability_scores = [
            agent.capabilities.get(cap, 0.5) for cap in relevant_capabilities
        ]
        
        generic_score = np.mean(capability_scores) if capability_scores else 0.5
        
        return {
            "evaluation_type": eval_type.value,
            "relevant_capabilities": [cap.value for cap in relevant_capabilities],
            "capability_scores": capability_scores,
            "score": generic_score
        }
    
    async def run_agent_benchmarks(self, agent: AgentProfile) -> Dict[str, Any]:
        
        benchmark_results = {}
        
        relevant_benchmarks = self.select_benchmarks_for_agent(agent)
        
        for benchmark_name in relevant_benchmarks:
            if benchmark_name in self.benchmarks:
                try:
                    test_agent = self.create_benchmark_agent(agent)
                    
                    benchmark = self.benchmarks[benchmark_name]
                    result = await benchmark.run(test_agent, max_samples=5)
                    
                    benchmark_results[benchmark_name] = {
                        "score": result.get("score", 0.0),
                        "details": result
                    }
                    
                except Exception as e:
                    self.logger.error(f"Benchmark {benchmark_name} error for {agent.name}: {e}")
                    benchmark_results[benchmark_name] = {"score": 0.0, "error": str(e)}
        
        return benchmark_results
    
    def select_benchmarks_for_agent(self, agent: AgentProfile) -> List[str]:
        
        benchmarks = []
        
        if agent.role in [AgentRole.ANALYTICAL_PROCESSING, AgentRole.SYSTEM_OPTIMIZATION]:
            benchmarks.extend(["humaneval", "mbpp"])
        
        if agent.role in [AgentRole.KNOWLEDGE_ACQUISITION, AgentRole.META_LEARNING]:
            benchmarks.extend(["hotpotqa", "math"])
        
        if agent.capabilities.get(AgentCapability.PROGRAMMING, 0) > 0.6:
            benchmarks.extend(["humaneval", "mbpp"])
        
        if agent.capabilities.get(AgentCapability.REASONING, 0) > 0.6:
            benchmarks.extend(["math", "hotpotqa"])
        
        return list(set(benchmarks))
    
    def create_benchmark_agent(self, agent_profile: AgentProfile) -> CustomizeAgent:
        
        benchmark_agent = CustomizeAgent(
            name=f"Benchmark_{agent_profile.name}",
            description=f"Benchmark agent for {agent_profile.name}",
            prompt="Solve the given problem: {problem}",
            llm_config=self.ecosystem.model_configs["reasoning"],
            inputs=[
                {"name": "problem", "type": "string", "description": "Problem to solve"}
            ],
            outputs=[
                {"name": "solution", "type": "string", "description": "Problem solution"}
            ]
        )
        
        return benchmark_agent
    
    def calculate_consistency_score(self, performance_history: List[Dict]) -> float:
        if len(performance_history) < 2:
            return 0.5
        
        success_rates = []
        window_size = 3
        
        for i in range(len(performance_history) - window_size + 1):
            window = performance_history[i:i + window_size]
            success_rate = sum(1 for p in window if p.get("success", False)) / len(window)
            success_rates.append(success_rate)
        
        if not success_rates:
            return 0.5
        
        variance = np.var(success_rates)
        consistency = max(0.0, 1.0 - variance)
        
        return consistency
    
    def calculate_accuracy_score(self, performance_history: List[Dict]) -> float:
        if not performance_history:
            return 0.5
        
        success_count = sum(1 for p in performance_history if p.get("success", False))
        return success_count / len(performance_history)
    
    def calculate_completeness_score(self, performance_history: List[Dict]) -> float:
        if not performance_history:
            return 0.5
        
        complete_count = sum(1 for p in performance_history if p.get("complete", True))
        return complete_count / len(performance_history)
    
    def get_evaluator_for_type(self, eval_type: EvaluationType) -> str:
        
        evaluator_map = {
            EvaluationType.PERFORMANCE: "performance",
            EvaluationType.QUALITY: "quality",
            EvaluationType.CREATIVITY: "creativity",
            EvaluationType.COLLABORATION: "collaboration",
            EvaluationType.EFFICIENCY: "performance",
            EvaluationType.ACCURACY: "quality",
            EvaluationType.LEARNING: "performance"
        }
        
        return evaluator_map.get(eval_type, "performance")
    
    def get_relevant_capabilities(self, eval_type: EvaluationType) -> List[AgentCapability]:
        
        capability_map = {
            EvaluationType.PERFORMANCE: [AgentCapability.EXECUTION, AgentCapability.EFFICIENCY],
            EvaluationType.QUALITY: [AgentCapability.ACCURACY, AgentCapability.ATTENTION_TO_DETAIL],
            EvaluationType.CREATIVITY: [AgentCapability.CREATIVITY, AgentCapability.INNOVATION],
            EvaluationType.COLLABORATION: [AgentCapability.COLLABORATION, AgentCapability.COMMUNICATION],
            EvaluationType.LEARNING: [AgentCapability.LEARNING, AgentCapability.ADAPTATION]
        }
        
        return capability_map.get(eval_type, [AgentCapability.PROBLEM_SOLVING])
    
    def calculate_overall_score(self, evaluation_results: Dict[str, Any]) -> float:
        
        scores = []
        
        for eval_type, result in evaluation_results.items():
            if eval_type == "benchmarks":
                benchmark_scores = [r.get("score", 0.0) for r in result.values() if isinstance(r, dict)]
                if benchmark_scores:
                    scores.append(np.mean(benchmark_scores))
            elif isinstance(result, dict) and "score" in result:
                scores.append(result["score"])
        
        return np.mean(scores) if scores else 0.0
    
    async def evaluate_ecosystem(self) -> Dict[str, Any]:
        
        self.logger.info("Starting ecosystem evaluation")
        
        agent_evaluations = []
        for agent in self.ecosystem.state.agents.values():
            try:
                evaluation = await self.evaluate_agent(agent, use_benchmarks=False)
                agent_evaluations.append(evaluation)
            except Exception as e:
                self.logger.error(f"Error evaluating agent {agent.name}: {e}")
        
        ecosystem_metrics = self.calculate_ecosystem_metrics(agent_evaluations)
        
        system_evaluation = await self.evaluate_system_performance()
        
        ecosystem_result = {
            "timestamp": datetime.now().isoformat(),
            "total_agents": len(self.ecosystem.state.agents),
            "agents_evaluated": len(agent_evaluations),
            "ecosystem_metrics": ecosystem_metrics,
            "system_evaluation": system_evaluation,
            "overall_ecosystem_score": ecosystem_metrics.get("average_score", 0.0)
        }
        
        self.logger.info(f"Ecosystem evaluation complete: {ecosystem_result['overall_ecosystem_score']:.3f}")
        
        return ecosystem_result
    
    def calculate_ecosystem_metrics(self, agent_evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        
        if not agent_evaluations:
            return {"average_score": 0.0, "score_variance": 0.0}
        
        overall_scores = [eval_result.get("overall_score", 0.0) for eval_result in agent_evaluations]
        
        return {
            "average_score": np.mean(overall_scores),
            "score_variance": np.var(overall_scores),
            "min_score": np.min(overall_scores),
            "max_score": np.max(overall_scores),
            "score_distribution": {
                "high_performers": sum(1 for score in overall_scores if score > 0.8),
                "medium_performers": sum(1 for score in overall_scores if 0.5 <= score <= 0.8),
                "low_performers": sum(1 for score in overall_scores if score < 0.5)
            }
        }
    
    async def evaluate_system_performance(self) -> Dict[str, Any]:
        
        system_metrics = {
            "intelligence_score": self.ecosystem.state.system_intelligence_score,
            "health_score": self.ecosystem.state.system_health_score,
            "agent_count": len(self.ecosystem.state.agents),
            "resource_usage": self.ecosystem.state.resource_usage,
            "emergency_count": self.ecosystem.state.emergency_count
        }
        
        performance_factors = [
            system_metrics["intelligence_score"],
            system_metrics["health_score"],
            min(1.0, system_metrics["agent_count"] / 20.0),
            max(0.0, 1.0 - system_metrics["resource_usage"]),
            max(0.0, 1.0 - system_metrics["emergency_count"] / 10.0)
        ]
        
        system_performance_score = np.mean(performance_factors)
        
        return {
            "system_metrics": system_metrics,
            "performance_factors": performance_factors,
            "system_performance_score": system_performance_score
        }
    
    def get_evaluation_status(self) -> Dict[str, Any]:
        
        recent_evaluations = [
            eval_record for eval_record in self.evaluation_history
            if datetime.fromisoformat(eval_record["timestamp"]) > datetime.now() - timedelta(hours=24)
        ]
        
        return {
            "total_evaluators": len(self.evaluators),
            "total_benchmarks": len(self.benchmarks),
            "total_evaluations": len(self.evaluation_history),
            "recent_evaluations": len(recent_evaluations),
            "average_score": np.mean([e["overall_score"] for e in recent_evaluations]) if recent_evaluations else 0.0,
            "active": True
        }
    
    async def shutdown(self):
        self.logger.info("Shutting down evaluation pipeline...")
        
        try:
            with open("evaluation_history.json", "w") as f:
                json.dump(self.evaluation_history, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving evaluation history: {e}")
        
        self.logger.info("Evaluation pipeline shutdown complete")