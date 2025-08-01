import asyncio
import json
import statistics
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import sys
import numpy as np
import logging

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from systemd import AgentRole, AgentCapability, AgentProfile, EcosystemState
from evoagentx.benchmark import Benchmark, HotPotQA, HumanEval, MATH, MBPP
from evoagentx.evaluators import Evaluator


class BenchmarkCategory(Enum):
    """Categories of benchmarks"""
    INTELLIGENCE = "intelligence"
    CREATIVITY = "creativity"
    PROBLEM_SOLVING = "problem_solving"
    LEARNING = "learning"
    COLLABORATION = "collaboration"
    ADAPTATION = "adaptation"
    EFFICIENCY = "efficiency"
    ROBUSTNESS = "robustness"
    SAFETY = "safety"
    ETHICAL_REASONING = "ethical_reasoning"


class BenchmarkDifficulty(Enum):
    """Difficulty levels for benchmarks"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class BenchmarkType(Enum):
    """Types of benchmarks"""
    AUTOMATED = "automated"
    HUMAN_EVALUATED = "human_evaluated"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


@dataclass
class BenchmarkResult:
    """Result of a benchmark execution"""
    benchmark_id: str
    agent_id: str
    category: BenchmarkCategory
    difficulty: BenchmarkDifficulty
    score: float  # 0-1
    max_score: float
    time_taken: float
    success: bool
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    execution_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSuite:
    """A collection of related benchmarks"""
    suite_id: str
    name: str
    description: str
    category: BenchmarkCategory
    benchmarks: List[str]  # List of benchmark IDs
    weightings: Dict[str, float]  # Weight of each benchmark in the suite
    difficulty_range: Tuple[BenchmarkDifficulty, BenchmarkDifficulty]
    estimated_duration: float  # Estimated time in minutes
    prerequisites: List[str] = field(default_factory=list)


@dataclass
class BenchmarkProfile:
    """Profile of an agent's benchmarking performance"""
    agent_id: str
    total_benchmarks: int
    completed_benchmarks: int
    average_score: float
    category_scores: Dict[BenchmarkCategory, float]
    difficulty_scores: Dict[BenchmarkDifficulty, float]
    improvement_trend: float  # Positive for improving, negative for declining
    strengths: List[str]
    weaknesses: List[str]
    last_updated: datetime = field(default_factory=datetime.now)


class BenchmarkGenerator:
    """Generate dynamic benchmarks based on agent capabilities and needs"""
    
    def __init__(self, ecosystem):
        self.ecosystem = ecosystem
        self.logger = ecosystem.logger
        self.benchmark_templates = self.load_benchmark_templates()
    
    def load_benchmark_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load benchmark templates for different categories"""
        return {
            "intelligence": {
                "logical_reasoning": {
                    "description": "Test logical reasoning and deduction skills",
                    "generator": self.generate_logical_reasoning_benchmark
                },
                "pattern_recognition": {
                    "description": "Test ability to recognize and extrapolate patterns",
                    "generator": self.generate_pattern_recognition_benchmark
                },
                "abstract_thinking": {
                    "description": "Test abstract thinking and conceptual understanding",
                    "generator": self.generate_abstract_thinking_benchmark
                }
            },
            "creativity": {
                "idea_generation": {
                    "description": "Test ability to generate novel ideas",
                    "generator": self.generate_idea_generation_benchmark
                },
                "creative_problem_solving": {
                    "description": "Test creative approaches to problem solving",
                    "generator": self.generate_creative_problem_solving_benchmark
                }
            },
            "problem_solving": {
                "mathematical_problems": {
                    "description": "Test mathematical problem solving abilities",
                    "generator": self.generate_mathematical_benchmark
                },
                "algorithmic_thinking": {
                    "description": "Test algorithmic thinking and optimization",
                    "generator": self.generate_algorithmic_benchmark
                }
            },
            "learning": {
                "concept_acquisition": {
                    "description": "Test ability to learn and apply new concepts",
                    "generator": self.generate_concept_acquisition_benchmark
                },
                "knowledge_transfer": {
                    "description": "Test ability to transfer knowledge between domains",
                    "generator": self.generate_knowledge_transfer_benchmark
                }
            }
        }
    
    async def generate_adaptive_benchmark(self, agent: AgentProfile, 
                                         target_capability: AgentCapability,
                                         difficulty: BenchmarkDifficulty = None) -> Dict[str, Any]:
        """Generate an adaptive benchmark based on agent's current capabilities"""
        
        if difficulty is None:
            difficulty = self.determine_appropriate_difficulty(agent, target_capability)
        
        # Select appropriate template
        category = self.map_capability_to_category(target_capability)
        templates = self.benchmark_templates.get(category.value, {})
        
        if not templates:
            template_name = list(templates.keys())[0]
        else:
            # Select template based on agent's performance history
            template_name = self.select_best_template(agent, category)
        
        template = templates[template_name]
        generator = template["generator"]
        
        # Generate benchmark
        benchmark = await generator(agent, target_capability, difficulty)
        
        return {
            "benchmark_id": f"adaptive_{int(time.time())}_{agent.agent_id[:8]}",
            "name": f"Adaptive {target_capability.value} Benchmark",
            "description": template["description"],
            "category": category,
            "difficulty": difficulty,
            "type": BenchmarkType.ADAPTIVE,
            "content": benchmark,
            "target_capability": target_capability.value,
            "estimated_duration": self.estimate_duration(difficulty),
            "scoring_criteria": self.generate_scoring_criteria(target_capability, difficulty)
        }
    
    def determine_appropriate_difficulty(self, agent: AgentProfile, 
                                        capability: AgentCapability) -> BenchmarkDifficulty:
        """Determine appropriate benchmark difficulty based on agent's capability score"""
        capability_score = agent.capabilities.get(capability, 0.5)
        
        if capability_score < 0.3:
            return BenchmarkDifficulty.BASIC
        elif capability_score < 0.6:
            return BenchmarkDifficulty.INTERMEDIATE
        elif capability_score < 0.8:
            return BenchmarkDifficulty.ADVANCED
        else:
            return BenchmarkDifficulty.EXPERT
    
    def map_capability_to_category(self, capability: AgentCapability) -> BenchmarkCategory:
        """Map agent capability to benchmark category"""
        mapping = {
            AgentCapability.REASONING: BenchmarkCategory.INTELLIGENCE,
            AgentCapability.CREATIVITY: BenchmarkCategory.CREATIVITY,
            AgentCapability.LEARNING: BenchmarkCategory.LEARNING,
            AgentCapability.ADAPTATION: BenchmarkCategory.ADAPTATION,
            AgentCapability.COLLABORATION: BenchmarkCategory.COLLABORATION,
            AgentCapability.PROBLEM_DECOMPOSITION: BenchmarkCategory.PROBLEM_SOLVING,
            AgentCapability.PATTERN_RECOGNITION: BenchmarkCategory.INTELLIGENCE,
            AgentCapability.SYSTEMS_THINKING: BenchmarkCategory.INTELLIGENCE,
            AgentCapability.ETHICAL_REASONING: BenchmarkCategory.ETHICAL_REASONING,
            AgentCapability.ABSTRACT_THINKING: BenchmarkCategory.INTELLIGENCE,
            AgentCapability.META_COGNITION: BenchmarkCategory.LEARNING,
            AgentCapability.SELF_AWARENESS: BenchmarkCategory.INTELLIGENCE
        }
        return mapping.get(capability, BenchmarkCategory.INTELLIGENCE)
    
    def select_best_template(self, agent: AgentProfile, category: BenchmarkCategory) -> str:
        """Select the best template based on agent's performance history"""
        # Simple selection - can be enhanced with ML-based selection
        templates = list(self.benchmark_templates.get(category.value, {}).keys())
        return templates[0] if templates else "default"
    
    async def generate_logical_reasoning_benchmark(self, agent: AgentProfile, 
                                                  capability: AgentCapability, 
                                                  difficulty: BenchmarkDifficulty) -> Dict[str, Any]:
        """Generate logical reasoning benchmark"""
        
        difficulty_levels = {
            BenchmarkDifficulty.BASIC: {
                "problems": [
                    {
                        "question": "If all roses are flowers and some flowers are red, can we conclude that some roses are red?",
                        "type": "syllogism",
                        "answer": False,
                        "explanation": "This is a classic case of the fallacy of the undistributed middle"
                    }
                ]
            },
            BenchmarkDifficulty.INTERMEDIATE: {
                "problems": [
                    {
                        "question": "In a group of 100 people, 70 speak English, 60 speak Spanish, and 40 speak both. How many speak neither?",
                        "type": "set_theory",
                        "answer": 10,
                        "explanation": "Using inclusion-exclusion: 70 + 60 - 40 = 90 speak at least one language, so 10 speak neither"
                    }
                ]
            },
            BenchmarkDifficulty.ADVANCED: {
                "problems": [
                    {
                        "question": "Three people (A, B, C) make statements. A says 'B is lying', B says 'C is lying', C says 'A and B are lying'. Who is telling the truth?",
                        "type": "logic_puzzle",
                        "answer": "C",
                        "explanation": "If A is truthful, then B is lying, which means C is truthful, but C says A is lying - contradiction. If A is lying, then B is truthful, which means C is lying, which is consistent with C's statement being false."
                    }
                ]
            }
        }
        
        return difficulty_levels.get(difficulty, difficulty_levels[BenchmarkDifficulty.BASIC])
    
    async def generate_pattern_recognition_benchmark(self, agent: AgentProfile, 
                                                  capability: AgentCapability, 
                                                  difficulty: BenchmarkDifficulty) -> Dict[str, Any]:
        """Generate pattern recognition benchmark"""
        
        patterns = {
            BenchmarkDifficulty.BASIC: {
                "sequences": [
                    {
                        "sequence": [2, 4, 6, 8, 10],
                        "pattern": "arithmetic progression +2",
                        "answer": 10
                    }
                ]
            },
            BenchmarkDifficulty.INTERMEDIATE: {
                "sequences": [
                    {
                        "sequence": [1, 1, 2, 3, 5, 8, 13],
                        "pattern": "Fibonacci sequence",
                        "answer": 13
                    }
                ]
            },
            BenchmarkDifficulty.ADVANCED: {
                "sequences": [
                    {
                        "sequence": [1, 4, 9, 16, 25, 36],
                        "pattern": "perfect squares",
                        "answer": 36
                    }
                ]
            }
        }
        
        return patterns.get(difficulty, patterns[BenchmarkDifficulty.BASIC])
    
    async def generate_abstract_thinking_benchmark(self, agent: AgentProfile, 
                                                  capability: AgentCapability, 
                                                  difficulty: BenchmarkDifficulty) -> Dict[str, Any]:
        """Generate abstract thinking benchmark"""
        
        abstract_problems = {
            BenchmarkDifficulty.BASIC: {
                "problems": [
                    {
                        "question": "If you have a square and you fold it in half diagonally, what shape do you get?",
                        "answer": "triangle",
                        "type": "spatial_reasoning"
                    }
                ]
            },
            BenchmarkDifficulty.INTERMEDIATE: {
                "problems": [
                    {
                        "question": "A bat and ball cost $1.10 together. The bat costs $1.00 more than the ball. How much does the ball cost?",
                        "answer": 0.05,
                        "type": "cognitive_reflection"
                    }
                ]
            }
        }
        
        return abstract_problems.get(difficulty, abstract_problems[BenchmarkDifficulty.BASIC])
    
    async def generate_idea_generation_benchmark(self, agent: AgentProfile, 
                                               capability: AgentCapability, 
                                               difficulty: BenchmarkDifficulty) -> Dict[str, Any]:
        """Generate idea generation benchmark"""
        
        creativity_tasks = {
            BenchmarkDifficulty.BASIC: {
                "task": "Generate 5 different uses for a paper clip",
                "evaluation_criteria": ["originality", "feasibility", "diversity"],
                "time_limit": 300  # 5 minutes
            },
            BenchmarkDifficulty.INTERMEDIATE: {
                "task": "Design a new type of renewable energy source",
                "evaluation_criteria": ["innovation", "practicality", "sustainability"],
                "time_limit": 600  # 10 minutes
            }
        }
        
        return creativity_tasks.get(difficulty, creativity_tasks[BenchmarkDifficulty.BASIC])
    
    async def generate_creative_problem_solving_benchmark(self, agent: AgentProfile, 
                                                         capability: AgentCapability, 
                                                         difficulty: BenchmarkDifficulty) -> Dict[str, Any]:
        """Generate creative problem solving benchmark"""
        
        problems = {
            BenchmarkDifficulty.BASIC: {
                "scenario": "You need to get a ping pong ball out of a long vertical pipe that's fixed to the floor.",
                "constraints": ["Cannot reach into pipe", "Cannot move pipe", "Limited materials available"],
                "evaluation_criteria": ["creativity", "effectiveness", "simplicity"]
            }
        }
        
        return problems.get(difficulty, problems[BenchmarkDifficulty.BASIC])
    
    async def generate_mathematical_benchmark(self, agent: AgentProfile, 
                                            capability: AgentCapability, 
                                            difficulty: BenchmarkDifficulty) -> Dict[str, Any]:
        """Generate mathematical benchmark"""
        
        math_problems = {
            BenchmarkDifficulty.BASIC: {
                "problems": [
                    {
                        "question": "Solve for x: 2x + 5 = 15",
                        "answer": 5,
                        "type": "linear_equation"
                    }
                ]
            },
            BenchmarkDifficulty.INTERMEDIATE: {
                "problems": [
                    {
                        "question": "Find the derivative of f(x) = x² + 3x - 2",
                        "answer": "2x + 3",
                        "type": "calculus"
                    }
                ]
            }
        }
        
        return math_problems.get(difficulty, math_problems[BenchmarkDifficulty.BASIC])
    
    async def generate_algorithmic_benchmark(self, agent: AgentProfile, 
                                           capability: AgentCapability, 
                                           difficulty: BenchmarkDifficulty) -> Dict[str, Any]:
        """Generate algorithmic thinking benchmark"""
        
        algorithmic_problems = {
            BenchmarkDifficulty.BASIC: {
                "problems": [
                    {
                        "question": "Write pseudocode to find the maximum number in an array",
                        "type": "algorithm_design",
                        "evaluation_criteria": ["correctness", "efficiency", "clarity"]
                    }
                ]
            }
        }
        
        return algorithmic_problems.get(difficulty, algorithmic_problems[BenchmarkDifficulty.BASIC])
    
    async def generate_concept_acquisition_benchmark(self, agent: AgentProfile, 
                                                   capability: AgentCapability, 
                                                   difficulty: BenchmarkDifficulty) -> Dict[str, Any]:
        """Generate concept acquisition benchmark"""
        
        learning_tasks = {
            BenchmarkDifficulty.BASIC: {
                "concept": "Machine Learning",
                "learning_material": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
                "test_questions": [
                    {
                        "question": "What is machine learning?",
                        "expected_answer": "A subset of AI that enables systems to learn from experience"
                    }
                ]
            }
        }
        
        return learning_tasks.get(difficulty, learning_tasks[BenchmarkDifficulty.BASIC])
    
    async def generate_knowledge_transfer_benchmark(self, agent: AgentProfile, 
                                                   capability: AgentCapability, 
                                                   difficulty: BenchmarkDifficulty) -> Dict[str, Any]:
        """Generate knowledge transfer benchmark"""
        
        transfer_tasks = {
            BenchmarkDifficulty.BASIC: {
                "source_domain": "Biology (evolution)",
                "target_domain": "Computer Science (optimization algorithms)",
                "task": "Explain how the concept of natural selection can be applied to genetic algorithms",
                "evaluation_criteria": ["accuracy", "depth_of_understanding", "application_quality"]
            }
        }
        
        return transfer_tasks.get(difficulty, transfer_tasks[BenchmarkDifficulty.BASIC])
    
    def estimate_duration(self, difficulty: BenchmarkDifficulty) -> float:
        """Estimate benchmark duration in minutes"""
        durations = {
            BenchmarkDifficulty.BASIC: 5,
            BenchmarkDifficulty.INTERMEDIATE: 15,
            BenchmarkDifficulty.ADVANCED: 30,
            BenchmarkDifficulty.EXPERT: 60
        }
        return durations.get(difficulty, 15)
    
    def generate_scoring_criteria(self, capability: AgentCapability, 
                                difficulty: BenchmarkDifficulty) -> Dict[str, Any]:
        """Generate scoring criteria for benchmark"""
        return {
            "accuracy_weight": 0.4,
            "efficiency_weight": 0.3,
            "creativity_weight": 0.2,
            "completeness_weight": 0.1,
            "difficulty_bonus": difficulty.value
        }


class BenchmarkingSystem:
    """Enhanced benchmarking system for the ecosystem"""
    
    def __init__(self, ecosystem):
        self.ecosystem = ecosystem
        self.logger = ecosystem.logger
        self.generator = BenchmarkGenerator(ecosystem)
        self.results: List[BenchmarkResult] = []
        self.suites: Dict[str, BenchmarkSuite] = {}
        self.profiles: Dict[str, BenchmarkProfile] = {}
        self.benchmark_history: List[Dict[str, Any]] = []
        self.running_benchmarks: Dict[str, asyncio.Task] = {}
        
        # Initialize standard benchmark suites
        self.initialize_standard_suites()
        
        # Make benchmarking available to ecosystem
        ecosystem.benchmarking_system = self
    
    def initialize_standard_suites(self):
        """Initialize standard benchmark suites"""
        
        # Intelligence Suite
        self.suites["intelligence_basic"] = BenchmarkSuite(
            suite_id="intelligence_basic",
            name="Basic Intelligence Assessment",
            description="Assesses fundamental intelligence capabilities",
            category=BenchmarkCategory.INTELLIGENCE,
            benchmarks=["logical_reasoning_basic", "pattern_recognition_basic"],
            weightings={"logical_reasoning_basic": 0.6, "pattern_recognition_basic": 0.4},
            difficulty_range=(BenchmarkDifficulty.BASIC, BenchmarkDifficulty.BASIC),
            estimated_duration=10
        )
        
        # Learning Suite
        self.suites["learning_comprehensive"] = BenchmarkSuite(
            suite_id="learning_comprehensive",
            name="Comprehensive Learning Assessment",
            description="Assesses learning and adaptation capabilities",
            category=BenchmarkCategory.LEARNING,
            benchmarks=["concept_acquisition_basic", "knowledge_transfer_basic"],
            weightings={"concept_acquisition_basic": 0.7, "knowledge_transfer_basic": 0.3},
            difficulty_range=(BenchmarkDifficulty.BASIC, BenchmarkDifficulty.INTERMEDIATE),
            estimated_duration=20
        )
        
        # Problem Solving Suite
        self.suites["problem_solving"] = BenchmarkSuite(
            suite_id="problem_solving",
            name="Problem Solving Assessment",
            description="Assesses problem solving capabilities",
            category=BenchmarkCategory.PROBLEM_SOLVING,
            benchmarks=["mathematical_basic", "algorithmic_basic"],
            weightings={"mathematical_basic": 0.5, "algorithmic_basic": 0.5},
            difficulty_range=(BenchmarkDifficulty.BASIC, BenchmarkDifficulty.INTERMEDIATE),
            estimated_duration=15
        )
    
    async def run_adaptive_benchmark(self, agent: AgentProfile, 
                                   target_capability: AgentCapability = None) -> BenchmarkResult:
        """Run an adaptive benchmark for an agent"""
        
        if target_capability is None:
            # Target agent's weakest capability
            weakest_caps = sorted(agent.capabilities.items(), key=lambda x: x[1])
            target_capability = weakest_caps[0][0] if weakest_caps else AgentCapability.REASONING
        
        # Generate adaptive benchmark
        benchmark_def = await self.generator.generate_adaptive_benchmark(
            agent, target_capability
        )
        
        # Execute benchmark
        result = await self.execute_benchmark(agent, benchmark_def)
        
        # Store result
        self.results.append(result)
        
        # Update agent profile
        await self.update_agent_profile(agent, result)
        
        return result
    
    async def execute_benchmark(self, agent: AgentProfile, 
                              benchmark_def: Dict[str, Any]) -> BenchmarkResult:
        """Execute a benchmark for an agent"""
        
        start_time = time.time()
        benchmark_id = benchmark_def["benchmark_id"]
        
        try:
            # Prepare benchmark context
            context = {
                "agent": agent,
                "benchmark": benchmark_def,
                "ecosystem": self.ecosystem
            }
            
            # Execute benchmark based on type
            if benchmark_def["type"] == BenchmarkType.ADAPTIVE:
                score, metrics = await self.execute_adaptive_benchmark(context)
            else:
                score, metrics = await self.execute_standard_benchmark(context)
            
            execution_time = time.time() - start_time
            
            # Create result
            result = BenchmarkResult(
                benchmark_id=benchmark_id,
                agent_id=agent.agent_id,
                category=benchmark_def["category"],
                difficulty=benchmark_def["difficulty"],
                score=score,
                max_score=1.0,
                time_taken=execution_time,
                success=True,
                metrics=metrics,
                execution_details={"benchmark_type": benchmark_def["type"].value}
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            return BenchmarkResult(
                benchmark_id=benchmark_id,
                agent_id=agent.agent_id,
                category=benchmark_def["category"],
                difficulty=benchmark_def["difficulty"],
                score=0.0,
                max_score=1.0,
                time_taken=execution_time,
                success=False,
                error_message=str(e)
            )
    
    async def execute_adaptive_benchmark(self, context: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Execute adaptive benchmark"""
        
        agent = context["agent"]
        benchmark = context["benchmark"]
        content = benchmark["content"]
        
        # Use the reasoning model to solve the benchmark
        reasoning_prompt = f"""
        You are an AI agent being tested on your {benchmark['target_capability']} capability.
        
        Benchmark: {benchmark['name']}
        Description: {benchmark['description']}
        Difficulty: {benchmark['difficulty'].value}
        
        Task: {json.dumps(content, indent=2)}
        
        Please solve this problem step by step. Provide your final answer in JSON format with:
        - answer: your solution
        - reasoning: step-by-step reasoning
        - confidence: your confidence level (0-1)
        """
        
        try:
            response = await self.ecosystem.models["reasoning"].agenerate(reasoning_prompt)
            
            # Extract and evaluate answer
            answer_data = self.extract_answer_from_response(response.content)
            score = self.evaluate_adaptive_answer(answer_data, content)
            
            metrics = {
                "confidence": answer_data.get("confidence", 0.5),
                "reasoning_quality": self.evaluate_reasoning_quality(answer_data.get("reasoning", "")),
                "response_length": len(response.content)
            }
            
            return score, metrics
            
        except Exception as e:
            self.logger.error(f"Error executing adaptive benchmark: {e}")
            return 0.0, {"error": str(e)}
    
    async def execute_standard_benchmark(self, context: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Execute standard benchmark"""
        
        # For now, delegate to adaptive execution
        # This can be enhanced with specific standard benchmark implementations
        return await self.execute_adaptive_benchmark(context)
    
    def extract_answer_from_response(self, response: str) -> Dict[str, Any]:
        """Extract structured answer from model response"""
        try:
            # Try to parse JSON from response
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
                return json.loads(json_str)
            else:
                # Fallback: extract key information
                return {
                    "answer": response,
                    "reasoning": response,
                    "confidence": 0.5
                }
        except Exception:
            return {
                "answer": response,
                "reasoning": response,
                "confidence": 0.5
            }
    
    def evaluate_adaptive_answer(self, answer_data: Dict[str, Any], 
                               benchmark_content: Dict[str, Any]) -> float:
        """Evaluate adaptive benchmark answer"""
        
        # This is a simplified evaluation - can be enhanced with more sophisticated scoring
        score = 0.0
        
        # Check if answer is provided
        if "answer" in answer_data and answer_data["answer"]:
            score += 0.4
        
        # Check reasoning quality
        if "reasoning" in answer_data and len(answer_data["reasoning"]) > 50:
            score += 0.3
        
        # Check confidence
        confidence = answer_data.get("confidence", 0.5)
        score += confidence * 0.3
        
        return min(1.0, score)
    
    def evaluate_reasoning_quality(self, reasoning: str) -> float:
        """Evaluate the quality of reasoning"""
        if not reasoning:
            return 0.0
        
        # Simple heuristics for reasoning quality
        quality_indicators = [
            "therefore", "because", "since", "thus", "hence",
            "first", "second", "finally", "conclusion"
        ]
        
        indicator_count = sum(1 for indicator in quality_indicators 
                            if indicator in reasoning.lower())
        
        return min(1.0, indicator_count / 5.0)
    
    async def update_agent_profile(self, agent: AgentProfile, result: BenchmarkResult):
        """Update agent's benchmarking profile"""
        
        agent_id = agent.agent_id
        
        # Get or create profile
        if agent_id not in self.profiles:
            self.profiles[agent_id] = BenchmarkProfile(
                agent_id=agent_id,
                total_benchmarks=0,
                completed_benchmarks=0,
                average_score=0.0,
                category_scores={},
                difficulty_scores={},
                improvement_trend=0.0,
                strengths=[],
                weaknesses=[]
            )
        
        profile = self.profiles[agent_id]
        
        # Update basic metrics
        profile.total_benchmarks += 1
        if result.success:
            profile.completed_benchmarks += 1
        
        # Update category scores
        if result.category not in profile.category_scores:
            profile.category_scores[result.category] = []
        profile.category_scores[result.category].append(result.score)
        
        # Update difficulty scores
        if result.difficulty not in profile.difficulty_scores:
            profile.difficulty_scores[result.difficulty] = []
        profile.difficulty_scores[result.difficulty].append(result.score)
        
        # Calculate average score
        all_scores = []
        for scores in profile.category_scores.values():
            all_scores.extend(scores)
        profile.average_score = statistics.mean(all_scores) if all_scores else 0.0
        
        # Calculate improvement trend
        profile.improvement_trend = self.calculate_improvement_trend(agent_id)
        
        # Identify strengths and weaknesses
        profile.strengths, profile.weaknesses = self.identify_strengths_weaknesses(profile)
        
        profile.last_updated = datetime.now()
    
    def calculate_improvement_trend(self, agent_id: str) -> float:
        """Calculate improvement trend for an agent"""
        agent_results = [r for r in self.results if r.agent_id == agent_id]
        
        if len(agent_results) < 2:
            return 0.0
        
        # Sort by timestamp
        agent_results.sort(key=lambda x: x.timestamp)
        
        # Calculate trend using simple linear regression
        recent_results = agent_results[-10:]  # Last 10 results
        if len(recent_results) < 2:
            return 0.0
        
        x_values = list(range(len(recent_results)))
        y_values = [r.score for r in recent_results]
        
        # Simple trend calculation
        trend = (y_values[-1] - y_values[0]) / len(y_values) if len(y_values) > 1 else 0.0
        
        return trend
    
    def identify_strengths_weaknesses(self, profile: BenchmarkProfile) -> Tuple[List[str], List[str]]:
        """Identify agent's strengths and weaknesses"""
        
        strengths = []
        weaknesses = []
        
        # Analyze category scores
        for category, scores in profile.category_scores.items():
            if scores:
                avg_score = statistics.mean(scores)
                if avg_score > 0.7:
                    strengths.append(f"{category.value} (score: {avg_score:.3f})")
                elif avg_score < 0.4:
                    weaknesses.append(f"{category.value} (score: {avg_score:.3f})")
        
        # Analyze difficulty performance
        for difficulty, scores in profile.difficulty_scores.items():
            if scores:
                avg_score = statistics.mean(scores)
                if avg_score > 0.7:
                    strengths.append(f"{difficulty.value} difficulty (score: {avg_score:.3f})")
                elif avg_score < 0.4:
                    weaknesses.append(f"{difficulty.value} difficulty (score: {avg_score:.3f})")
        
        return strengths, weaknesses
    
    async def run_benchmark_suite(self, agent: AgentProfile, suite_id: str) -> Dict[str, Any]:
        """Run a complete benchmark suite for an agent"""
        
        if suite_id not in self.suites:
            raise ValueError(f"Benchmark suite {suite_id} not found")
        
        suite = self.suites[suite_id]
        results = []
        
        self.logger.info(f"Running benchmark suite '{suite.name}' for agent {agent.name}")
        
        # Run each benchmark in the suite
        for benchmark_id in suite.benchmarks:
            try:
                # Generate benchmark definition
                benchmark_def = await self.generator.generate_adaptive_benchmark(
                    agent, 
                    self.map_suite_to_capability(suite.category),
                    suite.difficulty_range[0]  # Use minimum difficulty
                )
                benchmark_def["benchmark_id"] = benchmark_id
                
                # Execute benchmark
                result = await self.execute_benchmark(agent, benchmark_def)
                results.append(result)
                
                # Update agent profile
                await self.update_agent_profile(agent, result)
                
            except Exception as e:
                self.logger.error(f"Error running benchmark {benchmark_id}: {e}")
        
        # Calculate suite score
        suite_score = self.calculate_suite_score(results, suite.weightings)
        
        suite_result = {
            "suite_id": suite_id,
            "suite_name": suite.name,
            "agent_id": agent.agent_id,
            "suite_score": suite_score,
            "individual_results": [asdict(r) for r in results],
            "execution_time": sum(r.time_taken for r in results),
            "timestamp": datetime.now().isoformat()
        }
        
        # Record suite execution
        self.benchmark_history.append(suite_result)
        
        return suite_result
    
    def map_suite_to_capability(self, category: BenchmarkCategory) -> AgentCapability:
        """Map benchmark suite category to agent capability"""
        mapping = {
            BenchmarkCategory.INTELLIGENCE: AgentCapability.REASONING,
            BenchmarkCategory.CREATIVITY: AgentCapability.CREATIVITY,
            BenchmarkCategory.LEARNING: AgentCapability.LEARNING,
            BenchmarkCategory.PROBLEM_SOLVING: AgentCapability.PROBLEM_DECOMPOSITION,
            BenchmarkCategory.COLLABORATION: AgentCapability.COLLABORATION,
            BenchmarkCategory.ADAPTATION: AgentCapability.ADAPTATION,
            BenchmarkCategory.ETHICAL_REASONING: AgentCapability.ETHICAL_REASONING
        }
        return mapping.get(category, AgentCapability.REASONING)
    
    def calculate_suite_score(self, results: List[BenchmarkResult], 
                             weightings: Dict[str, float]) -> float:
        """Calculate weighted score for a benchmark suite"""
        
        if not results:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for result in results:
            weight = weightings.get(result.benchmark_id, 1.0)
            total_score += result.score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    async def generate_ecosystem_benchmark_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report for the ecosystem"""
        
        # Calculate ecosystem-level metrics
        total_benchmarks = len(self.results)
        successful_benchmarks = len([r for r in self.results if r.success])
        overall_success_rate = successful_benchmarks / total_benchmarks if total_benchmarks > 0 else 0.0
        
        # Category performance
        category_performance = {}
        for category in BenchmarkCategory:
            category_results = [r for r in self.results if r.category == category]
            if category_results:
                category_performance[category.value] = {
                    "count": len(category_results),
                    "average_score": statistics.mean([r.score for r in category_results]),
                    "success_rate": len([r for r in category_results if r.success]) / len(category_results)
                }
        
        # Agent performance summary
        agent_performance = {}
        for agent_id, profile in self.profiles.items():
            agent_performance[agent_id] = {
                "total_benchmarks": profile.total_benchmarks,
                "completed_benchmarks": profile.completed_benchmarks,
                "average_score": profile.average_score,
                "improvement_trend": profile.improvement_trend,
                "strengths": profile.strengths,
                "weaknesses": profile.weaknesses
            }
        
        # Recent trends
        recent_results = [r for r in self.results 
                         if (datetime.now() - r.timestamp).total_seconds() < 86400]  # Last 24 hours
        recent_trend = statistics.mean([r.score for r in recent_results]) if recent_results else 0.0
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_benchmarks": total_benchmarks,
                "successful_benchmarks": successful_benchmarks,
                "overall_success_rate": overall_success_rate,
                "recent_trend": recent_trend
            },
            "category_performance": category_performance,
            "agent_performance": agent_performance,
            "benchmark_suites": len(self.suites),
            "active_profiles": len(self.profiles)
        }
        
        return report
    
    async def benchmark_ecosystem_evolution(self) -> Dict[str, Any]:
        """Benchmark ecosystem evolution and provide insights"""
        
        # Get current ecosystem state
        current_fitness = await self.ecosystem.evolution_system.calculate_ecosystem_fitness()
        
        # Run benchmarks for all agents
        agent_benchmarks = {}
        for agent_id, agent in self.ecosystem.state.agents.items():
            try:
                result = await self.run_adaptive_benchmark(agent)
                agent_benchmarks[agent_id] = {
                    "score": result.score,
                    "category": result.category.value,
                    "improvement_needed": result.score < 0.6
                }
            except Exception as e:
                self.logger.error(f"Error benchmarking agent {agent_id}: {e}")
        
        # Calculate ecosystem benchmark score
        ecosystem_benchmark_score = statistics.mean([
            bench["score"] for bench in agent_benchmarks.values()
        ]) if agent_benchmarks else 0.0
        
        # Generate evolution insights
        evolution_insights = self.generate_evolution_insights(
            current_fitness, ecosystem_benchmark_score, agent_benchmarks
        )
        
        return {
            "timestamp": datetime.now().isoformat(),
            "ecosystem_fitness": current_fitness,
            "ecosystem_benchmark_score": ecosystem_benchmark_score,
            "agent_benchmarks": agent_benchmarks,
            "evolution_insights": evolution_insights,
            "recommendations": self.generate_evolution_recommendations(evolution_insights)
        }
    
    def generate_evolution_insights(self, fitness: float, benchmark_score: float, 
                                  agent_benchmarks: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights about ecosystem evolution"""
        
        insights = {
            "fitness_benchmark_correlation": fitness - benchmark_score,
            "overall_health": "good" if fitness > 0.7 and benchmark_score > 0.7 else "needs_attention",
            "improvement_areas": [],
            "strength_areas": []
        }
        
        # Identify improvement areas
        weak_agents = [agent_id for agent_id, bench in agent_benchmarks.items() 
                      if bench["score"] < 0.5]
        if weak_agents:
            insights["improvement_areas"].append(f"Weak agents: {len(weak_agents)}")
        
        # Identify strength areas
        strong_agents = [agent_id for agent_id, bench in agent_benchmarks.items() 
                        if bench["score"] > 0.8]
        if strong_agents:
            insights["strength_areas"].append(f"Strong agents: {len(strong_agents)}")
        
        # Check correlation between fitness and benchmark scores
        if abs(insights["fitness_benchmark_correlation"]) > 0.2:
            insights["correlation_analysis"] = "Significant correlation detected"
        else:
            insights["correlation_analysis"] = "Low correlation - may need alignment"
        
        return insights
    
    def generate_evolution_recommendations(self, insights: Dict[str, Any]) -> List[str]:
        """Generate recommendations for ecosystem evolution"""
        
        recommendations = []
        
        if insights["overall_health"] == "needs_attention":
            recommendations.append("Focus on improving overall ecosystem health")
        
        if "Weak agents" in str(insights["improvement_areas"]):
            recommendations.append("Implement targeted training for underperforming agents")
        
        if "Low correlation" in str(insights.get("correlation_analysis", "")):
            recommendations.append("Align fitness metrics with benchmark performance")
        
        if not insights["strength_areas"]:
            recommendations.append("Develop agent specialization strategies")
        
        return recommendations
    
    def get_benchmarking_status(self) -> Dict[str, Any]:
        """Get current benchmarking system status"""
        return {
            "total_results": len(self.results),
            "active_profiles": len(self.profiles),
            "available_suites": len(self.suites),
            "running_benchmarks": len(self.running_benchmarks),
            "recent_executions": len([
                r for r in self.results 
                if (datetime.now() - r.timestamp).total_seconds() < 3600
            ])
        }