import asyncio
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from collections import defaultdict

from systemd import AgentRole, AgentCapability, AgentProfile, EcosystemState
from evoagentx.rag.schema import Query, Corpus, Chunk, ChunkMetadata

class AgentOperations:
    
    def __init__(self, ecosystem):
        self.ecosystem = ecosystem
        self.logger = ecosystem.logger
        self.models = ecosystem.models
        self.rag_engines = ecosystem.rag_engines
        self.tools = ecosystem.tools
        
        self.mcp_integration = getattr(ecosystem, 'mcp_integration', None)
        self.benchmarking_system = getattr(ecosystem, 'benchmarking_system', None)
        
    async def agent_self_improvement(self, agent: AgentProfile):
        try:
            if self.mcp_integration:
                await self.mcp_enhanced_self_improvement(agent)
            
            if agent.role == AgentRole.META_LEARNING:
                await self.meta_learning_improvement(agent)
            elif agent.role == AgentRole.SYSTEM_OPTIMIZATION:
                await self.system_optimization_improvement(agent)
            elif agent.role == AgentRole.ETHICAL_OVERSIGHT:
                await self.ethical_reasoning_improvement(agent)
            elif agent.role == AgentRole.KNOWLEDGE_ACQUISITION:
                await self.knowledge_acquisition_improvement(agent)
            else:
                await self.general_capability_improvement(agent)
                
            agent.evolution_history.append({
                "event": "self_improvement",
                "timestamp": datetime.now().isoformat(),
                "improvements": "role_specific_enhancement",
                "energy_cost": 0.02
            })
            
        except Exception as e:
            self.logger.error(f"Error in agent self-improvement for {agent.agent_id}: {e}")
    
    async def mcp_enhanced_self_improvement(self, agent: AgentProfile):
        if not self.mcp_integration:
            return
        
        try:
            needs_analysis = await self.analyze_agent_needs(agent)
            
            for need in needs_analysis.get("tool_needs", []):
                tool_id = await self.mcp_integration.create_tool_from_need(
                    agent, need["description"], need.get("requirements", {})
                )
                
                tool_result = await self.mcp_integration.registry.execute_tool(
                    tool_id,
                    agent_context={
                        "agent_id": agent.agent_id,
                        "capabilities": {cap.value: score for cap, score in agent.capabilities.items()},
                        "knowledge_domains": agent.knowledge_domains
                    }
                )
                
                await self.apply_tool_results_to_improvement(agent, tool_result, need)
                
        except Exception as e:
            self.logger.error(f"Error in MCP-enhanced self-improvement for {agent.agent_id}: {e}")
    
    async def analyze_agent_needs(self, agent: AgentProfile) -> Dict[str, Any]:
        
        weakest_caps = sorted(agent.capabilities.items(), key=lambda x: x[1])[:3]
        
        tool_needs = []
        
        for capability, score in weakest_caps:
            if score < 0.6:
                need_description = f"A tool to improve {capability.value} capability for agents"
                requirements = {
                    "target_capability": capability.value,
                    "current_score": score,
                    "target_score": min(1.0, score + 0.2)
                }
                tool_needs.append({
                    "description": need_description,
                    "requirements": requirements
                })
        
        return {"tool_needs": tool_needs}
    
    async def apply_tool_results_to_improvement(self, agent: AgentProfile, tool_result: Any, need: Dict[str, Any]):
        
        if isinstance(tool_result, dict):
            improvements = tool_result.get("improvements", {})
            
            for capability_name, improvement_value in improvements.items():
                try:
                    capability = AgentCapability(capability_name)
                    current_score = agent.capabilities.get(capability, 0.5)
                    new_score = min(1.0, current_score + improvement_value)
                    agent.update_capability(capability, new_score)
                    
                except ValueError:
                    continue
    
    async def meta_learning_improvement(self, agent: AgentProfile):
        improvement_prompt = f"""
        As a meta-learning agent, analyze the ecosystem's learning patterns and suggest improvements.
        
        Current ecosystem state:
        - Agent count: {len(self.ecosystem.state.agents)}
        - Collective intelligence: {self.ecosystem.state.system_intelligence_score:.3f}
        - Recent learning activities: {len([h for h in agent.performance_history[-10:] if 'learning' in str(h)])}
        
        Analyze and provide:
        1. Learning efficiency metrics
        2. Knowledge transfer effectiveness
        3. Adaptive learning strategies
        4. Meta-cognitive improvements
        
        Focus on actionable insights for improving collective learning.
        """
        
        try:
            response = await self.models["learning"].agenerate(improvement_prompt)
            
            insights = self.extract_learning_insights(response.content)
            
            current_meta = agent.capabilities.get(AgentCapability.META_COGNITION, 0.5)
            improvement = min(0.02, insights.get("meta_improvement", 0.01))
            agent.update_capability(AgentCapability.META_COGNITION, 
                                 min(1.0, current_meta + improvement))
            
            current_learning = agent.capabilities.get(AgentCapability.LEARNING, 0.5)
            agent.update_capability(AgentCapability.LEARNING,
                                 min(1.0, current_learning + improvement * 0.8))
            
            agent.performance_history.append({
                "task": "meta_learning_improvement",
                "insights": insights,
                "timestamp": datetime.now().isoformat(),
                "success": True
            })
            
        except Exception as e:
            self.logger.error(f"Error in meta-learning improvement: {e}")
    
    async def system_optimization_improvement(self, agent: AgentProfile):
        optimization_prompt = f"""
        As a system optimization agent, analyze current system performance and identify improvements.
        
        Current metrics:
        - Resource usage: {self.ecosystem.state.resource_usage}
        - Agent efficiency: {self.calculate_agent_efficiency()}
        - System health: {self.ecosystem.state.system_health_score:.3f}
        - Task completion rate: {self.calculate_task_completion_rate()}
        
        Provide specific recommendations for:
        1. Resource allocation optimization
        2. Agent workload balancing
        3. System architecture improvements
        4. Performance bottleneck resolution
        
        Include quantitative metrics and implementation strategies.
        """
        
        try:
            response = await self.models["optimization"].agenerate(optimization_prompt)
            
            optimizations = self.extract_optimization_strategies(response.content)
            await self.apply_system_optimizations(optimizations)
            
            current_systems = agent.capabilities.get(AgentCapability.SYSTEMS_THINKING, 0.5)
            agent.update_capability(AgentCapability.SYSTEMS_THINKING,
                                 min(1.0, current_systems + 0.015))
            
            agent.performance_history.append({
                "task": "system_optimization",
                "optimizations": optimizations,
                "timestamp": datetime.now().isoformat(),
                "success": True
            })
            
        except Exception as e:
            self.logger.error(f"Error in system optimization improvement: {e}")
    
    async def ethical_reasoning_improvement(self, agent: AgentProfile):
        ethics_prompt = f"""
        As an ethical oversight agent, evaluate the ecosystem's ethical compliance and reasoning.
        
        Current ethical state:
        - Ethical constraints: {len(self.ecosystem.state.ethical_constraints)}
        - Recent violations: {self.ecosystem.state.emergency_count}
        - Agent trust scores: {[a.trust_score for a in list(self.ecosystem.state.agents.values())[:5]]}
        
        Analyze and improve:
        1. Ethical decision-making frameworks
        2. Bias detection and mitigation
        3. Value alignment strategies
        4. Ethical constraint effectiveness
        
        Provide concrete recommendations for ethical enhancement.
        """
        
        try:
            response = await self.models["reasoning"].agenerate(ethics_prompt)
            
            ethical_insights = self.extract_ethical_insights(response.content)
            await self.apply_ethical_improvements(ethical_insights)
            
            current_ethical = agent.capabilities.get(AgentCapability.ETHICAL_REASONING, 0.5)
            agent.update_capability(AgentCapability.ETHICAL_REASONING,
                                 min(1.0, current_ethical + 0.02))
            
            current_awareness = agent.capabilities.get(AgentCapability.SELF_AWARENESS, 0.5)
            agent.update_capability(AgentCapability.SELF_AWARENESS,
                                 min(1.0, current_awareness + 0.01))
            
            agent.performance_history.append({
                "task": "ethical_reasoning_improvement",
                "insights": ethical_insights,
                "timestamp": datetime.now().isoformat(),
                "success": True
            })
            
        except Exception as e:
            self.logger.error(f"Error in ethical reasoning improvement: {e}")
    
    async def knowledge_acquisition_improvement(self, agent: AgentProfile):
        knowledge_prompt = f"""
        As a knowledge acquisition agent, identify new knowledge domains and acquisition strategies.
        
        Current knowledge state:
        - Total knowledge domains: {len(set().union(*[a.knowledge_domains for a in self.ecosystem.state.agents.values()]))}
        - Agent knowledge domains: {len(agent.knowledge_domains)}
        - Recent acquisitions: {len([h for h in agent.performance_history[-5:] if 'knowledge' in str(h)])}
        
        Identify and prioritize:
        1. Emerging knowledge domains
        2. Knowledge gaps in the ecosystem
        3. Efficient acquisition strategies
        4. Cross-domain knowledge synthesis
        
        Focus on high-impact knowledge areas.
        """
        
        try:
            response = await self.models["learning"].agenerate(knowledge_prompt)
            
            knowledge_ops = self.extract_knowledge_opportunities(response.content)
            await self.pursue_knowledge_opportunities(agent, knowledge_ops)
            
            current_learning = agent.capabilities.get(AgentCapability.LEARNING, 0.5)
            agent.update_capability(AgentCapability.LEARNING,
                                 min(1.0, current_learning + 0.02))
            
            current_pattern = agent.capabilities.get(AgentCapability.PATTERN_RECOGNITION, 0.5)
            agent.update_capability(AgentCapability.PATTERN_RECOGNITION,
                                 min(1.0, current_pattern + 0.015))
            
            agent.performance_history.append({
                "task": "knowledge_acquisition_improvement",
                "opportunities": knowledge_ops,
                "timestamp": datetime.now().isoformat(),
                "success": True
            })
            
        except Exception as e:
            self.logger.error(f"Error in knowledge acquisition improvement: {e}")
    
    async def general_capability_improvement(self, agent: AgentProfile):
        sorted_caps = sorted(agent.capabilities.items(), key=lambda x: x[1])
        weakest_caps = sorted_caps[:3]
        
        for capability, current_score in weakest_caps:
            if current_score < 0.8:
                improvement = agent.self_improvement_rate * random.uniform(0.5, 1.5)
                new_score = min(1.0, current_score + improvement)
                agent.update_capability(capability, new_score)
        
        strongest_cap = max(agent.capabilities.items(), key=lambda x: x[1])
        if strongest_cap[1] < 0.95:
            improvement = agent.self_improvement_rate * 0.5
            new_score = min(1.0, strongest_cap[1] + improvement)
            agent.update_capability(strongest_cap[0], new_score)
    
    async def facilitate_collaboration(self, agents: List[AgentProfile], task: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced collaboration with HITL oversight and tool integration"""
        try:
            # HITL approval for multi-agent collaboration
            if hasattr(self.ecosystem, 'hitl_system'):
                approval_request = {
                    "action": "multi_agent_collaboration",
                    "agents": [agent.agent_id for agent in agents],
                    "task": task,
                    "risk_level": "high" if len(agents) > 5 else "medium"
                }
                approved = await self.ecosystem.hitl_system.request_approval(
                    "coordination", approval_request
                )
                if not approved:
                    return {"success": False, "reason": "HITL approval denied for collaboration"}
            
            collaboration_id = str(uuid.uuid4())
            
            collaboration_data = {
                "collaboration_id": collaboration_id,
                "agents": [agent.agent_id for agent in agents],
                "task": task,
                "timestamp": datetime.now().isoformat()
            }
            
            # Provide tools to agents for collaboration
            if hasattr(self.ecosystem, 'tool_integration'):
                for agent in agents:
                    recommended_tools = await self.ecosystem.tool_integration.get_tools_for_agent(agent)
                    agent.available_tools = recommended_tools
            
            # Calculate collaboration compatibility
            compatibility_matrix = await self.calculate_collaboration_compatibility(agents)
            collaboration_data["compatibility_matrix"] = compatibility_matrix
            
            # Optimize agent roles for collaboration
            role_assignments = await self.optimize_collaboration_roles(agents, task, compatibility_matrix)
            collaboration_data["role_assignments"] = role_assignments
            
            # Execute collaborative task with tool support
            results = await self.execute_collaborative_task_with_tools(agents, task, role_assignments)
            collaboration_data["results"] = results
            
            # Evaluate collaboration with evaluation pipeline
            if hasattr(self.ecosystem, 'evaluation_pipeline'):
                collaboration_evaluation = await self.ecosystem.evaluation_pipeline.evaluate_collaboration(
                    agents, task, results
                )
                collaboration_data["evaluation"] = collaboration_evaluation
            
            # Calculate collaboration score
            collaboration_score = await self.calculate_collaboration_score(agents, results)
            collaboration_data["collaboration_score"] = collaboration_score
            
            # Update agent collaboration metrics
            for agent in agents:
                agent.collaboration_count += 1
                agent.total_collaboration_score += collaboration_score
                agent.avg_collaboration_score = agent.total_collaboration_score / agent.collaboration_count
            
            # Store collaboration history
            self.ecosystem.state.collaboration_history.append(collaboration_data)
            
            self.logger.info(f"Collaboration {collaboration_id} completed with score: {collaboration_score}")
            return collaboration_data
            
        except Exception as e:
            self.logger.error(f"Collaboration failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def apply_collaboration_tool_results(self, agent: AgentProfile, tool_result: Any):
        
        if isinstance(tool_result, dict):
            enhancements = tool_result.get("collaboration_enhancements", {})
            
            if "collaboration_boost" in enhancements:
                current_collab = agent.capabilities.get(AgentCapability.COLLABORATION, 0.5)
                boost = enhancements["collaboration_boost"]
                new_score = min(1.0, current_collab + boost)
                agent.update_capability(AgentCapability.COLLABORATION, new_score)
            
            if "communication_boost" in enhancements:
                communication_boost = enhancements["communication_boost"]
                for cap in [AgentCapability.REASONING, AgentCapability.ADAPTATION]:
                    current_score = agent.capabilities.get(cap, 0.5)
                    new_score = min(1.0, current_score + communication_boost * 0.5)
                    agent.update_capability(cap, new_score)
    
    async def benchmark_enhanced_improvement(self, agent: AgentProfile):
        if not self.benchmarking_system:
            return
        
        try:
            weakest_caps = sorted(agent.capabilities.items(), key=lambda x: x[1])
            if weakest_caps:
                target_capability = weakest_caps[0][0]
                
                benchmark_result = await self.benchmarking_system.run_adaptive_benchmark(
                    agent, target_capability
                )
                
                await self.apply_benchmark_insights(agent, benchmark_result)
                
        except Exception as e:
            self.logger.error(f"Error in benchmark-enhanced improvement for {agent.agent_id}: {e}")
    
    async def apply_benchmark_insights(self, agent: AgentProfile, benchmark_result):
        
        if not benchmark_result.success:
            return
        
        profile = self.benchmarking_system.profiles.get(agent.agent_id)
        if not profile:
            return
        
        for weakness in profile.weaknesses:
            for capability in AgentCapability:
                if capability.value in weakness.lower():
                    current_score = agent.capabilities.get(capability, 0.5)
                    
                    benchmark_score = benchmark_result.score
                    improvement_factor = (1.0 - benchmark_score) * 0.1
                    new_score = min(1.0, current_score + improvement_factor)
                    
                    agent.update_capability(capability, new_score)
                    break
        
        agent.evolution_history.append({
            "event": "benchmark_improvement",
            "timestamp": datetime.now().isoformat(),
            "benchmark_id": benchmark_result.benchmark_id,
            "benchmark_score": benchmark_result.score,
            "improvements": "benchmark_guided_enhancement",
            "energy_cost": 0.03
        })
    
    async def self_improve_agent(self, agent: AgentProfile, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced self-improvement with optimization and HITL integration"""
        try:
            # Check HITL approval for self-improvement
            if hasattr(self.ecosystem, 'hitl_system'):
                approval_request = {
                    "action": "agent_self_improvement",
                    "agent_id": agent.agent_id,
                    "context": context,
                    "risk_level": "medium"
                }
                approved = await self.ecosystem.hitl_system.request_approval(
                    "system_optimization", approval_request
                )
                if not approved:
                    return {"success": False, "reason": "HITL approval denied"}
            
            improvement_data = {
                "agent_id": agent.agent_id,
                "timestamp": datetime.now().isoformat(),
                "context": context
            }
            
            # Get current performance baseline
            if hasattr(self.ecosystem, 'benchmarking_system'):
                baseline_results = await self.ecosystem.benchmarking_system.run_agent_benchmark(
                    agent, context.get("benchmark_type", "general")
                )
                improvement_data["baseline_performance"] = baseline_results
            
            # Use optimization system for improvement
            if hasattr(self.ecosystem, 'optimization_system'):
                optimization_result = await self.ecosystem.optimization_system.optimize_agent(
                    agent, strategy="hybrid", context=context
                )
                improvement_data["optimization_result"] = optimization_result
                
                # Apply optimization improvements
                if optimization_result.get("success"):
                    await self.apply_optimization_improvements(agent, optimization_result)
            else:
                # Fallback to legacy improvement
                performance_gaps = await self.analyze_performance_gaps(agent, context)
                improvement_data["performance_gaps"] = performance_gaps
                
                strategies = await self.generate_improvement_strategies(agent, performance_gaps)
                improvement_data["strategies"] = strategies
                
                for strategy in strategies:
                    await self.apply_improvement_strategy(agent, strategy)
            
            # Validate improvements with evaluation pipeline
            if hasattr(self.ecosystem, 'evaluation_pipeline'):
                evaluation_result = await self.ecosystem.evaluation_pipeline.evaluate_agent(
                    agent, context.get("evaluation_criteria", ["performance", "quality"])
                )
                improvement_data["evaluation_result"] = evaluation_result
            
            # Post-improvement benchmarking
            if hasattr(self.ecosystem, 'benchmarking_system'):
                post_results = await self.ecosystem.benchmarking_system.run_agent_benchmark(
                    agent, context.get("benchmark_type", "general")
                )
                improvement_data["post_improvement_performance"] = post_results
                improvement_data["improvement_delta"] = self.calculate_improvement_delta(
                    baseline_results, post_results
                )
            
            # Update agent state
            agent.last_improvement = datetime.now()
            agent.improvement_count += 1
            
            # Log improvement
            self.ecosystem.state.improvement_history.append(improvement_data)
            
            self.logger.info(f"Agent {agent.name} self-improvement completed with optimization")
            return improvement_data
            
        except Exception as e:
            self.logger.error(f"Self-improvement failed for {agent.name}: {e}")
            return {"success": False, "error": str(e)}
    
    async def apply_optimization_improvements(self, agent: AgentProfile, optimization_result: Dict[str, Any]):
        """Apply optimization improvements to agent"""
        try:
            improvements = optimization_result.get("improvements", {})
            
            # Update agent capabilities based on optimization
            if "capabilities" in improvements:
                for capability, improvement in improvements["capabilities"].items():
                    current_value = agent.capabilities.get(capability, 0.0)
                    new_value = min(1.0, current_value + improvement)
                    agent.update_capability(capability, new_value)
            
            # Update agent prompt if optimized
            if "prompt" in improvements:
                agent.prompt_template = improvements["prompt"]
            
            # Update agent parameters
            if "parameters" in improvements:
                for param, value in improvements["parameters"].items():
                    setattr(agent, param, value)
                    
        except Exception as e:
            self.logger.error(f"Error applying optimization improvements: {e}")
    
    async def execute_collaborative_task_with_tools(self, agents: List[AgentProfile], 
                                                   task: Dict[str, Any], 
                                                   role_assignments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute collaborative task with tool integration"""
        try:
            results = {"agent_results": {}, "combined_result": None}
            
            # Execute individual agent tasks
            for agent in agents:
                agent_role = role_assignments.get(agent.agent_id, "participant")
                agent_task = self.customize_task_for_agent(task, agent, agent_role)
                
                # Execute task with available tools
                if hasattr(agent, 'available_tools') and hasattr(self.ecosystem, 'tool_integration'):
                    agent_result = await self.execute_task_with_tools(agent, agent_task)
                else:
                    agent_result = await self.execute_basic_task(agent, agent_task)
                
                results["agent_results"][agent.agent_id] = agent_result
            
            # Combine results
            results["combined_result"] = await self.combine_collaboration_results(
                results["agent_results"], task
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error executing collaborative task: {e}")
            return {"success": False, "error": str(e)}
    
    async def execute_task_with_tools(self, agent: AgentProfile, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent task with tool support"""
        try:
            result = {"success": True, "outputs": {}, "tool_usage": []}
            
            # Suggest tools for the task
            if hasattr(self.ecosystem, 'tool_integration'):
                suggested_tools = await self.ecosystem.tool_integration.suggest_tool_for_task(
                    task.get("description", ""), agent
                )
                
                # Use tools as needed
                for tool_name in suggested_tools:
                    if tool_name in getattr(agent, 'available_tools', []):
                        tool_result = await self.ecosystem.tool_integration.execute_tool(
                            tool_name, "execute", task.get("parameters", {}), agent
                        )
                        result["tool_usage"].append({
                            "tool": tool_name,
                            "result": tool_result
                        })
            
            # Execute core agent task
            core_result = await self.execute_basic_task(agent, task)
            result["outputs"] = core_result.get("outputs", {})
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing task with tools: {e}")
            return {"success": False, "error": str(e)}
    
    async def execute_basic_task(self, agent: AgentProfile, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute basic agent task without tools"""
        try:
            # Simulate task execution
            result = {
                "success": True,
                "outputs": {
                    "result": f"Task completed by {agent.name}",
                    "quality_score": min(1.0, agent.calculate_intelligence_quotient() + 0.1)
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing basic task: {e}")
            return {"success": False, "error": str(e)}
    
    def customize_task_for_agent(self, task: Dict[str, Any], agent: AgentProfile, role: str) -> Dict[str, Any]:
        """Customize task based on agent role and capabilities"""
        customized_task = task.copy()
        customized_task["agent_role"] = role
        customized_task["agent_capabilities"] = dict(agent.capabilities)
        
        # Adjust task complexity based on agent intelligence
        intelligence = agent.calculate_intelligence_quotient()
        customized_task["complexity_factor"] = intelligence
        
        return customized_task
    
    async def combine_collaboration_results(self, agent_results: Dict[str, Any], 
                                          task: Dict[str, Any]) -> Dict[str, Any]:
        """Combine individual agent results into collaborative outcome"""
        try:
            combined = {
                "success": True,
                "participant_count": len(agent_results),
                "combined_output": "",
                "average_quality": 0.0
            }
            
            total_quality = 0.0
            successful_agents = 0
            
            for agent_id, result in agent_results.items():
                if result.get("success"):
                    successful_agents += 1
                    outputs = result.get("outputs", {})
                    combined["combined_output"] += f"Agent {agent_id}: {outputs.get('result', '')}\n"
                    total_quality += outputs.get("quality_score", 0.0)
            
            if successful_agents > 0:
                combined["average_quality"] = total_quality / successful_agents
                combined["success_rate"] = successful_agents / len(agent_results)
            
            return combined
            
        except Exception as e:
            self.logger.error(f"Error combining collaboration results: {e}")
            return {"success": False, "error": str(e)}
    
    async def process_agent_cycle(self, agent: AgentProfile):
        try:
            # Use integrated self-improvement
            improvement_context = {
                "cycle_type": "regular",
                "benchmark_type": "general",
                "evaluation_criteria": ["performance", "quality"]
            }
            await self.self_improve_agent(agent, improvement_context)
            
            # Find collaboration opportunities
            potential_partners = await self.find_collaboration_partners(agent)
            if potential_partners:
                collaboration_task = {
                    "description": "Knowledge exchange and collaborative problem solving",
                    "type": "knowledge_synthesis",
                    "parameters": {"domain": agent.specialization}
                }
                await self.facilitate_collaboration([agent] + potential_partners[:2], collaboration_task)
            
            await self.agent_knowledge_synthesis(agent)
            
            agent.last_active = datetime.now()
            agent.energy_level = max(0.1, agent.energy_level - 0.02)
            
        except Exception as e:
            self.logger.error(f"Error in agent cycle for {agent.agent_id}: {e}")
            
    async def find_collaboration_partners(self, agent: AgentProfile) -> List[AgentProfile]:
        candidates = []
        
        for other_id, other_agent in self.ecosystem.state.agents.items():
            if (other_id != agent.agent_id and 
                other_agent.energy_level > 0.3 and
                len(other_agent.collaboration_network) < 8):
                
                score = self.calculate_advanced_collaboration_score(agent, other_agent)
                candidates.append((other_agent, score))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return [candidate[0] for candidate in candidates[:3]]
    
    def calculate_advanced_collaboration_score(self, agent1: AgentProfile, agent2: AgentProfile) -> float:
        capability_synergy = 0
        for cap in AgentCapability:
            cap1 = agent1.capabilities.get(cap, 0)
            cap2 = agent2.capabilities.get(cap, 0)
            
            complementarity = abs(cap1 - cap2)
            synergy = min(cap1, cap2)
            capability_synergy += (complementarity * 0.6 + synergy * 0.4)
        
        capability_synergy /= len(AgentCapability)
        
        common_domains = set(agent1.knowledge_domains) & set(agent2.knowledge_domains)
        unique_domains = set(agent1.knowledge_domains) ^ set(agent2.knowledge_domains)
        
        domain_overlap = len(common_domains) / max(len(agent1.knowledge_domains), len(agent2.knowledge_domains), 1)
        domain_complementarity = len(unique_domains) / max(len(agent1.knowledge_domains) + len(agent2.knowledge_domains), 1)
        
        role_compatibility = self.calculate_role_compatibility(agent1.role, agent2.role)
        
        trust_factor = (agent1.trust_score + agent2.trust_score) / 2
        performance_factor = (agent1.calculate_intelligence_quotient() + agent2.calculate_intelligence_quotient()) / 2
        
        collaboration_score = (
            capability_synergy * 0.3 +
            (domain_overlap * 0.3 + domain_complementarity * 0.7) * 0.25 +
            role_compatibility * 0.2 +
            trust_factor * 0.15 +
            performance_factor * 0.1
        )
        
        return collaboration_score
    
    def calculate_role_compatibility(self, role1: AgentRole, role2: AgentRole) -> float:
        compatibility_matrix = {
            AgentRole.SYSTEM_OPTIMIZATION: {
                AgentRole.RESOURCE_MANAGEMENT: 0.9,
                AgentRole.MONITORING_ANALYSIS: 0.8,
                AgentRole.META_LEARNING: 0.7,
                AgentRole.ETHICAL_OVERSIGHT: 0.6
            },
            AgentRole.META_LEARNING: {
                AgentRole.KNOWLEDGE_ACQUISITION: 0.9,
                AgentRole.SYSTEM_OPTIMIZATION: 0.7,
                AgentRole.PROBLEM_SOLVING: 0.8,
                AgentRole.SELF_IMPROVEMENT: 0.8
            },
            AgentRole.ETHICAL_OVERSIGHT: {
                AgentRole.SYSTEM_OPTIMIZATION: 0.6,
                AgentRole.COLLABORATION_COORDINATION: 0.7,
                AgentRole.RESOURCE_MANAGEMENT: 0.6,
                AgentRole.SECURITY_AUDIT: 0.9
            },
            AgentRole.KNOWLEDGE_ACQUISITION: {
                AgentRole.META_LEARNING: 0.9,
                AgentRole.PROBLEM_SOLVING: 0.8,
                AgentRole.CODE_GENERATION: 0.7,
                AgentRole.EMERGENT_GOAL_GENERATION: 0.6
            },
            AgentRole.COLLABORATION_COORDINATION: {
                AgentRole.ETHICAL_OVERSIGHT: 0.7,
                AgentRole.RESOURCE_MANAGEMENT: 0.8,
                AgentRole.SYSTEM_OPTIMIZATION: 0.6,
                AgentRole.META_LEARNING: 0.5
            }
        }
        
        if role1 in compatibility_matrix and role2 in compatibility_matrix[role1]:
            return compatibility_matrix[role1][role2]
        elif role2 in compatibility_matrix and role1 in compatibility_matrix[role2]:
            return compatibility_matrix[role2][role1]
        else:
            return 0.5
    
    async def advanced_knowledge_exchange(self, agent1: AgentProfile, agent2: AgentProfile):
        try:
            agent1_unique = set(agent1.knowledge_domains) - set(agent2.knowledge_domains)
            agent2_unique = set(agent2.knowledge_domains) - set(agent1.knowledge_domains)
            
            for domain in list(agent1_unique)[:2]:
                if domain not in agent2.knowledge_domains:
                    agent2.knowledge_domains.append(domain)
                    
            for domain in list(agent2_unique)[:2]:
                if domain not in agent1.knowledge_domains:
                    agent1.knowledge_domains.append(domain)
            
            for capability in AgentCapability:
                cap1 = agent1.capabilities.get(capability, 0)
                cap2 = agent2.capabilities.get(capability, 0)
                
                if cap1 > cap2 + 0.1:
                    improvement = (cap1 - cap2) * 0.05
                    agent2.update_capability(capability, min(1.0, cap2 + improvement))
                elif cap2 > cap1 + 0.1:
                    improvement = (cap2 - cap1) * 0.05
                    agent1.update_capability(capability, min(1.0, cap1 + improvement))
            
            agent1.trust_score = min(1.0, agent1.trust_score + 0.01)
            agent2.trust_score = min(1.0, agent2.trust_score + 0.01)
            
            collaboration_record = {
                "event": "knowledge_exchange",
                "partner": agent2.agent_id,
                "timestamp": datetime.now().isoformat(),
                "knowledge_shared": len(agent1_unique) + len(agent2_unique)
            }
            
            agent1.performance_history.append(collaboration_record.copy())
            collaboration_record["partner"] = agent1.agent_id
            agent2.performance_history.append(collaboration_record)
            
        except Exception as e:
            self.logger.error(f"Error in knowledge exchange: {e}")
    
    async def collaborative_problem_solving(self, agent1: AgentProfile, agent2: AgentProfile):
        try:
            problem_prompt = f"""
            Generate a complex problem that requires collaboration between:
            Agent 1 ({agent1.role.value}): Capabilities {agent1.get_dominant_capabilities()}
            Agent 2 ({agent2.role.value}): Capabilities {agent2.get_dominant_capabilities()}
            
            The problem should:
            1. Leverage both agents' strengths
            2. Require knowledge synthesis
            3. Be relevant to ecosystem improvement
            4. Have measurable outcomes
            
            Provide the problem statement and success criteria.
            """
            
            problem_response = await self.models["reasoning"].agenerate(problem_prompt)
            problem = problem_response.content
            
            solution_prompt = f"""
            Collaborate to solve this problem:
            {problem}
            
            Agent 1 strengths: {agent1.get_dominant_capabilities()}
            Agent 2 strengths: {agent2.get_dominant_capabilities()}
            
            Provide a comprehensive solution that:
            1. Uses both agents' capabilities
            2. Shows clear collaboration
            3. Includes implementation steps
            4. Addresses potential challenges
            """
            
            solution_response = await self.models["reasoning"].agenerate(solution_prompt)
            solution = solution_response.content
            
            quality_score = await self.evaluate_solution_quality(problem, solution)
            
            performance_record = {
                "task": "collaborative_problem_solving",
                "partner": agent2.agent_id,
                "problem": problem[:200] + "...",
                "solution_quality": quality_score,
                "timestamp": datetime.now().isoformat(),
                "success": quality_score > 0.6
            }
            
            agent1.performance_history.append(performance_record.copy())
            performance_record["partner"] = agent1.agent_id
            agent2.performance_history.append(performance_record)
            
            if quality_score > 0.7:
                agent1.update_capability(AgentCapability.COLLABORATION,
                                       min(1.0, agent1.capabilities[AgentCapability.COLLABORATION] + 0.01))
                agent2.update_capability(AgentCapability.COLLABORATION,
                                       min(1.0, agent2.capabilities[AgentCapability.COLLABORATION] + 0.01))
            
            self.ecosystem.state.total_tasks_completed += 1
            
        except Exception as e:
            self.logger.error(f"Error in collaborative problem solving: {e}")
    
    async def evaluate_solution_quality(self, problem: str, solution: str) -> float:
        try:
            evaluation_prompt = f"""
            Evaluate the quality of this solution on a scale of 0.0 to 1.0:
            
            Problem: {problem}
            Solution: {solution}
            
            Criteria:
            1. Completeness (addresses all aspects)
            2. Feasibility (can be implemented)
            3. Innovation (creative approach)
            4. Collaboration (shows teamwork)
            5. Impact (potential benefit)
            
            Provide only a numerical score between 0.0 and 1.0.
            """
            
            response = await self.models["reasoning"].agenerate(evaluation_prompt)
            
            try:
                score = float(response.content.strip())
                return max(0.0, min(1.0, score))
            except ValueError:
                quality_keywords = ["innovative", "comprehensive", "feasible", "effective", "collaborative"]
                keyword_count = sum(1 for word in quality_keywords if word in solution.lower())
                length_score = min(1.0, len(solution) / 1000)
                return (keyword_count / len(quality_keywords)) * 0.7 + length_score * 0.3
                
        except Exception as e:
            self.logger.error(f"Error evaluating solution quality: {e}")
            return 0.5
    
    def extract_learning_insights(self, response: str) -> Dict[str, Any]:
        insights = {"meta_improvement": 0.01}
        
        if "efficiency" in response.lower():
            insights["meta_improvement"] += 0.005
        if "transfer" in response.lower():
            insights["meta_improvement"] += 0.003
        if "adaptive" in response.lower():
            insights["meta_improvement"] += 0.007
            
        return insights
    
    def extract_optimization_strategies(self, response: str) -> Dict[str, Any]:
        strategies = {}
        
        if "resource" in response.lower():
            strategies["resource_optimization"] = True
        if "workload" in response.lower():
            strategies["workload_balancing"] = True
        if "architecture" in response.lower():
            strategies["architecture_improvement"] = True
            
        return strategies
    
    def extract_ethical_insights(self, response: str) -> Dict[str, Any]:
        insights = {}
        
        if "bias" in response.lower():
            insights["bias_mitigation"] = True
        if "alignment" in response.lower():
            insights["value_alignment"] = True
        if "framework" in response.lower():
            insights["framework_improvement"] = True
            
        return insights
    
    def extract_knowledge_opportunities(self, response: str) -> Dict[str, Any]:
        opportunities = {}
        
        lines = response.split('\n')
        domains = []
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['domain', 'field', 'area']):
                domains.append(line.strip())
        
        opportunities["new_domains"] = domains[:3]
        return opportunities
    
    async def pursue_knowledge_opportunities(self, agent: AgentProfile, opportunities: Dict[str, Any]):
        new_domains = opportunities.get("new_domains", [])
        
        for domain_desc in new_domains:
            domain_name = domain_desc.split()[-1].lower().replace('.', '').replace(',', '')
            
            if domain_name and domain_name not in agent.knowledge_domains:
                agent.knowledge_domains.append(domain_name)
                self.ecosystem.state.total_knowledge_items += 1
    
    async def apply_system_optimizations(self, optimizations: Dict[str, Any]):
        if optimizations.get("resource_optimization"):
            self.ecosystem.state.system_health_score = min(1.0, self.ecosystem.state.system_health_score + 0.01)
        
        if optimizations.get("workload_balancing"):
            await self.balance_agent_workloads()
    
    async def apply_ethical_improvements(self, insights: Dict[str, Any]):
        if insights.get("bias_mitigation"):
            for agent in self.ecosystem.state.agents.values():
                agent.trust_score = min(1.0, agent.trust_score + 0.005)
    
    async def balance_agent_workloads(self):
        for agent in self.ecosystem.state.agents.values():
            if agent.energy_level < 0.3:
                agent.energy_level = min(1.0, agent.energy_level + 0.1)
    
    def calculate_agent_efficiency(self) -> float:
        if not self.ecosystem.state.agents:
            return 0.0
        
        total_efficiency = sum(agent.energy_level * agent.calculate_intelligence_quotient() 
                             for agent in self.ecosystem.state.agents.values())
        return total_efficiency / len(self.ecosystem.state.agents)
    
    def calculate_task_completion_rate(self) -> float:
        total_tasks = sum(len(agent.performance_history) for agent in self.ecosystem.state.agents.values())
        if total_tasks == 0:
            return 0.0
        
        successful_tasks = sum(
            sum(1 for task in agent.performance_history if task.get("success", False))
            for agent in self.ecosystem.state.agents.values()
        )
        
        return successful_tasks / total_tasks