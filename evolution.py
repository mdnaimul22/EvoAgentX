import asyncio
import json
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import traceback
from systemd import AgentRole, AgentCapability, AgentProfile, EcosystemState
from evoagentx.rag.schema import Chunk, ChunkMetadata, Corpus


class EcosystemEvolution:
    
    def __init__(self, ecosystem):
        self.ecosystem = ecosystem
        self.logger = ecosystem.logger
        self.models = ecosystem.models
        self.rag_engines = ecosystem.rag_engines
        self.config = ecosystem.config
        
    async def run_evolution_cycle(self):
        """Run a complete evolution cycle with integrated systems"""
        try:
            cycle_start = datetime.now()
            
            # Phase 1: Assessment with evaluation pipeline
            current_fitness = await self.calculate_ecosystem_fitness()
            
            # Run comprehensive agent evaluations
            if hasattr(self.ecosystem, 'evaluation_pipeline'):
                evaluation_results = await self.ecosystem.evaluation_pipeline.evaluate_all_agents()
            else:
                evaluation_results = {}
            
            # Phase 2: Benchmarking
            benchmark_results = await self.run_ecosystem_benchmarks()
            
            # Phase 3: MCP and Optimization Integration
            mcp_improvements = await self.optimize_mcp_ecosystem()
            
            # Apply optimization strategies to underperforming agents
            if hasattr(self.ecosystem, 'optimization_system'):
                optimization_results = await self.apply_ecosystem_optimizations(evaluation_results)
            else:
                optimization_results = {}
            
            # Phase 4: Agent Management with HITL oversight
            await self.manage_agent_population_with_hitl()
            
            # Phase 5: Capability Enhancement with tool integration
            await self.enhance_ecosystem_capabilities_with_tools()
            
            # Phase 6: Integrated Knowledge Synthesis
            knowledge_synthesis_results = await self.run_integrated_knowledge_synthesis_cycle()
            
            # Calculate new fitness
            new_fitness = await self.calculate_ecosystem_fitness()
            
            # Record comprehensive evolution data
            evolution_record = {
                "cycle_id": str(uuid.uuid4()),
                "timestamp": cycle_start.isoformat(),
                "duration": (datetime.now() - cycle_start).total_seconds(),
                "previous_fitness": current_fitness,
                "new_fitness": new_fitness,
                "improvement": new_fitness - current_fitness,
                "benchmark_results": benchmark_results,
                "evaluation_results": evaluation_results,
                "optimization_results": optimization_results,
                "mcp_improvements": mcp_improvements,
                "knowledge_synthesis": knowledge_synthesis_results,
                "agent_count": len(self.ecosystem.state.agents),
                "system_health": await self.ecosystem.calculate_system_health()
            }
            
            self.ecosystem.state.evolution_history.append(evolution_record)
            self.ecosystem.state.system_intelligence_score = new_fitness
            
            self.logger.info(f"Integrated evolution cycle completed. Fitness: {current_fitness:.3f} → {new_fitness:.3f}")
            
            return evolution_record
            
        except Exception as e:
            self.logger.error(f"Error in evolution cycle: {e}")
            return None
    
    async def apply_ecosystem_optimizations(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply optimization strategies based on evaluation results"""
        try:
            optimization_results = {"optimized_agents": [], "strategies_applied": []}
            
            # Identify underperforming agents
            underperforming_agents = []
            for agent_id, eval_data in evaluation_results.items():
                if eval_data.get("overall_score", 0.5) < 0.6:
                    if agent_id in self.ecosystem.state.agents:
                        underperforming_agents.append(self.ecosystem.state.agents[agent_id])
            
            # Apply optimization strategies
            for agent in underperforming_agents:
                optimization_result = await self.ecosystem.optimization_system.optimize_agent(
                    agent, strategy="adaptive", context={"reason": "ecosystem_evolution"}
                )
                
                if optimization_result.get("success"):
                    optimization_results["optimized_agents"].append(agent.agent_id)
                    optimization_results["strategies_applied"].append(optimization_result.get("strategy"))
            
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Error applying ecosystem optimizations: {e}")
            return {}
    
    async def manage_agent_population_with_hitl(self):
        """Manage agent population with HITL oversight"""
        try:
            current_count = len(self.ecosystem.state.agents)
            target_count = self.config.get("target_agent_count", 10)
            
            if current_count < target_count:
                # Request HITL approval for agent creation
                if hasattr(self.ecosystem, 'hitl_system'):
                    approval_request = {
                        "action": "create_new_agents",
                        "current_count": current_count,
                        "target_count": target_count,
                        "risk_level": "medium"
                    }
                    approved = await self.ecosystem.hitl_system.request_approval(
                        "system_optimization", approval_request
                    )
                    
                    if approved:
                        await self.create_new_agents(target_count - current_count)
                else:
                    await self.create_new_agents(target_count - current_count)
            
            elif current_count > target_count * 1.2:
                # Consider agent retirement with HITL approval
                if hasattr(self.ecosystem, 'hitl_system'):
                    approval_request = {
                        "action": "retire_underperforming_agents",
                        "current_count": current_count,
                        "target_count": target_count,
                        "risk_level": "high"
                    }
                    approved = await self.ecosystem.hitl_system.request_approval(
                        "system_optimization", approval_request
                    )
                    
                    if approved:
                        await self.retire_underperforming_agents(current_count - target_count)
            
        except Exception as e:
            self.logger.error(f"Error managing agent population: {e}")
    
    async def enhance_ecosystem_capabilities_with_tools(self):
        """Enhance ecosystem capabilities with tool integration"""
        try:
            if hasattr(self.ecosystem, 'tool_integration'):
                # Auto-discover new tools
                await self.ecosystem.tool_integration.auto_discover_tools()
                
                # Optimize tool usage across agents
                await self.ecosystem.tool_integration.optimize_tool_usage()
                
                # Update agent tool assignments
                for agent in self.ecosystem.state.agents.values():
                    recommended_tools = await self.ecosystem.tool_integration.get_tools_for_agent(agent)
                    agent.available_tools = recommended_tools
            
        except Exception as e:
            self.logger.error(f"Error enhancing capabilities with tools: {e}")
    
    async def run_integrated_knowledge_synthesis_cycle(self) -> Dict[str, Any]:
        """Run knowledge synthesis cycle with integration"""
        try:
            synthesis_results = {"domains_synthesized": 0, "knowledge_updated": False}
            
            if hasattr(self.ecosystem, 'knowledge_synthesis'):
                # Run synthesis cycle
                cycle_result = await self.ecosystem.knowledge_synthesis.run_synthesis_cycle()
                synthesis_results.update(cycle_result)
                
                # Update RAG engines with new knowledge
                if cycle_result.get("knowledge_updated"):
                    for engine_name, engine in self.ecosystem.rag_engines.items():
                        await engine.update_knowledge_base(cycle_result.get("synthesized_knowledge", {}))
            
            return synthesis_results
            
        except Exception as e:
            self.logger.error(f"Error in integrated knowledge synthesis: {e}")
            return {}
    
    async def create_new_agents(self, count: int):
        """Create new agents based on ecosystem needs"""
        try:
            for i in range(count):
                # Determine needed role based on current distribution
                needed_role = self.determine_needed_agent_role()
                
                agent_config = {
                    "name": f"EvoAgent_{datetime.now().strftime('%H%M%S')}_{i}",
                    "role": needed_role.value,
                    "specialization": self.determine_specialization(needed_role)
                }
                
                await self.ecosystem.create_agent(agent_config)
                
        except Exception as e:
            self.logger.error(f"Error creating new agents: {e}")
    
    async def retire_underperforming_agents(self, count: int):
        """Retire underperforming agents"""
        try:
            # Sort agents by performance
            agents_by_performance = sorted(
                self.ecosystem.state.agents.values(),
                key=lambda a: a.calculate_intelligence_quotient()
            )
            
            # Retire lowest performing agents
            for agent in agents_by_performance[:count]:
                self.logger.info(f"Retiring underperforming agent: {agent.name}")
                del self.ecosystem.state.agents[agent.agent_id]
                
        except Exception as e:
            self.logger.error(f"Error retiring agents: {e}")
    
    def determine_needed_agent_role(self):
        """Determine what agent role is most needed"""
        from ecosystem import AgentRole
        from collections import Counter
        
        # Count current roles
        current_roles = Counter(agent.role for agent in self.ecosystem.state.agents.values())
        
        # Find least represented role
        all_roles = list(AgentRole)
        least_represented = min(all_roles, key=lambda role: current_roles.get(role, 0))
        
        return least_represented
    
    def determine_specialization(self, role):
        """Determine specialization based on role"""
        from ecosystem import AgentRole
        
        specializations = {
            AgentRole.KNOWLEDGE_ACQUISITION: "research",
            AgentRole.ANALYTICAL_PROCESSING: "analysis",
            AgentRole.CREATIVE_SYNTHESIS: "creativity",
            AgentRole.SYSTEM_OPTIMIZATION: "optimization",
            AgentRole.META_LEARNING: "learning",
            AgentRole.COORDINATION: "coordination",
            AgentRole.QUALITY_ASSURANCE: "quality",
            AgentRole.ETHICAL_OVERSIGHT: "ethics"
        }
        
        return specializations.get(role, "general")
    
    async def run_ecosystem_benchmarking(self) -> Dict[str, Any]:
        try:
            benchmarking_system = getattr(self.ecosystem, 'benchmarking_system', None)
            if not benchmarking_system:
                return {"status": "no_benchmarking_system"}
            
            benchmarking_results = await benchmarking_system.benchmark_ecosystem_evolution()
            
            self.logger.info(f"Ecosystem benchmarking completed. Score: {benchmarking_results.get('ecosystem_benchmark_score', 0):.3f}")
            
            return benchmarking_results
            
        except Exception as e:
            self.logger.error(f"Error in ecosystem benchmarking: {e}")
            return {"status": "error", "error": str(e)}
    
    async def optimize_mcp_ecosystem(self):
        try:
            mcp_integration = getattr(self.ecosystem, 'mcp_integration', None)
            if not mcp_integration:
                return
            
            await mcp_integration.optimize_tool_ecosystem()
            self.mcp_optimization_results = mcp_integration.get_mcp_status()
            
            self.logger.info("MCP tool ecosystem optimization completed")
            
        except Exception as e:
            self.logger.error(f"Error in MCP ecosystem optimization: {e}")
    
    async def analyze_evolution_needs(self, benchmarking_insights: Dict[str, Any] = None) -> Dict[str, Any]:
        current_count = len(self.ecosystem.state.agents)
        
        needs = {
            "create_agents": 0,
            "remove_agents": 0,
            "enhance_capabilities": [AgentCapability.LEARNING, AgentCapability.ADAPTATION],
            "knowledge_gaps": ["optimization", "collaboration"],
            "system_optimizations": ["efficiency"],
            "benchmarking_recommendations": [],
            "mcp_tool_needs": []
        }
        
        if benchmarking_insights and benchmarking_insights.get("evolution_insights"):
            insights = benchmarking_insights["evolution_insights"]
            needs["benchmarking_recommendations"] = insights.get("recommendations", [])
            
            if insights.get("overall_health") == "needs_attention":
                needs["enhance_capabilities"].extend([
                    AgentCapability.REASONING,
                    AgentCapability.PROBLEM_DECOMPOSITION
                ])
        
        if current_count < self.config.min_agent_count:
            needs["create_agents"] = self.config.min_agent_count - current_count
        elif current_count > self.config.max_agent_count:
            needs["remove_agents"] = current_count - self.config.max_agent_count
        elif current_count < self.config.initial_agent_count:
            needs["create_agents"] = min(2, self.config.initial_agent_count - current_count)
        
        await self.analyze_mcp_tool_needs(needs)
        
        return needs
    
    async def analyze_mcp_tool_needs(self, needs: Dict[str, Any]):
        try:
            mcp_integration = getattr(self.ecosystem, 'mcp_integration', None)
            if not mcp_integration:
                return
            
            tool_needs = []
            
            if needs["benchmarking_recommendations"]:
                tool_needs.append({
                    "description": "A tool to analyze benchmarking results and generate improvement recommendations",
                    "priority": "high"
                })
            
            if needs["enhance_capabilities"]:
                for capability in needs["enhance_capabilities"]:
                    tool_needs.append({
                        "description": f"A tool to enhance {capability.value} capability across the ecosystem",
                        "priority": "medium"
                    })
            
            collaboration_score = self.calculate_collaboration_effectiveness()
            if collaboration_score < 0.6:
                tool_needs.append({
                    "description": "A tool to optimize agent collaboration and knowledge sharing",
                    "priority": "medium"
                })
            
            needs["mcp_tool_needs"] = tool_needs
            
        except Exception as e:
            self.logger.error(f"Error analyzing MCP tool needs: {e}")
    
    async def calculate_ecosystem_fitness(self) -> float:
        try:
            intelligence_score = self.ecosystem.state.system_intelligence_score
            diversity_score = self.calculate_diversity_score()
            collaboration_score = self.calculate_collaboration_effectiveness()
            efficiency_score = self.calculate_efficiency_score()
            benchmarking_score = await self.calculate_benchmarking_fitness()
            tool_creation_score = await self.calculate_tool_creation_fitness()
            
            fitness = (
                intelligence_score * 0.25 +
                diversity_score * 0.15 +
                collaboration_score * 0.25 +
                efficiency_score * 0.15 +
                benchmarking_score * 0.1 +
                tool_creation_score * 0.1
            )
            
            return max(0.0, min(1.0, fitness))
            
        except Exception as e:
            self.logger.error(f"Error calculating ecosystem fitness: {e}")
            return 0.5
    
    async def calculate_benchmarking_fitness(self) -> float:
        try:
            benchmarking_system = getattr(self.ecosystem, 'benchmarking_system', None)
            if not benchmarking_system:
                return 0.5
            
            benchmark_report = await benchmarking_system.generate_ecosystem_benchmark_report()
            
            overall_success_rate = benchmark_report["summary"]["overall_success_rate"]
            recent_trend = benchmark_report["summary"]["recent_trend"]
            
            success_component = overall_success_rate * 0.6
            trend_component = max(0.0, recent_trend) * 0.4
            
            benchmarking_fitness = success_component + trend_component
            
            return max(0.0, min(1.0, benchmarking_fitness))
            
        except Exception as e:
            self.logger.error(f"Error calculating benchmarking fitness: {e}")
            return 0.5
    
    async def calculate_tool_creation_fitness(self) -> float:
        try:
            mcp_integration = getattr(self.ecosystem, 'mcp_integration', None)
            if not mcp_integration:
                return 0.5
            
            mcp_status = mcp_integration.get_mcp_status()
            
            total_tools = mcp_status["total_tools"]
            active_tools = mcp_status["active_tools"]
            recent_creations = mcp_status["recent_creations"]
            
            utilization_rate = active_tools / max(total_tools, 1)
            innovation_rate = min(1.0, recent_creations / 10.0)
            
            tool_fitness = (utilization_rate * 0.6 + innovation_rate * 0.4)
            
            return max(0.0, min(1.0, tool_fitness))
            
        except Exception as e:
            self.logger.error(f"Error calculating tool creation fitness: {e}")
            return 0.5
    
    def calculate_diversity_score(self) -> float:
        if not self.ecosystem.state.agents:
            return 0.0
        
        role_counts = defaultdict(int)
        for agent in self.ecosystem.state.agents.values():
            role_counts[agent.role] += 1
        
        role_entropy = self.calculate_entropy(list(role_counts.values()))
        max_role_entropy = np.log(len(AgentRole))
        role_diversity = role_entropy / max_role_entropy if max_role_entropy > 0 else 0
        
        capability_variances = []
        for capability in AgentCapability:
            scores = [agent.capabilities.get(capability, 0) for agent in self.ecosystem.state.agents.values()]
            capability_variances.append(np.var(scores))
        
        capability_diversity = np.mean(capability_variances)
        
        diversity = (role_diversity * 0.6 + capability_diversity * 0.4)
        return max(0.0, min(1.0, diversity))
    
    def calculate_entropy(self, values: List[int]) -> float:
        if not values or sum(values) == 0:
            return 0.0
        
        total = sum(values)
        probabilities = [v / total for v in values if v > 0]
        return -sum(p * np.log(p) for p in probabilities)
    
    def calculate_collaboration_effectiveness(self) -> float:
        if not self.ecosystem.state.agents:
            return 0.0
        
        total_possible = len(self.ecosystem.state.agents) * (len(self.ecosystem.state.agents) - 1)
        actual_connections = sum(len(agent.collaboration_network) for agent in self.ecosystem.state.agents.values())
        network_density = actual_connections / total_possible if total_possible > 0 else 0
        
        avg_trust = np.mean([agent.trust_score for agent in self.ecosystem.state.agents.values()])
        
        return (network_density * 0.5 + avg_trust * 0.5)
    
    def calculate_efficiency_score(self) -> float:
        if not self.ecosystem.state.agents:
            return 0.0
        
        avg_energy = np.mean([agent.energy_level for agent in self.ecosystem.state.agents.values()])
        
        if self.ecosystem.state.total_tasks_completed > 0:
            task_efficiency = self.ecosystem.state.total_tasks_completed / max(self.ecosystem.state.uptime_hours, 1)
        else:
            task_efficiency = 0
        
        task_efficiency = min(1.0, task_efficiency)
        
        return (avg_energy * 0.6 + task_efficiency * 0.4)
    
    async def execute_evolution_strategies(self, evolution_needs: Dict[str, Any]) -> Dict[str, Any]:
        results = {
            "agents_created": 0,
            "agents_removed": 0,
            "capabilities_enhanced": 0,
            "knowledge_added": 0
        }
        
        try:
            if evolution_needs.get("create_agents", 0) > 0:
                created = await self.create_evolved_agents(evolution_needs["create_agents"])
                results["agents_created"] = created
            
            if evolution_needs.get("remove_agents", 0) > 0:
                removed = await self.remove_underperforming_agents(evolution_needs["remove_agents"])
                results["agents_removed"] = removed
            
            enhanced = await self.enhance_ecosystem_capabilities(evolution_needs.get("enhance_capabilities", []))
            results["capabilities_enhanced"] = enhanced
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error executing evolution strategies: {e}")
            return results
    
    async def create_evolved_agents(self, count: int) -> int:
        created = 0
        
        try:
            role_counts = defaultdict(int)
            for agent in self.ecosystem.state.agents.values():
                role_counts[agent.role] += 1
            
            underrepresented_roles = [role for role in AgentRole if role_counts[role] < 2]
            if not underrepresented_roles:
                underrepresented_roles = list(AgentRole)
            
            for i in range(count):
                role = random.choice(underrepresented_roles)
                capabilities = await self.generate_evolved_capabilities(role)
                knowledge_domains = await self.generate_evolved_knowledge_domains(role)
                
                agent_id = self.ecosystem.create_agent(
                    role=role,
                    name=f"EvolvedAgent_{role.value}_{created + 1}",
                    capabilities=capabilities,
                    knowledge_domains=knowledge_domains
                )
                
                created += 1
                self.logger.info(f"Created evolved agent: {agent_id}")
                
                role_counts[role] += 1
                if role_counts[role] >= 2 and role in underrepresented_roles:
                    underrepresented_roles.remove(role)
                    if not underrepresented_roles:
                        break
            
        except Exception as e:
            self.logger.error(f"Error creating evolved agents: {e}")
        
        return created
    
    async def generate_evolved_capabilities(self, role: AgentRole) -> Dict[AgentCapability, float]:
        role_preferences = {
            AgentRole.SYSTEM_OPTIMIZATION: {
                AgentCapability.SYSTEMS_THINKING: (0.7, 0.9),
                AgentCapability.REASONING: (0.6, 0.8),
                AgentCapability.PROBLEM_DECOMPOSITION: (0.6, 0.8)
            },
            AgentRole.META_LEARNING: {
                AgentCapability.META_COGNITION: (0.8, 0.95),
                AgentCapability.LEARNING: (0.8, 0.95),
                AgentCapability.ADAPTATION: (0.6, 0.8)
            },
            AgentRole.ETHICAL_OVERSIGHT: {
                AgentCapability.ETHICAL_REASONING: (0.8, 0.95),
                AgentCapability.REASONING: (0.7, 0.85),
                AgentCapability.SELF_AWARENESS: (0.6, 0.8)
            }
        }
        
        preferences = role_preferences.get(role, {})
        capabilities = {}
        
        for capability in AgentCapability:
            if capability in preferences:
                min_val, max_val = preferences[capability]
                score = random.uniform(min_val, max_val)
            else:
                score = random.beta(2, 3)
                score = max(0.1, min(0.8, score))
            
            capabilities[capability] = score
        
        return capabilities
    
    async def generate_evolved_knowledge_domains(self, role: AgentRole) -> List[str]:
        role_domains = {
            AgentRole.SYSTEM_OPTIMIZATION: [
                "system_architecture", "performance_optimization", "resource_management",
                "distributed_systems", "scalability"
            ],
            AgentRole.META_LEARNING: [
                "machine_learning", "cognitive_science", "educational_psychology",
                "knowledge_representation", "transfer_learning"
            ],
            AgentRole.ETHICAL_OVERSIGHT: [
                "ethics", "moral_philosophy", "ai_safety", "value_alignment",
                "governance", "risk_assessment"
            ]
        }
        
        domain_pool = role_domains.get(role, [
            "general_intelligence", "problem_solving", "reasoning",
            "creativity", "adaptation", "learning"
        ])
        
        num_domains = random.randint(3, 5)
        selected_domains = random.sample(domain_pool, min(num_domains, len(domain_pool)))
        
        general_domains = ["mathematics", "logic", "communication"]
        selected_domains.extend(random.sample(general_domains, 1))
        
        return selected_domains
    
    async def remove_underperforming_agents(self, count: int) -> int:
        if len(self.ecosystem.state.agents) <= self.config.min_agent_count:
            return 0
        
        removed = 0
        
        try:
            agent_scores = []
            for agent_id, agent in self.ecosystem.state.agents.items():
                score = self.calculate_agent_performance_score(agent)
                agent_scores.append((agent_id, agent, score))
            
            agent_scores.sort(key=lambda x: x[2])
            
            max_removable = min(count, len(self.ecosystem.state.agents) - self.config.min_agent_count)
            
            for i in range(max_removable):
                agent_id, agent, score = agent_scores[i]
                
                role_count = sum(1 for a in self.ecosystem.state.agents.values() if a.role == agent.role)
                if role_count <= 1:
                    continue
                
                del self.ecosystem.state.agents[agent_id]
                removed += 1
                
                self.logger.info(f"Removed underperforming agent: {agent.name} (score: {score:.3f})")
        
        except Exception as e:
            self.logger.error(f"Error removing underperforming agents: {e}")
        
        return removed
    
    def calculate_agent_performance_score(self, agent: AgentProfile) -> float:
        capability_score = agent.calculate_intelligence_quotient()
        
        hours_since_active = (datetime.now() - agent.last_active).total_seconds() / 3600
        activity_score = max(0, 1 - hours_since_active / 168)
        
        if agent.performance_history:
            recent_tasks = agent.performance_history[-10:]
            success_rate = sum(1 for task in recent_tasks if task.get("success", False)) / len(recent_tasks)
        else:
            success_rate = 0.5
        
        performance = (
            capability_score * 0.4 +
            activity_score * 0.3 +
            success_rate * 0.3
        )
        
        return max(0.0, min(1.0, performance))
    
    async def enhance_ecosystem_capabilities(self, capabilities_to_enhance: List[AgentCapability]) -> int:
        enhanced = 0
        
        try:
            for capability in capabilities_to_enhance:
                candidates = [
                    agent for agent in self.ecosystem.state.agents.values()
                    if agent.capabilities.get(capability, 0) < 0.7
                ]
                
                for agent in candidates[:5]:
                    current_score = agent.capabilities.get(capability, 0)
                    improvement = random.uniform(0.05, 0.15)
                    new_score = min(1.0, current_score + improvement)
                    
                    agent.update_capability(capability, new_score)
                    enhanced += 1
                    
                    agent.evolution_history.append({
                        "event": "capability_enhancement",
                        "capability": capability.value,
                        "improvement": improvement,
                        "timestamp": datetime.now().isoformat()
                    })
        
        except Exception as e:
            self.logger.error(f"Error enhancing capabilities: {e}")
        
        return enhanced
    
    async def update_post_evolution_metrics(self, evolution_record: Dict[str, Any]):
        new_intelligence = self.calculate_collective_intelligence()
        self.ecosystem.state.system_intelligence_score = new_intelligence
        
        if evolution_record["success"]:
            self.ecosystem.state.system_health_score = min(1.0, self.ecosystem.state.system_health_score + 0.01)
        
        self.ecosystem.db_manager.save_system_state(self.ecosystem.state)
    
    def calculate_collective_intelligence(self) -> float:
        if not self.ecosystem.state.agents:
            return 0.0
        
        agent_intelligence = sum(agent.calculate_intelligence_quotient() for agent in self.ecosystem.state.agents.values())
        agent_intelligence /= len(self.ecosystem.state.agents)
        
        collaboration_score = self.calculate_collaboration_effectiveness()
        
        collective_intelligence = (agent_intelligence * 0.7 + collaboration_score * 0.3)
        return max(0.0, min(1.0, collective_intelligence))
    
    async def knowledge_synthesis_cycle(self):
        self.logger.info("🧠 Starting knowledge synthesis cycle")
        
        try:
            all_knowledge = {}
            for agent in self.ecosystem.state.agents.values():
                for domain in agent.knowledge_domains:
                    if domain not in all_knowledge:
                        all_knowledge[domain] = []
                    all_knowledge[domain].append({
                        "agent_id": agent.agent_id,
                        "capability_level": agent.calculate_intelligence_quotient(),
                        "last_updated": agent.last_active.isoformat()
                    })
            
            self.ecosystem.state.collective_knowledge = {
                "synthesis_timestamp": datetime.now().isoformat(),
                "knowledge_domains": list(all_knowledge.keys()),
                "contributing_agents": len(self.ecosystem.state.agents),
                "total_domains": len(all_knowledge)
            }
            
            await self.index_synthesized_knowledge(all_knowledge)
            
            self.logger.info(f"Knowledge synthesis completed. {len(all_knowledge)} domains synthesized.")
            
        except Exception as e:
            self.logger.error(f"Error in knowledge synthesis cycle: {e}")
    
    async def index_synthesized_knowledge(self, knowledge: Dict[str, Any]):
        try:
            knowledge_summary = f"Collective Knowledge Summary\n\n"
            for domain, contributors in knowledge.items():
                knowledge_summary += f"Domain: {domain}\n"
                knowledge_summary += f"Contributors: {len(contributors)} agents\n"
                knowledge_summary += f"Average capability: {np.mean([c['capability_level'] for c in contributors]):.3f}\n\n"
            
            chunks = []
            chunk_id = f"knowledge_synthesis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            chunk = Chunk(
                chunk_id=chunk_id,
                text=knowledge_summary,
                metadata=ChunkMetadata(
                    doc_id=chunk_id,
                    corpus_id="collective_knowledge"
                ),
                start_char_idx=0,
                end_char_idx=len(knowledge_summary),
                excluded_embed_metadata_keys=[],
                excluded_llm_metadata_keys=[],
                relationships={}
            )
            chunk.metadata.title = "Collective Knowledge Synthesis"
            chunks.append(chunk)
            
            corpus = Corpus(chunks=chunks, corpus_id="collective_knowledge")
            
            if self.ecosystem.primary_rag:
                self.ecosystem.primary_rag.add(index_type="vector", nodes=corpus, corpus_id="collective_knowledge")
            
        except Exception as e:
            self.logger.error(f"Error indexing synthesized knowledge: {e}")