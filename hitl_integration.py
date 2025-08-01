import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

from evoagentx.hitl import (
    HITLInterceptorAgent,
    HITLInteractionType,
    HITLMode,
    HITLManager,
    HITLUserInputCollectorAgent,
    HITLOutsideConversationAgent
)
from evoagentx.core import Message
from evoagentx.agents import Agent, CustomizeAgent
from systemd import AgentProfile, AgentRole


class HITLIntegrationLevel(Enum):
    MINIMAL = "minimal"
    MODERATE = "moderate"
    COMPREHENSIVE = "comprehensive"
    FULL_OVERSIGHT = "full_oversight"


class ProductionHITLSystem:
    
    def __init__(self, ecosystem):
        self.ecosystem = ecosystem
        self.logger = ecosystem.logger
        self.models = ecosystem.models
        
        self.integration_level = HITLIntegrationLevel.MODERATE
        self.human_timeout = 300
        self.auto_approve_threshold = 0.8
        
        self.hitl_manager = HITLManager()
        self.interceptor_agents = {}
        self.input_collectors = {}
        self.conversation_agents = {}
        
        self.pending_approvals = asyncio.Queue()
        self.human_responses = {}
        
        self.setup_hitl_agents()
        
        self.logger.info("Production HITL system initialized")
    
    def setup_hitl_agents(self):
        
        for role in AgentRole:
            interceptor = HITLInterceptorAgent(
                target_agent_name=role.value,
                target_action_name="any",
                name=f"HITL_Interceptor_{role.value}",
                llm_config=self.ecosystem.model_configs["reasoning"],
                interaction_type=self.get_interaction_type_for_role(role),
                mode=HITLMode.PRE_EXECUTION
            )
            self.interceptor_agents[role] = interceptor
        
        self.input_collector = HITLUserInputCollectorAgent(
            name="HITL_InputCollector",
            llm_config=self.ecosystem.model_configs["reasoning"],
            timeout=self.human_timeout
        )
        
        self.conversation_agent = HITLOutsideConversationAgent(
            name="HITL_ConversationAgent",
            llm_config=self.ecosystem.model_configs["creativity"],
            conversation_context="Production AI Ecosystem Management"
        )
    
    def get_interaction_type_for_role(self, role: AgentRole) -> HITLInteractionType:
        role_interaction_map = {
            AgentRole.ETHICAL_OVERSIGHT: HITLInteractionType.APPROVAL_REQUEST,
            AgentRole.SYSTEM_OPTIMIZATION: HITLInteractionType.FEEDBACK_REQUEST,
            AgentRole.META_LEARNING: HITLInteractionType.LEARNING_VALIDATION,
            AgentRole.KNOWLEDGE_ACQUISITION: HITLInteractionType.INFORMATION_GATHERING,
            AgentRole.CREATIVE_SYNTHESIS: HITLInteractionType.CREATIVE_COLLABORATION,
            AgentRole.ANALYTICAL_PROCESSING: HITLInteractionType.ANALYSIS_REVIEW,
            AgentRole.COORDINATION: HITLInteractionType.DECISION_SUPPORT,
            AgentRole.QUALITY_ASSURANCE: HITLInteractionType.QUALITY_CHECK
        }
        return role_interaction_map.get(role, HITLInteractionType.GENERAL_INQUIRY)
    
    async def should_involve_human(self, agent: AgentProfile, action: str, 
                                  context: Dict[str, Any]) -> bool:
        
        critical_actions = [
            "system_shutdown", "agent_deletion", "security_override",
            "ethical_violation_response", "emergency_protocol"
        ]
        
        if action in critical_actions:
            return True
        
        if self.integration_level == HITLIntegrationLevel.FULL_OVERSIGHT:
            return True
        
        if self.integration_level == HITLIntegrationLevel.MINIMAL:
            return action in critical_actions
        
        confidence_score = context.get("confidence", 0.5)
        risk_level = context.get("risk_level", "medium")
        
        if confidence_score < self.auto_approve_threshold:
            return True
        
        if risk_level in ["high", "critical"]:
            return True
        
        if agent.role == AgentRole.ETHICAL_OVERSIGHT:
            return True
        
        if agent.role == AgentRole.SYSTEM_OPTIMIZATION and risk_level != "low":
            return True
        
        return False
    
    async def request_human_approval(self, agent: AgentProfile, action: str,
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        
        approval_request = {
            "request_id": f"approval_{datetime.now().timestamp()}",
            "agent_id": agent.agent_id,
            "agent_name": agent.name,
            "agent_role": agent.role.value,
            "action": action,
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "timeout": self.human_timeout
        }
        
        interceptor = self.interceptor_agents.get(agent.role)
        if not interceptor:
            interceptor = self.interceptor_agents[AgentRole.COORDINATION]
        
        try:
            response = await interceptor(
                action_input_data={
                    "request": approval_request,
                    "requires_approval": True
                }
            )
            
            return {
                "approved": response.content.approved if hasattr(response.content, 'approved') else False,
                "feedback": response.content.feedback if hasattr(response.content, 'feedback') else "",
                "modifications": response.content.modifications if hasattr(response.content, 'modifications') else {},
                "human_involved": True
            }
            
        except asyncio.TimeoutError:
            self.logger.warning(f"Human approval timeout for {approval_request['request_id']}")
            return {
                "approved": False,
                "feedback": "Human approval timeout - action denied for safety",
                "modifications": {},
                "human_involved": False,
                "timeout": True
            }
        except Exception as e:
            self.logger.error(f"Error in human approval request: {e}")
            return {
                "approved": False,
                "feedback": f"Error in approval process: {e}",
                "modifications": {},
                "human_involved": False,
                "error": True
            }
    
    async def collect_human_feedback(self, agent: AgentProfile, task: str,
                                   result: Any) -> Dict[str, Any]:
        
        feedback_request = {
            "agent_id": agent.agent_id,
            "agent_name": agent.name,
            "task": task,
            "result": str(result)[:1000],
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            response = await self.input_collector(
                action_input_data={
                    "feedback_request": feedback_request,
                    "collect_rating": True,
                    "collect_suggestions": True
                }
            )
            
            return {
                "rating": response.content.rating if hasattr(response.content, 'rating') else 0,
                "feedback": response.content.feedback if hasattr(response.content, 'feedback') else "",
                "suggestions": response.content.suggestions if hasattr(response.content, 'suggestions') else [],
                "human_involved": True
            }
            
        except Exception as e:
            self.logger.error(f"Error collecting human feedback: {e}")
            return {
                "rating": 0,
                "feedback": f"Error collecting feedback: {e}",
                "suggestions": [],
                "human_involved": False,
                "error": True
            }
    
    async def facilitate_human_agent_conversation(self, agent: AgentProfile,
                                                topic: str) -> List[Message]:
        
        conversation_context = {
            "agent_id": agent.agent_id,
            "agent_name": agent.name,
            "agent_role": agent.role.value,
            "topic": topic,
            "capabilities": {cap.value: score for cap, score in agent.capabilities.items()}
        }
        
        try:
            conversation_messages = await self.conversation_agent(
                action_input_data={
                    "context": conversation_context,
                    "max_turns": 10,
                    "enable_learning": True
                }
            )
            
            return conversation_messages.content.messages if hasattr(conversation_messages.content, 'messages') else []
            
        except Exception as e:
            self.logger.error(f"Error in human-agent conversation: {e}")
            return []
    
    async def integrate_with_agent_operations(self, agent_ops):
        
        original_self_improvement = agent_ops.agent_self_improvement
        original_collaboration = agent_ops.agent_collaboration
        
        async def hitl_wrapped_self_improvement(agent: AgentProfile):
            context = {
                "confidence": 0.7,
                "risk_level": "medium",
                "current_capabilities": {cap.value: score for cap, score in agent.capabilities.items()}
            }
            
            if await self.should_involve_human(agent, "self_improvement", context):
                approval = await self.request_human_approval(agent, "self_improvement", context)
                if not approval["approved"]:
                    self.logger.info(f"Self-improvement denied for agent {agent.agent_id}: {approval['feedback']}")
                    return
            
            await original_self_improvement(agent)
            
            if self.integration_level in [HITLIntegrationLevel.COMPREHENSIVE, HITLIntegrationLevel.FULL_OVERSIGHT]:
                await self.collect_human_feedback(agent, "self_improvement", "Capability improvements applied")
        
        async def hitl_wrapped_collaboration(agent: AgentProfile):
            context = {
                "confidence": 0.8,
                "risk_level": "low",
                "collaboration_network_size": len(agent.collaboration_network)
            }
            
            if await self.should_involve_human(agent, "collaboration", context):
                approval = await self.request_human_approval(agent, "collaboration", context)
                if not approval["approved"]:
                    self.logger.info(f"Collaboration denied for agent {agent.agent_id}: {approval['feedback']}")
                    return
            
            await original_collaboration(agent)
        
        agent_ops.agent_self_improvement = hitl_wrapped_self_improvement
        agent_ops.agent_collaboration = hitl_wrapped_collaboration
    
    async def setup_human_interface(self):
        
        self.logger.info("HITL human interface ready")
        self.logger.info(f"Integration level: {self.integration_level.value}")
        self.logger.info(f"Human timeout: {self.human_timeout} seconds")
    
    def get_hitl_status(self) -> Dict[str, Any]:
        return {
            "integration_level": self.integration_level.value,
            "human_timeout": self.human_timeout,
            "auto_approve_threshold": self.auto_approve_threshold,
            "interceptor_agents": len(self.interceptor_agents),
            "pending_approvals": self.pending_approvals.qsize(),
            "active": True
        }
    
    async def shutdown(self):
        self.logger.info("Shutting down HITL system...")
        
        while not self.pending_approvals.empty():
            try:
                self.pending_approvals.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        self.logger.info("HITL system shutdown complete")