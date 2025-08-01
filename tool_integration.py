import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from evoagentx.tools.mcp import MCPToolkit
from evoagentx.tools import (
    FileToolkit, PythonInterpreterToolkit, WikipediaSearchToolkit, 
    GoogleFreeSearchToolkit, ArxivToolkit
)
from evoagentx.agents import Agent, CustomizeAgent
from evoagentx.workflow import WorkFlowGraph, WorkFlow
from systemd import AgentProfile, AgentRole

class ToolCategory(Enum):
    FILE_OPS = "file_ops"
    CODE_EXEC = "code_exec"
    SEARCH = "search"
    ANALYSIS = "analysis"
    MCP = "mcp"

class ToolIntegration:
    def __init__(self, ecosystem):
        self.ecosystem = ecosystem
        self.logger = ecosystem.logger
        self.models = ecosystem.models
        
        self.tools = {}
        self.tool_categories = {}
        self.agent_preferences = {}
        self.usage_history = []
        
        self.setup_core_tools()
        self.setup_mcp_tools()
        
        self.logger.info("Tool integration system initialized")
    
    def setup_core_tools(self):
        self.tools["file"] = {
            "instance": FileToolkit(),
            "category": ToolCategory.FILE_OPS,
            "capabilities": ["read_file", "write_file", "list_directory"],
            "description": "File system operations"
        }
        
        self.tools["python"] = {
            "instance": PythonInterpreterToolkit(),
            "category": ToolCategory.CODE_EXEC,
            "capabilities": ["execute_python", "code_analysis"],
            "description": "Python code execution"
        }
        
        self.tools["wikipedia"] = {
            "instance": WikipediaSearchToolkit(),
            "category": ToolCategory.SEARCH,
            "capabilities": ["search_wikipedia", "get_article"],
            "description": "Wikipedia knowledge search"
        }
        
        self.tools["google"] = {
            "instance": GoogleFreeSearchToolkit(),
            "category": ToolCategory.SEARCH,
            "capabilities": ["web_search", "get_results"],
            "description": "Web search capabilities"
        }
        
        self.tools["arxiv"] = {
            "instance": ArxivToolkit(),
            "category": ToolCategory.SEARCH,
            "capabilities": ["search_papers", "get_paper"],
            "description": "Academic paper search"
        }
        
        for tool_name, tool_info in self.tools.items():
            category = tool_info["category"]
            if category not in self.tool_categories:
                self.tool_categories[category] = []
            self.tool_categories[category].append(tool_name)
        
        self.logger.info(f"Initialized {len(self.tools)} core tools")
    
    def setup_mcp_tools(self):
        try:
            self.tools["mcp"] = {
                "instance": MCPToolkit(),
                "category": ToolCategory.MCP,
                "capabilities": ["dynamic_tools", "server_management"],
                "description": "MCP dynamic tool system"
            }
            
            if ToolCategory.MCP not in self.tool_categories:
                self.tool_categories[ToolCategory.MCP] = []
            self.tool_categories[ToolCategory.MCP].append("mcp")
            
            self.logger.info("MCP tools integrated")
            
        except Exception as e:
            self.logger.error(f"MCP tools setup error: {e}")
    
    async def get_tools_for_agent(self, agent: AgentProfile) -> List[str]:
        role_tools = {
            AgentRole.KNOWLEDGE_ACQUISITION: ["wikipedia", "arxiv", "google"],
            AgentRole.ANALYTICAL_PROCESSING: ["python", "file", "mcp"],
            AgentRole.CREATIVE_SYNTHESIS: ["google", "wikipedia", "mcp"],
            AgentRole.SYSTEM_OPTIMIZATION: ["python", "file", "mcp"],
            AgentRole.META_LEARNING: ["arxiv", "wikipedia", "python"],
            AgentRole.COORDINATION: ["file", "mcp"],
            AgentRole.QUALITY_ASSURANCE: ["python", "file"],
            AgentRole.ETHICAL_OVERSIGHT: ["wikipedia", "arxiv"]
        }
        
        recommended_tools = role_tools.get(agent.role, ["file", "python"])
        
        from ecosystem import AgentCapability
        
        if agent.capabilities.get(AgentCapability.RESEARCH, 0) > 0.7:
            recommended_tools.extend(["wikipedia", "arxiv", "google"])
        
        if agent.capabilities.get(AgentCapability.PROGRAMMING, 0) > 0.7:
            recommended_tools.append("python")
        
        if agent.capabilities.get(AgentCapability.DATA_ANALYSIS, 0) > 0.7:
            recommended_tools.extend(["python", "file"])
        
        recommended_tools = list(set(recommended_tools))
        available_tools = [tool for tool in recommended_tools if tool in self.tools]
        
        return available_tools
    
    async def execute_tool(self, tool_name: str, action: str, 
                          parameters: Dict[str, Any], 
                          agent: Optional[AgentProfile] = None) -> Dict[str, Any]:
        if tool_name not in self.tools:
            return {"success": False, "error": f"Tool {tool_name} not found"}
        
        tool_info = self.tools[tool_name]
        tool_instance = tool_info["instance"]
        
        try:
            if hasattr(tool_instance, action):
                result = await getattr(tool_instance, action)(**parameters)
            else:
                return {"success": False, "error": f"Action {action} not available in {tool_name}"}
            
            usage_record = {
                "timestamp": datetime.now().isoformat(),
                "tool": tool_name,
                "action": action,
                "agent_id": agent.agent_id if agent else None,
                "success": True
            }
            self.usage_history.append(usage_record)
            
            return {"success": True, "result": result}
            
        except Exception as e:
            self.logger.error(f"Tool execution error - {tool_name}.{action}: {e}")
            
            usage_record = {
                "timestamp": datetime.now().isoformat(),
                "tool": tool_name,
                "action": action,
                "agent_id": agent.agent_id if agent else None,
                "success": False,
                "error": str(e)
            }
            self.usage_history.append(usage_record)
            
            return {"success": False, "error": str(e)}
    
    async def suggest_tool_for_task(self, task_description: str, 
                                   agent: Optional[AgentProfile] = None) -> List[str]:
        task_lower = task_description.lower()
        suggested_tools = []
        
        if any(keyword in task_lower for keyword in ["file", "read", "write", "save", "load"]):
            suggested_tools.append("file")
        
        if any(keyword in task_lower for keyword in ["code", "python", "execute", "run", "script"]):
            suggested_tools.append("python")
        
        if any(keyword in task_lower for keyword in ["search", "find", "lookup", "research"]):
            if "academic" in task_lower or "paper" in task_lower:
                suggested_tools.append("arxiv")
            elif "wikipedia" in task_lower or "encyclopedia" in task_lower:
                suggested_tools.append("wikipedia")
            else:
                suggested_tools.append("google")
        
        if not suggested_tools and agent:
            suggested_tools = await self.get_tools_for_agent(agent)
        
        return suggested_tools[:3]
    
    def create_tool_enabled_agent(self, agent_config: Dict[str, Any], 
                                 tools: List[str]) -> CustomizeAgent:
        agent_tools = []
        for tool_name in tools:
            if tool_name in self.tools:
                agent_tools.append(self.tools[tool_name]["instance"])
        
        agent = CustomizeAgent(
            name=agent_config.get("name", "ToolEnabledAgent"),
            description=agent_config.get("description", "Agent with integrated tools"),
            prompt=agent_config.get("prompt", "Use available tools to complete tasks: {task}"),
            llm_config=agent_config.get("llm_config"),
            inputs=agent_config.get("inputs", [
                {"name": "task", "type": "string", "description": "Task to complete"}
            ]),
            outputs=agent_config.get("outputs", [
                {"name": "result", "type": "string", "description": "Task result"}
            ]),
            tools=agent_tools
        )
        
        return agent
    
    async def enhance_workflow_with_tools(self, workflow: WorkFlowGraph) -> WorkFlowGraph:
        for node in workflow.nodes:
            suggested_tools = await self.suggest_tool_for_task(node.description)
            
            if hasattr(node, 'agents') and node.agents:
                for agent_config in node.agents:
                    if isinstance(agent_config, dict):
                        if "tools" not in agent_config:
                            agent_config["tools"] = suggested_tools
        
        return workflow
    
    async def auto_discover_tools(self):
        if not self.auto_tool_discovery:
            return
        
        try:
            if "mcp" in self.tools:
                mcp_toolkit = self.tools["mcp"]["instance"]
                new_tools = await mcp_toolkit.discover_tools()
                
                for tool_name, tool_info in new_tools.items():
                    if tool_name not in self.tools:
                        self.tools[f"mcp_{tool_name}"] = {
                            "instance": tool_info,
                            "category": ToolCategory.MCP,
                            "capabilities": tool_info.get("capabilities", []),
                            "description": tool_info.get("description", "MCP discovered tool")
                        }
                
                self.logger.info(f"Auto-discovered {len(new_tools)} new MCP tools")
            
        except Exception as e:
            self.logger.error(f"Auto tool discovery error: {e}")
    
    def get_tool_usage_stats(self) -> Dict[str, Any]:
        tool_usage_count = {}
        successful_usage = {}
        
        for record in self.usage_history:
            tool = record["tool"]
            tool_usage_count[tool] = tool_usage_count.get(tool, 0) + 1
            
            if record["success"]:
                successful_usage[tool] = successful_usage.get(tool, 0) + 1
        
        success_rates = {}
        for tool in tool_usage_count:
            success_rates[tool] = successful_usage.get(tool, 0) / tool_usage_count[tool]
        
        return {
            "total_tools": len(self.tools),
            "total_usage": len(self.usage_history),
            "tool_usage_count": tool_usage_count,
            "success_rates": success_rates,
            "most_used_tool": max(tool_usage_count.items(), key=lambda x: x[1])[0] if tool_usage_count else None
        }
    
    async def optimize_tool_usage(self):
        stats = self.get_tool_usage_stats()
        
        underperforming_tools = [
            tool for tool, rate in stats["success_rates"].items() 
            if rate < 0.7
        ]
        
        if underperforming_tools:
            self.logger.warning(f"Underperforming tools detected: {underperforming_tools}")
        
        for agent_id, agent in self.ecosystem.state.agents.items():
            agent_usage = [
                record for record in self.usage_history 
                if record.get("agent_id") == agent_id and record["success"]
            ]
            
            if agent_usage:
                successful_tools = [record["tool"] for record in agent_usage]
                self.agent_preferences[agent_id] = list(set(successful_tools))
    
    def get_integration_status(self) -> Dict[str, Any]:
        return {
            "total_tools": len(self.tools),
            "tool_categories": {cat.value: len(tools) for cat, tools in self.tool_categories.items()},
            "total_usage": len(self.usage_history),
            "agents_with_preferences": len(self.agent_preferences),
            "auto_discovery_enabled": self.auto_tool_discovery,
            "active": True
        }
    
    async def shutdown(self):
        self.logger.info("Shutting down tool integration system...")
        
        try:
            with open("tool_usage_history.json", "w") as f:
                json.dump(self.usage_history, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving tool usage history: {e}")
        
        self.logger.info("Tool integration system shutdown complete")