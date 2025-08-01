import asyncio
import json
import importlib
import inspect
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Type
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import logging
from pathlib import Path
import sys
import ast
import textwrap
import traceback

sys.path.append(str(Path(__file__).parent))

from systemd import AgentRole, AgentCapability, AgentProfile
from evoagentx.tools.tool import Toolkit
from evoagentx.actions import ActionInput, ActionOutput


class MCPToolStatus(Enum):
    DRAFT = "draft"
    VALIDATING = "validating"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ERROR = "error"
    RETIRED = "retired"


class MCPExecutionMode(Enum):
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    PARALLEL = "parallel"
    DISTRIBUTED = "distributed"


@dataclass
class MCPToolMetadata:
    tool_id: str
    name: str
    description: str
    version: str = "1.0.0"
    author: str = "auto_generated"
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    security_level: int = 1
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    estimated_execution_time: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    success_rate: float = 0.0
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class MCPToolSchema:
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    parameters: Dict[str, Any]
    return_type: str
    error_handling: Dict[str, Any]
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MCPToolDefinition:
    metadata: MCPToolMetadata
    schema: MCPToolSchema
    code: str
    execution_mode: MCPExecutionMode = MCPExecutionMode.SYNCHRONOUS
    test_cases: List[Dict[str, Any]] = field(default_factory=list)
    documentation: str = ""
    examples: List[Dict[str, Any]] = field(default_factory=list)


class MCPValidationError(Exception):
    pass


class MCPExecutionError(Exception):
    pass


class DynamicToolRegistry:
    
    def __init__(self, ecosystem):
        self.ecosystem = ecosystem
        self.logger = ecosystem.logger
        self.tools: Dict[str, MCPToolDefinition] = {}
        self.active_tools: Dict[str, Callable] = {}
        self.tool_categories: Dict[str, List[str]] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.validation_queue: asyncio.Queue = asyncio.Queue()
        self.performance_monitor = PerformanceMonitor()
        
    async def register_tool(self, tool_definition: MCPToolDefinition) -> str:
        try:
            await self.validate_tool_definition(tool_definition)
            
            tool_definition.metadata.status = MCPToolStatus.VALIDATING
            
            tool_id = tool_definition.metadata.tool_id
            self.tools[tool_id] = tool_definition
            
            for tag in tool_definition.metadata.tags:
                if tag not in self.tool_categories:
                    self.tool_categories[tag] = []
                self.tool_categories[tag].append(tool_id)
            
            await self.compile_and_activate_tool(tool_definition)
            
            self.logger.info(f"Registered MCP tool: {tool_definition.metadata.name} ({tool_id})")
            return tool_id
            
        except Exception as e:
            self.logger.error(f"Failed to register tool {tool_definition.metadata.name}: {e}")
            if hasattr(tool_definition.metadata, 'tool_id'):
                tool_definition.metadata.status = MCPToolStatus.ERROR
            raise MCPValidationError(f"Tool registration failed: {e}")
    
    async def validate_tool_definition(self, tool_definition: MCPToolDefinition):
        validation_errors = []
        
        if not tool_definition.metadata.name or not tool_definition.metadata.description:
            validation_errors.append("Tool name and description are required")
        
        try:
            ast.parse(tool_definition.code)
        except SyntaxError as e:
            validation_errors.append(f"Syntax error in tool code: {e}")
        
        if not tool_definition.schema.input_schema or not tool_definition.schema.output_schema:
            validation_errors.append("Input and output schemas are required")
        
        security_issues = await self.security_scan(tool_definition.code)
        if security_issues:
            validation_errors.extend(security_issues)
        
        if validation_errors:
            raise MCPValidationError(f"Tool validation failed: {validation_errors}")
    
    async def security_scan(self, code: str) -> List[str]:
        security_issues = []
        
        dangerous_patterns = [
            "import os",
            "import subprocess",
            "exec(",
            "eval(",
            "__import__",
            "open(",
            "file(",
            "rm ",
            "del ",
            "shutdown",
            "reboot"
        ]
        
        for pattern in dangerous_patterns:
            if pattern in code.lower():
                security_issues.append(f"Potentially dangerous pattern detected: {pattern}")
        
        loop_patterns = ["while True:", "for i in range(1000000):"]
        for pattern in loop_patterns:
            if pattern in code:
                security_issues.append(f"Potential infinite loop detected: {pattern}")
        
        return security_issues
    
    async def compile_and_activate_tool(self, tool_definition: MCPToolDefinition):
        try:
            tool_namespace = {
                '__name__': f'mcp_tool_{tool_definition.metadata.tool_id}',
                '__doc__': tool_definition.metadata.description,
                'logger': self.logger,
                'datetime': datetime,
                'json': json,
                'asyncio': asyncio
            }
            
            exec(tool_definition.code, tool_namespace)
            
            tool_callable = None
            for name, obj in tool_namespace.items():
                if callable(obj) and not name.startswith('_'):
                    tool_callable = obj
                    break
            
            if not tool_callable:
                raise MCPValidationError("No callable function or class found in tool code")
            
            self.active_tools[tool_definition.metadata.tool_id] = tool_callable
            
            tool_definition.metadata.status = MCPToolStatus.ACTIVE
            
            if tool_definition.test_cases:
                await self.run_tool_tests(tool_definition, tool_callable)
            
            self.logger.info(f"Tool activated: {tool_definition.metadata.name}")
            
        except Exception as e:
            tool_definition.metadata.status = MCPToolStatus.ERROR
            raise MCPExecutionError(f"Failed to compile tool: {e}")
    
    async def run_tool_tests(self, tool_definition: MCPToolDefinition, tool_callable: Callable):
        passed_tests = 0
        total_tests = len(tool_definition.test_cases)
        
        for test_case in tool_definition.test_cases:
            try:
                test_input = test_case.get('input', {})
                expected_output = test_case.get('expected_output')
                
                if asyncio.iscoroutinefunction(tool_callable):
                    result = await tool_callable(**test_input)
                else:
                    result = tool_callable(**test_input)
                
                if expected_output:
                    if self.compare_results(result, expected_output):
                        passed_tests += 1
                else:
                    passed_tests += 1
                    
            except Exception as e:
                self.logger.warning(f"Test case failed for {tool_definition.metadata.name}: {e}")
        
        if total_tests > 0:
            tool_definition.metadata.success_rate = passed_tests / total_tests
    
    def compare_results(self, result: Any, expected: Any) -> bool:
        if isinstance(result, dict) and isinstance(expected, dict):
            return all(key in result and result[key] == expected[key] for key in expected)
        return result == expected
    
    async def execute_tool(self, tool_id: str, **kwargs) -> Any:
        if tool_id not in self.active_tools:
            raise MCPExecutionError(f"Tool {tool_id} not found or not active")
        
        tool_definition = self.tools[tool_id]
        start_time = datetime.now()
        
        try:
            await self.validate_input(tool_definition.schema, kwargs)
            
            tool_callable = self.active_tools[tool_id]
            
            if asyncio.iscoroutinefunction(tool_callable):
                result = await tool_callable(**kwargs)
            else:
                result = tool_callable(**kwargs)
            
            await self.validate_output(tool_definition.schema, result)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            await self.update_tool_metrics(tool_id, execution_time, True)
            
            self.record_execution(tool_id, kwargs, result, execution_time, True)
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            await self.update_tool_metrics(tool_id, execution_time, False)
            self.record_execution(tool_id, kwargs, None, execution_time, False, str(e))
            raise MCPExecutionError(f"Tool execution failed: {e}")
    
    async def validate_input(self, schema: MCPToolSchema, input_data: Dict[str, Any]):
        required_params = schema.parameters.get('required', [])
        for param in required_params:
            if param not in input_data:
                raise MCPValidationError(f"Missing required parameter: {param}")
    
    async def validate_output(self, schema: MCPToolSchema, output_data: Any):
        if output_data is None:
            raise MCPValidationError("Tool returned None output")
    
    async def update_tool_metrics(self, tool_id: str, execution_time: float, success: bool):
        if tool_id not in self.tools:
            return
        
        tool = self.tools[tool_id]
        tool.metadata.usage_count += 1
        
        if tool.metadata.usage_count > 1:
            current_successes = tool.metadata.success_rate * (tool.metadata.usage_count - 1)
            new_successes = current_successes + (1 if success else 0)
            tool.metadata.success_rate = new_successes / tool.metadata.usage_count
        else:
            tool.metadata.success_rate = 1.0 if success else 0.0
        
        tool.metadata.performance_metrics['avg_execution_time'] = (
            (tool.metadata.performance_metrics.get('avg_execution_time', 0) * (tool.metadata.usage_count - 1) + execution_time) / 
            tool.metadata.usage_count
        )
        
        tool.metadata.last_modified = datetime.now()
    
    def record_execution(self, tool_id: str, input_data: Dict[str, Any], 
                        output_data: Any, execution_time: float, success: bool, error: str = None):
        execution_record = {
            'tool_id': tool_id,
            'timestamp': datetime.now().isoformat(),
            'input_data': input_data,
            'output_data': str(output_data)[:1000] if output_data else None,
            'execution_time': execution_time,
            'success': success,
            'error': error
        }
        self.execution_history.append(execution_record)
    
    def discover_tools(self, tags: List[str] = None, capability: str = None) -> List[str]:
        matching_tools = []
        
        for tool_id, tool_def in self.tools.items():
            if tool_def.metadata.status != MCPToolStatus.ACTIVE:
                continue
            
            if tags:
                if not any(tag in tool_def.metadata.tags for tag in tags):
                    continue
            
            if capability:
                if capability not in tool_def.metadata.description.lower():
                    continue
            
            matching_tools.append(tool_id)
        
        return matching_tools
    
    def get_tool_info(self, tool_id: str) -> Dict[str, Any]:
        if tool_id not in self.tools:
            return {}
        
        tool = self.tools[tool_id]
        return {
            'metadata': asdict(tool.metadata),
            'schema': asdict(tool.schema),
            'execution_mode': tool.execution_mode.value,
            'documentation': tool.documentation,
            'examples': tool.examples
        }
    
    async def retire_tool(self, tool_id: str):
        if tool_id not in self.tools:
            raise MCPExecutionError(f"Tool {tool_id} not found")
        
        tool = self.tools[tool_id]
        tool.metadata.status = MCPToolStatus.RETIRED
        
        if tool_id in self.active_tools:
            del self.active_tools[tool_id]
        
        self.logger.info(f"Retired tool: {tool.metadata.name}")


class PerformanceMonitor:
    
    def __init__(self):
        self.performance_data = {}
        self.alerts = []
    
    def record_performance(self, tool_id: str, metrics: Dict[str, float]):
        if tool_id not in self.performance_data:
            self.performance_data[tool_id] = []
        
        self.performance_data[tool_id].append({
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        })
    
    def get_performance_summary(self, tool_id: str) -> Dict[str, float]:
        if tool_id not in self.performance_data:
            return {}
        
        data = self.performance_data[tool_id]
        if not data:
            return {}
        
        metrics_sum = {}
        for record in data:
            for key, value in record['metrics'].items():
                if key not in metrics_sum:
                    metrics_sum[key] = []
                metrics_sum[key].append(value)
        
        summary = {}
        for key, values in metrics_sum.items():
            summary[f'avg_{key}'] = sum(values) / len(values)
            summary[f'max_{key}'] = max(values)
            summary[f'min_{key}'] = min(values)
        
        return summary


class MCPToolGenerator:
    
    def __init__(self, ecosystem):
        self.ecosystem = ecosystem
        self.logger = ecosystem.logger
        self.registry = ecosystem.mcp_registry if hasattr(ecosystem, 'mcp_registry') else None
    
    async def generate_tool_from_description(self, description: str, 
                                           requirements: Dict[str, Any] = None) -> MCPToolDefinition:
        
        generation_prompt = f"""
        Create a Python tool function based on this description:
        
        Description: {description}
        
        Requirements: {requirements or {}}
        
        Generate a complete Python function that:
        1. Has clear input parameters with type hints
        2. Includes proper error handling
        3. Returns structured results
        4. Includes docstring documentation
        5. Is safe and secure to execute
        
        Format your response as:
        ```python
        def tool_function_name(param1: type, param2: type) -> return_type:
            \"\"\"Tool description\"\"\"
            # Implementation
            return result
        ```
        """
        
        try:
            response = await self.ecosystem.models["learning"].agenerate(generation_prompt)
            
            code = self.extract_code_from_response(response.content)
            
            metadata = MCPToolMetadata(
                tool_id=str(uuid.uuid4()),
                name=self.generate_tool_name(description),
                description=description,
                tags=self.extract_tags_from_description(description),
                author="auto_generated"
            )
            
            schema = self.generate_schema_from_code(code)
            
            tool_definition = MCPToolDefinition(
                metadata=metadata,
                schema=schema,
                code=code,
                documentation=response.content
            )
            
            return tool_definition
            
        except Exception as e:
            self.logger.error(f"Failed to generate tool from description: {e}")
            raise MCPExecutionError(f"Tool generation failed: {e}")
    
    def extract_code_from_response(self, response: str) -> str:
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            return response[start:end].strip()
        return response.strip()
    
    def generate_tool_name(self, description: str) -> str:
        words = description.split()[:3]
        name = "_".join([word.lower().replace('.', '').replace(',', '') for word in words])
        return f"mcp_{name}"
    
    def extract_tags_from_description(self, description: str) -> List[str]:
        tags = []
        keywords = ["data", "analysis", "calculation", "search", "process", "transform", "validate"]
        
        for keyword in keywords:
            if keyword in description.lower():
                tags.append(keyword)
        
        return tags or ["general"]
    
    def generate_schema_from_code(self, code: str) -> MCPToolSchema:
        return MCPToolSchema(
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            parameters={},
            return_type="Any",
            error_handling={"retry_count": 3, "timeout": 30}
        )


class MCPIntegration:
    
    def __init__(self, ecosystem):
        self.ecosystem = ecosystem
        self.logger = ecosystem.logger
        self.registry = DynamicToolRegistry(ecosystem)
        self.generator = MCPToolGenerator(ecosystem)
        
        ecosystem.mcp_registry = self.registry
        ecosystem.mcp_generator = self.generator
    
    async def create_tool_from_need(self, agent: AgentProfile, need_description: str, 
                                  requirements: Dict[str, Any] = None) -> str:
        
        self.logger.info(f"Agent {agent.name} requesting tool creation for: {need_description}")
        
        tool_definition = await self.generator.generate_tool_from_description(
            need_description, requirements
        )
        
        tool_definition.metadata.author = agent.name
        tool_definition.metadata.tags.extend([agent.role.value])
        
        tool_id = await self.registry.register_tool(tool_definition)
        
        agent.evolution_history.append({
            "event": "tool_creation",
            "timestamp": datetime.now().isoformat(),
            "tool_id": tool_id,
            "tool_name": tool_definition.metadata.name,
            "description": need_description,
            "energy_cost": 0.05
        })
        
        self.logger.info(f"Created tool {tool_definition.metadata.name} for agent {agent.name}")
        
        return tool_id
    
    async def discover_and_use_tools(self, agent: AgentProfile, task: str) -> Dict[str, Any]:
        
        relevant_tools = self.registry.discover_tools(
            tags=[agent.role.value], 
            capability=task
        )
        
        results = {}
        
        for tool_id in relevant_tools[:3]:
            try:
                tool_info = self.registry.get_tool_info(tool_id)
                
                result = await self.registry.execute_tool(
                    tool_id, 
                    task=task, 
                    agent_context={
                        "agent_id": agent.agent_id,
                        "agent_role": agent.role.value,
                        "agent_capabilities": {cap.value: score for cap, score in agent.capabilities.items()}
                    }
                )
                
                results[tool_id] = {
                    "success": True,
                    "result": result,
                    "tool_info": tool_info
                }
                
            except Exception as e:
                results[tool_id] = {
                    "success": False,
                    "error": str(e),
                    "tool_info": tool_info
                }
        
        return results
    
    async def optimize_tool_ecosystem(self):
        
        usage_stats = {}
        for tool_id, tool_def in self.registry.tools.items():
            usage_stats[tool_id] = {
                "usage_count": tool_def.metadata.usage_count,
                "success_rate": tool_def.metadata.success_rate,
                "avg_execution_time": tool_def.metadata.performance_metrics.get('avg_execution_time', 0)
            }
        
        underperforming = [
            tool_id for tool_id, stats in usage_stats.items()
            if stats["success_rate"] < 0.5 or stats["usage_count"] == 0
        ]
        
        for tool_id in underperforming:
            await self.registry.retire_tool(tool_id)
            self.logger.info(f"Retired underperforming tool: {tool_id}")
        
        await self.generate_common_tools()
    
    async def generate_common_tools(self):
        
        common_tool_descriptions = [
            "A tool to validate data formats and schemas",
            "A tool to transform and clean data",
            "A tool to analyze text content and extract insights",
            "A tool to perform mathematical calculations",
            "A tool to search and filter information",
            "A tool to generate summaries and reports",
            "A tool to optimize and improve code",
            "A tool to validate and test functions"
        ]
        
        for description in common_tool_descriptions:
            existing_tools = self.registry.discover_tools(tags=["general"])
            similar_exists = any(
                description.lower() in self.registry.tools[tool_id].metadata.description.lower()
                for tool_id in existing_tools
            )
            
            if not similar_exists:
                try:
                    tool_definition = await self.generator.generate_tool_from_description(description)
                    await self.registry.register_tool(tool_definition)
                    self.logger.info(f"Generated common tool: {tool_definition.metadata.name}")
                except Exception as e:
                    self.logger.warning(f"Failed to generate common tool: {e}")
    
    def get_mcp_status(self) -> Dict[str, Any]:
        return {
            "total_tools": len(self.registry.tools),
            "active_tools": len(self.registry.active_tools),
            "tool_categories": self.registry.tool_categories,
            "total_executions": len(self.registry.execution_history),
            "recent_creations": len([
                t for t in self.registry.tools.values()
                if (datetime.now() - t.metadata.created_at).total_seconds() < 3600
            ])
        }