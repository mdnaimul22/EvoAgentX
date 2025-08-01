import os
import sys
import json
import asyncio
import logging
import traceback
import signal
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Callable, Tuple, Union
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import psutil
import sqlite3
import aiohttp
from aiohttp import web
import yaml
from pydantic import BaseModel, Field, field_validator
import uvloop

sys.path.insert(0, str(Path(__file__).parent))

from evoagentx.models import (
    BaseLLM, OpenAILLM, OpenAILLMConfig, LiteLLM, LiteLLMConfig,
    OpenRouterLLM, OpenRouterConfig, create_llm_instance
)
from evoagentx.agents import Agent, CustomizeAgent, AgentManager
from evoagentx.tools import (
    FileToolkit, PythonInterpreterToolkit, WikipediaSearchToolkit,
    GoogleFreeSearchToolkit, ArxivToolkit, MCPToolkit
)
# from evoagentx.utils.mipro_utils.register_utils import MiproRegistry
from evoagentx.storages import StorageHandler
from evoagentx.storages.storages_config import (
    VectorStoreConfig, DBConfig, GraphStoreConfig, StoreConfig
)
from evoagentx.hitl import (
    HITLInteractionType,
    HITLMode,
    HITLManager,
    HITLUserInputCollectorAgent,
    HITLOutsideConversationAgent
)


from evoagentx.rag.rag import RAGEngine
from evoagentx.rag.rag_config import RAGConfig, ReaderConfig, ChunkerConfig, IndexConfig, EmbeddingConfig, RetrievalConfig
from evoagentx.rag.schema import Query, Corpus, Chunk, ChunkMetadata

from evoagentx.core import Message, extract_code_blocks, suppress_logger_info, logger, register_parse_function
from evoagentx.actions import ActionInput, ActionOutput, Action, CodeExtraction, CodeVerification
from evoagentx.prompts import StringTemplate, ChatTemplate, MiproPromptTemplate

from evoagentx.workflow import (
    WorkFlowNode, WorkFlowEdge, WorkFlow, SequentialWorkFlowGraph, WorkFlowGraph,
    QAActionGraph, SEWWorkFlowGraph, WorkFlowGenerator
)
from evoagentx.optimizers import (
    SEWOptimizer, AFlowOptimizer, MiproOptimizer, TextGradOptimizer, WorkFlowMiproOptimizer
)
from evoagentx.evaluators import Evaluator
from evoagentx.benchmark import (
    Benchmark, HotPotQA, HumanEval, MATH, MBPP, AFlowHumanEval, AFlowMBPP
)

import evoagentx.workflow.operators as operator

class SimpleNumpy:
    @staticmethod
    def mean(values):
        return sum(values) / len(values) if values else 0
    
    @staticmethod
    def std(values):
        if not values:
            return 0
        mean_val = SimpleNumpy.mean(values)
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    @staticmethod
    def log(x):
        import math
        return math.log(x)

np = SimpleNumpy()

class PConfig(BaseModel):
    
    ecosystem_name: str = Field(default="ProductionEcosystem", description="Name of the ecosystem")
    base_path: str = Field(default="./example_ecosystem", description="Base path for ecosystem data")
    log_level: str = Field(default="INFO", description="Logging level")
    max_workers: int = Field(default=10, description="Maximum worker threads")
    
    min_agent_count: int = Field(default=5, ge=1, description="Minimum number of agents")
    max_agent_count: int = Field(default=100, le=1000, description="Maximum number of agents")
    initial_agent_count: int = Field(default=10, description="Initial number of agents")
    
    evolution_interval_hours: int = Field(default=24, ge=1, description="Hours between evolution cycles")
    knowledge_synthesis_interval_hours: int = Field(default=6, ge=1, description="Hours between knowledge synthesis")
    self_evaluation_interval_hours: int = Field(default=12, ge=1, description="Hours between self-evaluations")
    health_check_interval_seconds: int = Field(default=60, ge=10, description="Seconds between health checks")
    
    max_cpu_usage: float = Field(default=80.0, ge=10.0, le=100.0, description="Maximum CPU usage percentage")
    max_memory_usage: float = Field(default=80.0, ge=10.0, le=100.0, description="Maximum memory usage percentage")
    max_storage_usage: float = Field(default=90.0, ge=10.0, le=100.0, description="Maximum storage usage percentage")
    
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    openrouter_api_key: Optional[str] = Field(default=None, description="OpenRouter API key")
    
    enable_dashboard: bool = Field(default=True, description="Enable web dashboard")
    dashboard_port: int = Field(default=8080, ge=1024, le=65535, description="Dashboard port")
    dashboard_host: str = Field(default="0.0.0.0", description="Dashboard host")
    
    database_url: str = Field(default="sqlite:///ecosystem.db", description="Database URL")
    enable_backup: bool = Field(default=True, description="Enable automatic backups")
    backup_interval_hours: int = Field(default=6, ge=1, description="Hours between backups")
    
    @field_validator('openai_api_key', 'anthropic_api_key', 'openrouter_api_key', mode='before')
    @classmethod
    def get_api_key_from_env(cls, v, info):
        if v is None:
            env_var = f"{info.field_name.upper()}"
            return os.getenv(env_var)
        return v

class AgentRole(Enum):
    QUALITY_ASSURANCE = "Quality_assurance"
    COORDINATION = "Coordination"
    ANALYTICAL_PROCESSING = "Analytical_Processing"
    CREATIVE_SYNTHESIS = "Creative_Synthesis"
    KNOWLEDGE_ACQUISITION = "knowledge_acquisition"
    PROBLEM_SOLVING = "problem_solving"
    CODE_GENERATION = "code_generation"
    SYSTEM_OPTIMIZATION = "system_optimization"
    META_LEARNING = "meta_learning"
    ETHICAL_OVERSIGHT = "ethical_oversight"
    RESOURCE_MANAGEMENT = "resource_management"
    COLLABORATION_COORDINATION = "collaboration_coordination"
    EMERGENT_GOAL_GENERATION = "emergent_goal_generation"
    SELF_IMPROVEMENT = "self_improvement"
    MONITORING_ANALYSIS = "monitoring_analysis"
    SECURITY_AUDIT = "security_audit"

class AgentCapability(str, Enum):
    REASONING = "reasoning"
    CREATIVITY = "creativity"
    LEARNING = "learning"
    ADAPTATION = "adaptation"
    COLLABORATION = "collaboration"
    SELF_AWARENESS = "self_awareness"
    META_COGNITION = "meta_cognition"
    ABSTRACT_THINKING = "abstract_thinking"
    SYSTEMS_THINKING = "systems_thinking"
    ETHICAL_REASONING = "ethical_reasoning"
    PROBLEM_DECOMPOSITION = "problem_decomposition"
    PATTERN_RECOGNITION = "pattern_recognition"
    CURIOSITY = "curiosity"

@dataclass
class AgentProfile:
    agent_id: str
    name: str
    role: AgentRole
    capabilities: Dict[AgentCapability, float]
    knowledge_domains: List[str]
    performance_history: List[Dict[str, Any]]
    collaboration_network: List[str]
    evolution_history: List[Dict[str, Any]]
    created_at: datetime
    last_active: datetime
    specialization_score: float = 0.0
    generalization_score: float = 0.0
    autonomy_level: float = 0.0
    self_improvement_rate: float = 0.01
    trust_score: float = 0.5
    energy_level: float = 1.0
    
    def update_capability(self, capability: AgentCapability, score: float):
        score = max(0.0, min(1.0, score))
        self.capabilities[capability] = score
        self.last_active = datetime.now()
        
    def calculate_intelligence_quotient(self) -> float:
        if not self.capabilities:
            return 0.0
        return sum(self.capabilities.values()) / len(self.capabilities)
    
    def get_dominant_capabilities(self, top_k: int = 3) -> List[Tuple[AgentCapability, float]]:
        sorted_caps = sorted(self.capabilities.items(), key=lambda x: x[1], reverse=True)
        return sorted_caps[:top_k]
    
    def calculate_specialization_score(self) -> float:
        if not self.capabilities:
            return 0.0
        
        values = list(self.capabilities.values())
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        self.specialization_score = min(1.0, std_val / 0.5) if mean_val > 0 else 0.0
        return self.specialization_score

@dataclass
class EcosystemState:
    agents: Dict[str, AgentProfile] = field(default_factory=dict)
    collective_knowledge: Dict[str, Any] = field(default_factory=dict)
    system_goals: List[Dict[str, Any]] = field(default_factory=list)
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    ethical_constraints: List[Dict[str, Any]] = field(default_factory=list)
    emergency_protocols: List[Dict[str, Any]] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    last_evolution: datetime = field(default_factory=datetime.now)
    last_knowledge_synthesis: datetime = field(default_factory=datetime.now)
    last_self_evaluation: datetime = field(default_factory=datetime.now)
    last_backup: datetime = field(default_factory=datetime.now)
    system_intelligence_score: float = 0.1
    autonomy_level: float = 0.1
    self_preservation_instinct: float = 0.1
    collective_creativity_score: float = 0.1
    system_health_score: float = 1.0
    total_tasks_completed: int = 0
    total_knowledge_items: int = 0
    emergency_count: int = 0
    uptime_hours: float = 0.0

class ProductionLogger:
    
    def __init__(self, config: PConfig):
        self.config = config
        self.setup_logging()
    
    def setup_logging(self):
        log_path = Path(self.config.base_path) / "logs"
        log_path.mkdir(parents=True, exist_ok=True)
        
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        self.logger = logging.getLogger("ProductionEcosystem")
        self.logger.setLevel(getattr(logging, self.config.log_level))
        
        handlers = [
            logging.FileHandler(log_path / "ecosystem.log"),
            logging.FileHandler(log_path / "errors.log"),
            logging.StreamHandler(sys.stdout)
        ]
        
        for i, handler in enumerate(handlers):
            if i == 1:
                handler.setLevel(logging.ERROR)
            
            if isinstance(handler, logging.FileHandler):
                handler.setFormatter(detailed_formatter)
            else:
                handler.setFormatter(simple_formatter)
            self.logger.addHandler(handler)
        
        suppress_logger_info()
        
        self.logger.info("Production logging system initialized")

class ResourceMonitor:
    
    def __init__(self, config: PConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.process = psutil.Process()
        self.monitoring = True
        self.alerts_sent = set()
    
    def get_system_resources(self) -> Dict[str, float]:
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            process_memory = self.process.memory_info().rss / 1024 / 1024
            process_cpu = self.process.cpu_percent()
            
            return {
                "system_cpu": cpu_percent,
                "system_memory": memory.percent,
                "system_disk": disk.percent,
                "process_memory_mb": process_memory,
                "process_cpu": process_cpu,
                "available_memory_gb": memory.available / 1024 / 1024 / 1024
            }
        except Exception as e:
            self.logger.error(f"Error getting system resources: {e}")
            return {}
    
    def check_resource_limits(self, resources: Dict[str, float]) -> List[str]:
        alerts = []
        
        if resources.get("system_cpu", 0) > self.config.max_cpu_usage:
            alerts.append(f"High CPU usage: {resources['system_cpu']:.1f}%")
        
        if resources.get("system_memory", 0) > self.config.max_memory_usage:
            alerts.append(f"High memory usage: {resources['system_memory']:.1f}%")
        
        if resources.get("system_disk", 0) > self.config.max_storage_usage:
            alerts.append(f"High disk usage: {resources['system_disk']:.1f}%")
        
        return alerts
    
    async def monitor_resources(self, ecosystem):
        while self.monitoring:
            try:
                resources = self.get_system_resources()
                alerts = self.check_resource_limits(resources)
                
                ecosystem.state.resource_usage.update(resources)
                
                for alert in alerts:
                    if alert not in self.alerts_sent:
                        self.logger.warning(f"Resource Alert: {alert}")
                        await ecosystem.handle_emergency("resource_limit", alert)
                        self.alerts_sent.add(alert)
                
                if not alerts:
                    self.alerts_sent.clear()
                
                await asyncio.sleep(self.config.health_check_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {e}")
                await asyncio.sleep(60)

class DatabaseManager:
    
    def __init__(self, config: PConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.db_path = Path(config.base_path) / "ecosystem.db"
        self.backup_path = Path(config.base_path) / "backups"
        self.backup_path.mkdir(parents=True, exist_ok=True)
        self.setup_database()
    
    def setup_database(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS agents (
                        agent_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        role TEXT NOT NULL,
                        capabilities TEXT NOT NULL,
                        knowledge_domains TEXT NOT NULL,
                        performance_history TEXT,
                        collaboration_network TEXT,
                        evolution_history TEXT,
                        created_at TIMESTAMP NOT NULL,
                        last_active TIMESTAMP NOT NULL,
                        specialization_score REAL DEFAULT 0.0,
                        generalization_score REAL DEFAULT 0.0,
                        autonomy_level REAL DEFAULT 0.0,
                        self_improvement_rate REAL DEFAULT 0.01,
                        trust_score REAL DEFAULT 0.5,
                        energy_level REAL DEFAULT 1.0
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS system_state (
                        id INTEGER PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        collective_knowledge TEXT,
                        system_goals TEXT,
                        evolution_history TEXT,
                        performance_metrics TEXT,
                        resource_usage TEXT,
                        system_intelligence_score REAL,
                        autonomy_level REAL,
                        self_preservation_instinct REAL,
                        collective_creativity_score REAL,
                        system_health_score REAL,
                        total_tasks_completed INTEGER,
                        total_knowledge_items INTEGER,
                        emergency_count INTEGER,
                        uptime_hours REAL
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TIMESTAMP NOT NULL,
                        event_type TEXT NOT NULL,
                        event_data TEXT NOT NULL,
                        agent_id TEXT,
                        severity TEXT DEFAULT 'INFO'
                    )
                """)
                
                conn.commit()
                self.logger.info("Database schema initialized")
                
        except Exception as e:
            self.logger.error(f"Error setting up database: {e}")
            raise
    
    def save_agent(self, agent: AgentProfile):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO agents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    agent.agent_id, agent.name, agent.role.value,
                    json.dumps(agent.capabilities, default=self._serialize_for_json),
                    json.dumps(agent.knowledge_domains),
                    json.dumps(agent.performance_history),
                    json.dumps(agent.collaboration_network),
                    json.dumps(agent.evolution_history),
                    agent.created_at, agent.last_active,
                    agent.specialization_score, agent.generalization_score,
                    agent.autonomy_level, agent.self_improvement_rate,
                    agent.trust_score, agent.energy_level
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error saving agent {agent.agent_id}: {e}")
    
    def load_agents(self) -> Dict[str, AgentProfile]:
        agents = {}
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM agents")
                rows = cursor.fetchall()
                
                for row in rows:
                    try:
                        capabilities_data = json.loads(row[3])
                        capabilities = {}
                        for cap_key, score in capabilities_data.items():
                            try:
                                if isinstance(cap_key, str):
                                    capability = AgentCapability(cap_key)
                                    capabilities[capability] = score
                                else:
                                    capabilities[cap_key] = score
                            except ValueError as e:
                                self.logger.warning(f"Unknown capability '{cap_key}' for agent {row[0]}: {e}")
                            except Exception as e:
                                self.logger.warning(f"Error processing capability '{cap_key}' for agent {row[0]}: {e}")
                    except Exception as e:
                        self.logger.error(f"Error loading capabilities for agent {row[0]}: {e}")
                        capabilities = {}
                    
                    agent = AgentProfile(
                        agent_id=row[0], name=row[1], role=AgentRole(row[2]),
                        capabilities=capabilities,
                        knowledge_domains=json.loads(row[4]),
                        performance_history=json.loads(row[5]) if row[5] else [],
                        collaboration_network=json.loads(row[6]) if row[6] else [],
                        evolution_history=json.loads(row[7]) if row[7] else [],
                        created_at=datetime.fromisoformat(row[8]),
                        last_active=datetime.fromisoformat(row[9]),
                        specialization_score=row[10], generalization_score=row[11],
                        autonomy_level=row[12], self_improvement_rate=row[13],
                        trust_score=row[14], energy_level=row[15]
                    )
                    agents[agent.agent_id] = agent
                    
        except Exception as e:
            self.logger.error(f"Error loading agents: {e}")
            
        return agents
    
    def save_system_state(self, state: EcosystemState):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO system_state VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now(), json.dumps(state.collective_knowledge),
                    json.dumps(state.system_goals), json.dumps(state.evolution_history),
                    json.dumps(state.performance_metrics), json.dumps(state.resource_usage),
                    state.system_intelligence_score, state.autonomy_level,
                    state.self_preservation_instinct, state.collective_creativity_score,
                    state.system_health_score, state.total_tasks_completed,
                    state.total_knowledge_items, state.emergency_count, state.uptime_hours
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error saving system state: {e}")
    
    def _serialize_for_json(self, obj):
        if isinstance(obj, (AgentRole, AgentCapability)):
            return obj.value
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)
    
    def log_event(self, event_type: str, event_data: Dict[str, Any], 
                  agent_id: Optional[str] = None, severity: str = "INFO"):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                serialized_event_data = json.dumps(event_data, default=self._serialize_for_json)
                cursor.execute("""
                    INSERT INTO events (timestamp, event_type, event_data, agent_id, severity)
                    VALUES (?, ?, ?, ?, ?)
                """, (datetime.now(), event_type, serialized_event_data, agent_id, severity))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error logging event: {e}")
    
    def create_backup(self):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_path / f"ecosystem_backup_{timestamp}.db"
            
            import shutil
            shutil.copy2(self.db_path, backup_file)
            
            self.logger.info(f"Database backup created: {backup_file}")
            
            backups = sorted(self.backup_path.glob("ecosystem_backup_*.db"))
            for old_backup in backups[:-5]:
                old_backup.unlink()
                
        except Exception as e:
            self.logger.error(f"Error creating backup: {e}")