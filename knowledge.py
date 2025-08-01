import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import numpy as np

from evoagentx.rag.schema import Query, Corpus, Chunk, ChunkMetadata
from evoagentx.rag.rag import RAGEngine
from evoagentx.agents import CustomizeAgent
from systemd import AgentProfile, AgentRole, AgentCapability


class KnowledgeType:
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    EXPERIENTIAL = "experiential"
    COLLABORATIVE = "collaborative"
    EMERGENT = "emergent"


class KnowledgeSynthesis:
    
    def __init__(self, ecosystem):
        self.ecosystem = ecosystem
        self.logger = ecosystem.logger
        self.models = ecosystem.models
        self.rag_engines = ecosystem.rag_engines
        self.knowledge_base = {}
        self.synthesis_history = []
        self.knowledge_networks = defaultdict(list)
        self.synthesis_agents = {}
        self.setup_synthesis_agents()
        self.logger.info("Knowledge synthesis system initialized")
    
    def setup_synthesis_agents(self):
        self.synthesis_agents["factual"] = CustomizeAgent(
            name="FactualSynthesizer",
            description="Synthesizes factual knowledge from multiple sources",
            prompt="""
            Analyze and synthesize factual knowledge from the provided sources:
            
            Sources: {sources}
            
            Create a comprehensive synthesis that:
            1. Identifies key facts and relationships
            2. Resolves conflicts between sources
            3. Highlights gaps in knowledge
            4. Provides confidence ratings
            
            Output format:
            ## synthesis
            [Synthesized knowledge]
            
            ## confidence
            [Confidence score 0-1]
            
            ## gaps
            [Identified knowledge gaps]
            """,
            llm_config=self.ecosystem.model_configs["reasoning"],
            inputs=[
                {"name": "sources", "type": "string", "description": "Knowledge sources to synthesize"}
            ],
            outputs=[
                {"name": "synthesis", "type": "string", "description": "Synthesized knowledge"},
                {"name": "confidence", "type": "string", "description": "Confidence assessment"},
                {"name": "gaps", "type": "string", "description": "Knowledge gaps"}
            ]
        )
        
        self.synthesis_agents["procedural"] = CustomizeAgent(
            name="ProceduralSynthesizer",
            description="Synthesizes procedural knowledge and workflows",
            prompt="""
            Synthesize procedural knowledge from agent experiences and workflows:
            
            Experiences: {experiences}
            
            Create a synthesis that:
            1. Identifies common patterns and procedures
            2. Extracts best practices
            3. Creates generalized workflows
            4. Suggests optimizations
            
            Output format:
            ## procedures
            [Synthesized procedures]
            
            ## patterns
            [Common patterns identified]
            
            ## optimizations
            [Suggested improvements]
            """,
            llm_config=self.ecosystem.model_configs["optimization"],
            inputs=[
                {"name": "experiences", "type": "string", "description": "Agent experiences to analyze"}
            ],
            outputs=[
                {"name": "procedures", "type": "string", "description": "Synthesized procedures"},
                {"name": "patterns", "type": "string", "description": "Identified patterns"},
                {"name": "optimizations", "type": "string", "description": "Optimization suggestions"}
            ]
        )
        
        self.synthesis_agents["collaborative"] = CustomizeAgent(
            name="CollaborativeSynthesizer",
            description="Synthesizes knowledge from agent collaborations",
            prompt="""
            Analyze collaborative interactions and synthesize collective knowledge:
            
            Collaborations: {collaborations}
            
            Extract:
            1. Successful collaboration patterns
            2. Knowledge transfer mechanisms
            3. Collective intelligence insights
            4. Synergy opportunities
            
            Output format:
            ## collaboration_patterns
            [Successful patterns]
            
            ## knowledge_transfer
            [Transfer mechanisms]
            
            ## collective_insights
            [Collective intelligence findings]
            """,
            llm_config=self.ecosystem.model_configs["learning"],
            inputs=[
                {"name": "collaborations", "type": "string", "description": "Collaboration data"}
            ],
            outputs=[
                {"name": "collaboration_patterns", "type": "string", "description": "Collaboration patterns"},
                {"name": "knowledge_transfer", "type": "string", "description": "Transfer mechanisms"},
                {"name": "collective_insights", "type": "string", "description": "Collective insights"}
            ]
        )
    
    async def synthesize_ecosystem_knowledge(self) -> Dict[str, Any]:
        self.logger.info("Starting ecosystem knowledge synthesis")
        knowledge_sources = await self.collect_knowledge_sources()
        synthesis_results = {}
        
        if knowledge_sources.get("factual"):
            factual_result = await self.synthesize_factual_knowledge(
                knowledge_sources["factual"]
            )
            synthesis_results["factual"] = factual_result
        
        if knowledge_sources.get("procedural"):
            procedural_result = await self.synthesize_procedural_knowledge(
                knowledge_sources["procedural"]
            )
            synthesis_results["procedural"] = procedural_result
        
        if knowledge_sources.get("collaborative"):
            collaborative_result = await self.synthesize_collaborative_knowledge(
                knowledge_sources["collaborative"]
            )
            synthesis_results["collaborative"] = collaborative_result
        
        cross_domain_result = await self.synthesize_cross_domain_knowledge(
            synthesis_results
        )
        synthesis_results["cross_domain"] = cross_domain_result
        
        await self.update_knowledge_base(synthesis_results)
        
        synthesis_record = {
            "timestamp": datetime.now().isoformat(),
            "synthesis_types": list(synthesis_results.keys()),
            "knowledge_sources_count": sum(len(sources) for sources in knowledge_sources.values()),
            "synthesis_quality": await self.assess_synthesis_quality(synthesis_results)
        }
        self.synthesis_history.append(synthesis_record)
        
        self.logger.info(f"Knowledge synthesis complete: {len(synthesis_results)} types synthesized")
        return synthesis_results
    
    async def collect_knowledge_sources(self) -> Dict[str, List[Dict[str, Any]]]:
        knowledge_sources = {
            "factual": [],
            "procedural": [],
            "collaborative": [],
            "experiential": []
        }
        
        for agent in self.ecosystem.state.agents.values():
            for performance in agent.performance_history:
                if performance.get("success"):
                    knowledge_sources["procedural"].append({
                        "source": f"agent_{agent.agent_id}",
                        "type": "performance",
                        "content": performance,
                        "timestamp": performance.get("timestamp"),
                        "confidence": 0.8
                    })
            
            for collab_partner_id in agent.collaboration_network:
                if collab_partner_id in self.ecosystem.state.agents:
                    knowledge_sources["collaborative"].append({
                        "source": f"collaboration_{agent.agent_id}_{collab_partner_id}",
                        "type": "collaboration",
                        "content": {
                            "agent1": agent.agent_id,
                            "agent2": collab_partner_id,
                            "domains": list(set(agent.knowledge_domains) & 
                                          set(self.ecosystem.state.agents[collab_partner_id].knowledge_domains))
                        },
                        "timestamp": datetime.now().isoformat(),
                        "confidence": 0.7
                    })
        
        for rag_name, rag_engine in self.rag_engines.items():
            try:
                recent_query = Query(
                    text="recent knowledge and insights",
                    top_k=20,
                    filters={"timestamp": {"$gte": (datetime.now() - timedelta(days=7)).isoformat()}}
                )
                
                rag_results = await rag_engine.retrieve(recent_query)
                
                for result in rag_results:
                    knowledge_sources["factual"].append({
                        "source": f"rag_{rag_name}",
                        "type": "retrieved",
                        "content": result.content,
                        "metadata": result.metadata,
                        "confidence": result.score if hasattr(result, 'score') else 0.6
                    })
                    
            except Exception as e:
                self.logger.error(f"Error collecting from RAG engine {rag_name}: {e}")
        
        if hasattr(self.ecosystem, 'mcp_integration'):
            try:
                mcp_knowledge = await self.ecosystem.mcp_integration.extract_knowledge()
                
                for knowledge_item in mcp_knowledge:
                    knowledge_sources["experiential"].append({
                        "source": "mcp_integration",
                        "type": "tool_usage",
                        "content": knowledge_item,
                        "timestamp": datetime.now().isoformat(),
                        "confidence": 0.6
                    })
                    
            except Exception as e:
                self.logger.error(f"Error collecting MCP knowledge: {e}")
        
        return knowledge_sources
    
    async def synthesize_factual_knowledge(self, factual_sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        sources_text = "\n\n".join([
            f"Source: {source['source']}\nContent: {source['content']}\nConfidence: {source['confidence']}"
            for source in factual_sources[:10]
        ])
        
        result = await self.synthesis_agents["factual"](
            inputs={"sources": sources_text}
        )
        
        return {
            "synthesis": result.content.synthesis,
            "confidence": result.content.confidence,
            "gaps": result.content.gaps,
            "source_count": len(factual_sources),
            "type": "factual"
        }
    
    async def synthesize_procedural_knowledge(self, procedural_sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        experiences_text = "\n\n".join([
            f"Agent: {source['source']}\nExperience: {source['content']}\nSuccess: {source.get('confidence', 0.5)}"
            for source in procedural_sources[:15]
        ])
        
        result = await self.synthesis_agents["procedural"](
            inputs={"experiences": experiences_text}
        )
        
        return {
            "procedures": result.content.procedures,
            "patterns": result.content.patterns,
            "optimizations": result.content.optimizations,
            "source_count": len(procedural_sources),
            "type": "procedural"
        }
    
    async def synthesize_collaborative_knowledge(self, collaborative_sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        collaborations_text = "\n\n".join([
            f"Collaboration: {source['content']['agent1']} <-> {source['content']['agent2']}\n"
            f"Shared domains: {source['content']['domains']}"
            for source in collaborative_sources[:10]
        ])
        
        result = await self.synthesis_agents["collaborative"](
            inputs={"collaborations": collaborations_text}
        )
        
        return {
            "collaboration_patterns": result.content.collaboration_patterns,
            "knowledge_transfer": result.content.knowledge_transfer,
            "collective_insights": result.content.collective_insights,
            "source_count": len(collaborative_sources),
            "type": "collaborative"
        }
    
    async def synthesize_cross_domain_knowledge(self, synthesis_results: Dict[str, Any]) -> Dict[str, Any]:
        cross_domain_prompt = f"""
        Analyze the following synthesized knowledge from different domains and create cross-domain insights:
        
        Factual Knowledge: {synthesis_results.get('factual', {}).get('synthesis', 'None')}
        
        Procedural Knowledge: {synthesis_results.get('procedural', {}).get('procedures', 'None')}
        
        Collaborative Knowledge: {synthesis_results.get('collaborative', {}).get('collective_insights', 'None')}
        
        Identify:
        1. Cross-domain patterns and connections
        2. Emergent insights that span multiple domains
        3. Opportunities for knowledge transfer between domains
        4. Novel combinations and innovations
        
        Provide a comprehensive cross-domain synthesis.
        """
        
        response = await self.models["reasoning"].agenerate(cross_domain_prompt)
        
        return {
            "cross_domain_insights": response.content,
            "domains_analyzed": list(synthesis_results.keys()),
            "type": "cross_domain"
        }
    
    async def update_knowledge_base(self, synthesis_results: Dict[str, Any]):
        timestamp = datetime.now().isoformat()
        
        for synthesis_type, result in synthesis_results.items():
            knowledge_entry = {
                "id": f"synthesis_{synthesis_type}_{timestamp}",
                "type": synthesis_type,
                "content": result,
                "timestamp": timestamp,
                "confidence": self.calculate_synthesis_confidence(result),
                "sources": result.get("source_count", 0)
            }
            
            self.knowledge_base[knowledge_entry["id"]] = knowledge_entry
        
        await self.update_rag_with_synthesis(synthesis_results)
    
    async def update_rag_with_synthesis(self, synthesis_results: Dict[str, Any]):
        for synthesis_type, result in synthesis_results.items():
            chunks = []
            
            if synthesis_type == "factual" and "synthesis" in result:
                chunk = Chunk(
                    content=result["synthesis"],
                    metadata=ChunkMetadata(
                        source="knowledge_synthesis",
                        type="factual_synthesis",
                        timestamp=datetime.now().isoformat(),
                        confidence=result.get("confidence", "0.5")
                    )
                )
                chunks.append(chunk)
            
            elif synthesis_type == "procedural" and "procedures" in result:
                chunk = Chunk(
                    content=result["procedures"],
                    metadata=ChunkMetadata(
                        source="knowledge_synthesis",
                        type="procedural_synthesis",
                        timestamp=datetime.now().isoformat()
                    )
                )
                chunks.append(chunk)
            
            if chunks:
                try:
                    if "semantic" in self.rag_engines:
                        corpus = Corpus(
                            name=f"synthesis_{synthesis_type}",
                            chunks=chunks
                        )
                        await self.rag_engines["semantic"].add_corpus(corpus)
                        
                except Exception as e:
                    self.logger.error(f"Error updating RAG with synthesis: {e}")
    
    def calculate_synthesis_confidence(self, synthesis_result: Dict[str, Any]) -> float:
        base_confidence = 0.7
        source_count = synthesis_result.get("source_count", 1)
        source_factor = min(1.0, source_count / 10.0)
        
        type_factors = {
            "factual": 0.9,
            "procedural": 0.8,
            "collaborative": 0.7,
            "cross_domain": 0.6
        }
        
        synthesis_type = synthesis_result.get("type", "unknown")
        type_factor = type_factors.get(synthesis_type, 0.5)
        
        final_confidence = base_confidence * source_factor * type_factor
        return min(1.0, final_confidence)
    
    async def assess_synthesis_quality(self, synthesis_results: Dict[str, Any]) -> float:
        quality_scores = []
        
        for synthesis_type, result in synthesis_results.items():
            content_length = len(str(result))
            source_count = result.get("source_count", 0)
            quality_score = min(1.0, (content_length / 1000.0) * (source_count / 10.0))
            quality_scores.append(quality_score)
        
        return np.mean(quality_scores) if quality_scores else 0.0
    
    async def query_synthesized_knowledge(self, query: str, 
                                        knowledge_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if knowledge_types is None:
            knowledge_types = ["factual", "procedural", "collaborative", "cross_domain"]
        
        results = []
        
        for knowledge_id, knowledge_entry in self.knowledge_base.items():
            if knowledge_entry["type"] in knowledge_types:
                content_str = str(knowledge_entry["content"]).lower()
                if query.lower() in content_str:
                    results.append({
                        "id": knowledge_id,
                        "type": knowledge_entry["type"],
                        "content": knowledge_entry["content"],
                        "confidence": knowledge_entry["confidence"],
                        "timestamp": knowledge_entry["timestamp"]
                    })
        
        results.sort(key=lambda x: (x["confidence"], x["timestamp"]), reverse=True)
        return results[:10]
    
    def get_synthesis_status(self) -> Dict[str, Any]:
        return {
            "total_knowledge_entries": len(self.knowledge_base),
            "synthesis_agents": len(self.synthesis_agents),
            "synthesis_history": len(self.synthesis_history),
            "knowledge_types": list(set(entry["type"] for entry in self.knowledge_base.values())),
            "recent_synthesis": len([
                h for h in self.synthesis_history 
                if datetime.fromisoformat(h["timestamp"]) > datetime.now() - timedelta(hours=24)
            ]),
            "active": True
        }
    
    async def shutdown(self):
        self.logger.info("Shutting down knowledge synthesis system...")
        
        try:
            with open("knowledge_base.json", "w") as f:
                json.dump(self.knowledge_base, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving knowledge base: {e}")
        
        self.logger.info("Knowledge synthesis system shutdown complete")