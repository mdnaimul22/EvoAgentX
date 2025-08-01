import asyncio
import sys
import os
import signal
import uvloop
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from systemd import PConfig
from ecosystem import ProductionEcosystem
from operations import AgentOperations
from evolution import EcosystemEvolution
from web_dashboard import WebDashboard

class ProductionEcosystemManager:
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.ecosystem = None
        self.dashboard = None
        self.running = False
        
    async def initialize(self):
        print("🚀 Initializing Production AI Agent Ecosystem...")
        
        self.ecosystem = ProductionEcosystem(self.config_path)
        
        self.ecosystem.agent_ops = AgentOperations(self.ecosystem)
        
        self.ecosystem.evolution_system = EcosystemEvolution(self.ecosystem)
        
        from mcp_integration import MCPIntegration
        self.ecosystem.mcp_integration = MCPIntegration(self.ecosystem)
        
        if self.ecosystem.config.enable_dashboard:
            self.dashboard = WebDashboard(self.ecosystem)
            await self.dashboard.start_dashboard()
        
        self.ecosystem.ecosystem_evolution_cycle = self.ecosystem.evolution_system.ecosystem_evolution_cycle
        self.ecosystem.knowledge_synthesis_cycle = self.ecosystem.evolution_system.knowledge_synthesis_cycle
        
        self.ecosystem.agent_self_improvement = self.ecosystem.agent_ops.agent_self_improvement
        self.ecosystem.agent_collaboration = self.ecosystem.agent_ops.agent_collaboration
        self.ecosystem.process_agent_cycle = self.ecosystem.agent_ops.process_agent_cycle
        
        await self.initialize_advanced_systems()
        
        await self.initialize_integrated_systems()
        
        print("✅ Ecosystem initialization complete!")
    
    async def initialize_advanced_systems(self):
        try:
            await self.ecosystem.mcp_integration.generate_common_tools()
            
            if len(self.ecosystem.state.agents) > 0:
                sample_agent = list(self.ecosystem.state.agents.values())[0]
                await self.ecosystem.benchmarking_system.run_adaptive_benchmark(sample_agent)
            
            print("🔧 Advanced systems (MCP & Benchmarking) initialized")
            
        except Exception as e:
            print(f"⚠️  Advanced systems initialization warning: {e}")
    
    async def initialize_integrated_systems(self):
        """Initialize advanced integrated systems with proper connections"""
        try:
            # Set agent operations reference in ecosystem
            self.ecosystem.agent_ops = self.ecosystem.agent_ops
            
            # Connect all subsystems
            self.ecosystem._connect_subsystems()
            
            # Initialize HITL system
            if hasattr(self.ecosystem, 'hitl_system'):
                await self.ecosystem.hitl_system.initialize()
                print("HITL system initialized")
            
            # Initialize optimization system
            if hasattr(self.ecosystem, 'optimization_system'):
                await self.ecosystem.optimization_system.initialize()
                print("Optimization system initialized")
            
            # Initialize tool integration with auto-discovery
            if hasattr(self.ecosystem, 'tool_integration'):
                await self.ecosystem.tool_integration.auto_discover_tools()
                print("Tool integration initialized")
            
            # Initialize knowledge synthesis
            if hasattr(self.ecosystem, 'knowledge_synthesis'):
                await self.ecosystem.knowledge_synthesis.initialize()
                # Start knowledge synthesis cycle
                asyncio.create_task(self.ecosystem.run_knowledge_synthesis_cycle())
                print("Knowledge synthesis initialized and cycle started")
            
            # Initialize evaluation pipeline
            if hasattr(self.ecosystem, 'evaluation_pipeline'):
                await self.ecosystem.evaluation_pipeline.initialize()
                print("Evaluation pipeline initialized")
            
            # Start integrated system monitoring
            asyncio.create_task(self.ecosystem.monitor_integrated_systems())
            
            print("All integrated systems initialized and connected successfully")
            
        except Exception as e:
            print(f"Error initializing integrated systems: {e}")
            raise
    
    async def start(self):
        if not self.ecosystem:
            await self.initialize()
        
        self.running = True
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║                 🤖 PRODUCTION AI ECOSYSTEM                   ║
║                      SYSTEM ONLINE                           ║
╠══════════════════════════════════════════════════════════════╣
║ Ecosystem: {self.ecosystem.config.ecosystem_name:<45} ║
║ Agents: {len(self.ecosystem.state.agents):<50} ║
║ Dashboard: {'http://localhost:' + str(self.ecosystem.config.dashboard_port) if self.ecosystem.config.enable_dashboard else 'Disabled':<45} ║
║ Intelligence: {self.ecosystem.state.system_intelligence_score:.3f}/<1.000>                                    ║
║ Health: {self.ecosystem.state.system_health_score:.3f}/<1.000>                                        ║
║ MCP Tools: {getattr(self.ecosystem.mcp_integration, 'get_mcp_status', lambda: {'total_tools': 0})().get('total_tools', 0):<48} ║
║ Benchmarks: {len(getattr(self.ecosystem.benchmarking_system, 'results', [])):<49} ║
║ HITL System: {'Active' if hasattr(self.ecosystem, 'hitl_system') else 'Inactive':<46} ║
║ Tool Integration: {len(getattr(self.ecosystem.tool_integration, 'tools', {})) if hasattr(self.ecosystem, 'tool_integration') else 0:<43} ║
║ Knowledge Base: {len(getattr(self.ecosystem.knowledge_synthesis, 'knowledge_base', {})) if hasattr(self.ecosystem, 'knowledge_synthesis') else 0:<44} ║
╚══════════════════════════════════════════════════════════════╝
        """)
        
        try:
            await self.ecosystem.start()
        except KeyboardInterrupt:
            print("\n🛑 Shutdown signal received...")
            await self.shutdown()
        except Exception as e:
            print(f"❌ Critical error: {e}")
            await self.shutdown()
    
    async def shutdown(self):
        if not self.running:
            return
            
        print("🔄 Initiating graceful shutdown...")
        self.running = False
        
        if self.ecosystem:
            await self.ecosystem.shutdown()
        
        print("✅ Ecosystem shutdown complete")
    
    def setup_signal_handlers(self):
        def signal_handler(signum, frame):
            print(f"\n📡 Received signal {signum}")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

async def main():
    uvloop.install()
    
    import argparse
    parser = argparse.ArgumentParser(description='Production AI Agent Ecosystem')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--ecosystem-name', type=str, default='ProductionEcosystem', help='Ecosystem name')
    parser.add_argument('--dashboard-port', type=int, default=8080, help='Dashboard port')
    parser.add_argument('--min-agents', type=int, default=5, help='Minimum number of agents')
    parser.add_argument('--max-agents', type=int, default=50, help='Maximum number of agents')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level')
    parser.add_argument('--index-dir', type=str, help='Directory to index for codebase search')
    
    args = parser.parse_args()
    
    manager = ProductionEcosystemManager(args.config)
    
    if not args.config:
        config = PConfig(
            ecosystem_name=args.ecosystem_name,
            dashboard_port=args.dashboard_port,
            min_agent_count=args.min_agents,
            max_agent_count=args.max_agents,
            log_level=args.log_level
        )
        manager.ecosystem = ProductionEcosystem(config)
    
    manager.setup_signal_handlers()
    
    if args.index_dir:
        await manager.initialize()
        await manager.ecosystem.index_codebase(args.index_dir)
        print(f"✅ Codebase indexing complete for directory: {args.index_dir}")
        return
        
    await manager.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)